"""
model.py — ST-GCN adapted for presentation scoring (regression, not classification)

Architecture:
    Input: (batch, C=3, T=150, V=n_joints)
    → 9 ST-GCN blocks (spatial-temporal graph convolutions)
    → Global average pooling → (batch, 256)
    → Dropout
    → Linear → (batch, 5)  ← your 5 presentation scores

The graph adjacency matrix encodes human skeleton topology:
    Each joint is a node. Each bone is an edge.
    We use the "distance partitioning" strategy from the original paper:
    edges split into: self-loops, centripetal (toward body center), centrifugal (away).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ------------------------------------------------------------------ #
#  Adjacency matrix builder                                           #
# ------------------------------------------------------------------ #

def build_adjacency(n_joints, edges, strategy="distance"):
    """
    Build normalized adjacency matrix for a skeleton graph.

    Args:
        n_joints: total number of joints (V)
        edges:    list of (i, j) tuples — connected joint pairs
        strategy: "uniform" | "distance" (distance partitioning from ST-GCN paper)

    Returns:
        A: numpy array (K, V, V) — K adjacency matrices stacked
           K=1 for uniform, K=3 for distance partitioning
    """
    if strategy == "uniform":
        A = np.zeros((1, n_joints, n_joints), dtype=np.float32)
        for i, j in edges:
            A[0, i, j] = 1
            A[0, j, i] = 1
        # Add self-loops
        for i in range(n_joints):
            A[0, i, i] = 1
        # Normalize rows
        D = A[0].sum(axis=1)
        D_inv = np.where(D > 0, 1.0 / D, 0)
        A[0] = np.diag(D_inv) @ A[0]
        return A

    # Distance partitioning: 3 matrices
    # A[0] = self-connections
    # A[1] = centripetal (toward root/center)
    # A[2] = centrifugal (away from root/center)

    # Build hop distance matrix from edges
    hop = np.full((n_joints, n_joints), np.inf)
    np.fill_diagonal(hop, 0)
    for i, j in edges:
        hop[i, j] = 1
        hop[j, i] = 1
    # Floyd-Warshall to get all-pairs shortest paths
    for k in range(n_joints):
        for i in range(n_joints):
            for j in range(n_joints):
                if hop[i, k] + hop[k, j] < hop[i, j]:
                    hop[i, j] = hop[i, k] + hop[k, j]

    # Root = joint 0 (typically pelvis/center)
    root = 0
    d_root = hop[root]

    A = np.zeros((3, n_joints, n_joints), dtype=np.float32)
    for i in range(n_joints):
        for j in range(n_joints):
            if i == j:
                A[0, i, j] = 1                    # self
            elif hop[i, j] == 1:
                if d_root[j] == d_root[i] - 1:
                    A[1, i, j] = 1                # centripetal
                else:
                    A[2, i, j] = 1                # centrifugal

    # Normalize each matrix
    for k in range(3):
        D = A[k].sum(axis=1)
        D_inv = np.where(D > 0, 1.0 / D, 0)
        A[k] = np.diag(D_inv) @ A[k]

    return A


def body25_edges():
    """
    OpenPose Body25 joint connections.
    Joint indices: https://github.com/CMU-Perceptual-Computing-Lab/openpose
    """
    return [
        (0, 1),   # nose → neck
        (1, 2),   # neck → right shoulder
        (2, 3),   # right shoulder → right elbow
        (3, 4),   # right elbow → right wrist
        (1, 5),   # neck → left shoulder
        (5, 6),   # left shoulder → left elbow
        (6, 7),   # left elbow → left wrist
        (1, 8),   # neck → mid hip
        (8, 9),   # mid hip → right hip
        (9, 10),  # right hip → right knee
        (10, 11), # right knee → right ankle
        (8, 12),  # mid hip → left hip
        (12, 13), # left hip → left knee
        (13, 14), # left knee → left ankle
        (0, 15),  # nose → right eye
        (15, 17), # right eye → right ear
        (0, 16),  # nose → left eye
        (16, 18), # left eye → left ear
    ]


# ------------------------------------------------------------------ #
#  ST-GCN building blocks                                             #
# ------------------------------------------------------------------ #

class GraphConv(nn.Module):
    """
    Spatial graph convolution: aggregates features from neighboring joints.
    For each of K adjacency matrices, applies a separate linear transform
    then sums the results.
    """
    def __init__(self, in_ch, out_ch, K):
        super().__init__()
        self.K = K
        self.conv = nn.Conv2d(in_ch, out_ch * K, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x, A):
        """
        x: (B, in_ch, T, V)
        A: (K, V, V)
        """
        B, C, T, V = x.shape
        out = self.conv(x)                          # (B, out_ch*K, T, V)
        out = out.view(B, self.K, -1, T, V)         # (B, K, out_ch, T, V)

        # For each K: multiply V dimension by adjacency matrix
        A = A.to(x.device)
        out = torch.einsum('bkctv,kvw->bctw', out, A)  # (B, out_ch, T, V)

        return self.bn(out)


class STGCNBlock(nn.Module):
    """
    One ST-GCN block = spatial graph conv + temporal conv + residual.
    """
    def __init__(self, in_ch, out_ch, K, stride=1, dropout=0.0):
        super().__init__()

        self.gcn = GraphConv(in_ch, out_ch, K)

        # Temporal convolution: kernel spans 9 frames
        padding = 4  # (kernel_size - 1) // 2
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=(9, 1),
                      stride=(stride, 1), padding=(padding, 0)),
            nn.BatchNorm2d(out_ch),
            nn.Dropout(dropout),
        )

        # Residual connection
        if in_ch != out_ch or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.residual = nn.Identity()

        self.relu = nn.ReLU()

    def forward(self, x, A):
        res = self.residual(x)
        x = self.gcn(x, A)
        x = self.tcn(x)
        return self.relu(x + res)


# ------------------------------------------------------------------ #
#  Full ST-GCN model                                                  #
# ------------------------------------------------------------------ #

class STGCN(nn.Module):
    """
    ST-GCN for presentation scoring (regression).

    Args:
        in_channels:  coordinate dims per joint (3 for x,y,confidence)
        n_joints:     number of skeleton joints (V)
        n_outputs:    number of scores to predict (5 for our use case)
        edges:        list of (i,j) joint connections
        dropout:      dropout rate in temporal conv layers
    """

    def __init__(self, in_channels=3, n_joints=25, n_outputs=5,
                 edges=None, dropout=0.5):
        super().__init__()

        if edges is None:
            edges = body25_edges()

        # Build adjacency matrix (K=3 for distance partitioning)
        A = build_adjacency(n_joints, edges, strategy="distance")
        self.register_buffer("A", torch.FloatTensor(A))
        K = A.shape[0]

        # Input batch normalisation
        self.data_bn = nn.BatchNorm1d(in_channels * n_joints)

        # 9 ST-GCN blocks — channel progression matches original paper
        self.blocks = nn.ModuleList([
            STGCNBlock(in_channels, 64, K, dropout=dropout),   # block 1
            STGCNBlock(64,  64,  K, dropout=dropout),           # block 2
            STGCNBlock(64,  64,  K, dropout=dropout),           # block 3
            STGCNBlock(64,  64,  K, dropout=dropout),           # block 4
            STGCNBlock(64,  128, K, stride=2, dropout=dropout), # block 5 ↓
            STGCNBlock(128, 128, K, dropout=dropout),           # block 6
            STGCNBlock(128, 128, K, dropout=dropout),           # block 7
            STGCNBlock(128, 256, K, stride=2, dropout=dropout), # block 8 ↓
            STGCNBlock(256, 256, K, dropout=dropout),           # block 9
        ])

        # Regression head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),    # (B, 256, 1, 1)
            nn.Flatten(),                     # (B, 256)
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, n_outputs),
            nn.Sigmoid(),                     # outputs in [0, 1]
        )

    def forward(self, x):
        """
        x: (B, C, T, V)
        returns: (B, n_outputs) — scores between 0 and 1
        """
        B, C, T, V = x.shape

        # Batch norm on raw input
        x = x.permute(0, 3, 1, 2).contiguous()    # (B, V, C, T)
        x = x.view(B, V * C, T)
        x = self.data_bn(x)
        x = x.view(B, V, C, T).permute(0, 2, 3, 1)  # back to (B, C, T, V)

        # ST-GCN blocks
        for block in self.blocks:
            x = block(x, self.A)

        # Regression head
        return self.head(x)


# ------------------------------------------------------------------ #
#  Pretrained weight loader                                           #
# ------------------------------------------------------------------ #

def load_pretrained(model, checkpoint_path):
    """
    Load pretrained weights from NTU RGB+D checkpoint into our model.
    Skips the final classification layer (we replaced it with our regression head).

    Args:
        model:           your STGCN instance
        checkpoint_path: path to downloaded .pth file

    Usage:
        model = STGCN(in_channels=3, n_joints=25, n_outputs=5)
        model = load_pretrained(model, 'stgcn_ntu_xview.pth')
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Handle different checkpoint formats
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model_dict = model.state_dict()

    # Filter out incompatible keys (final head has different shape)
    compatible = {
        k: v for k, v in state_dict.items()
        if k in model_dict and v.shape == model_dict[k].shape
    }

    skipped = [k for k in state_dict if k not in compatible]
    if skipped:
        print(f"Skipped {len(skipped)} incompatible layers (expected for the head):")
        for k in skipped[:5]:
            print(f"  {k}")

    model_dict.update(compatible)
    model.load_state_dict(model_dict)

    loaded_pct = len(compatible) / len(model_dict) * 100
    print(f"Loaded {len(compatible)}/{len(model_dict)} layers ({loaded_pct:.0f}%) from pretrained checkpoint")

    return model


# ------------------------------------------------------------------ #
#  Lightweight LSTM baseline (easier to get running first)           #
# ------------------------------------------------------------------ #

class SkeletonLSTM(nn.Module):
    """
    Simpler LSTM-based baseline. Use this to verify your data pipeline
    is working before committing to ST-GCN.

    Input:  (B, C, T, V)
    Output: (B, n_outputs)
    """
    def __init__(self, n_joints=25, n_coords=3, hidden=256,
                 n_layers=2, n_outputs=5, dropout=0.3):
        super().__init__()
        self.input_size = n_joints * n_coords
        self.lstm = nn.LSTM(
            self.input_size, hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, n_outputs),
            nn.Sigmoid(),
        )

    def forward(self, x):
        B, C, T, V = x.shape
        x = x.permute(0, 2, 3, 1)      # (B, T, V, C)
        x = x.reshape(B, T, V * C)     # (B, T, V*C)
        _, (h, _) = self.lstm(x)
        return self.head(h[-1])          # last layer's hidden state

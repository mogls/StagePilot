// Global Variables
let frames = [];
let currentFrame = 0;
let playing = false;
let playInterval = null;
let cocoModel = null;
let poseDetector = null;
let modelLoading = false;

// DOM Elements
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const statusSection = document.getElementById('status-section');
const statusText = document.getElementById('status-text');
const progressSection = document.getElementById('progress-section');
const progressLabel = document.getElementById('progress-label');
const progressPercent = document.getElementById('progress-percent');
const progressFill = document.getElementById('progress-fill');
const playerSection = document.getElementById('player-section');
const mainCanvas = document.getElementById('main-canvas');
const overlayCanvas = document.getElementById('overlay-canvas');
const playBtn = document.getElementById('play-btn');
const playIcon = document.getElementById('play-icon');
const playText = document.getElementById('play-text');
const scrubber = document.getElementById('scrubber');
const timeLabel = document.getElementById('time-label');
const statPersons = document.getElementById('stat-persons');
const statFrame = document.getElementById('stat-frame');
const statTotal = document.getElementById('stat-total');
const newBtn = document.getElementById('new-btn');

// Canvas Contexts
const ctx = mainCanvas.getContext('2d');
const octx = overlayCanvas.getContext('2d');

// Event Listeners
dropZone.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) handleFile(file);
});

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
});

playBtn.addEventListener('click', togglePlay);

scrubber.addEventListener('input', (e) => {
    currentFrame = parseInt(e.target.value);
    renderFrame(currentFrame);
});

newBtn.addEventListener('click', resetAll);

// Utility Functions
function setStatus(message, isError = false) {
    statusSection.style.display = 'block';
    statusText.textContent = message;
    const statusCard = statusSection.querySelector('.status-card');
    if (isError) {
        statusCard.classList.add('error');
    } else {
        statusCard.classList.remove('error');
    }
}

function setProgress(percent, label) {
    progressSection.style.display = 'block';
    progressFill.style.width = percent + '%';
    progressLabel.textContent = label;
    progressPercent.textContent = Math.round(percent) + '%';
}

function hideProgress() {
    progressSection.style.display = 'none';
}

function formatTime(seconds) {
    const minutes = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${minutes}:${String(secs).padStart(2, '0')}`;
}

// Main File Handler
async function handleFile(file) {
    // Reset state
    frames = [];
    currentFrame = 0;
    playing = false;
    clearInterval(playInterval);
    
    // Hide player section
    playerSection.style.display = 'none';
    
    // Show status
    setStatus('Video wird geladen...');
    setProgress(5, 'Video wird dekodiert...');
    
    // Create object URL and extract frames
    const url = URL.createObjectURL(file);
    await extractFrames(url);
    URL.revokeObjectURL(url);
}

// Frame Extraction
async function extractFrames(url) {
    return new Promise((resolve, reject) => {
        const video = document.createElement('video');
        video.src = url;
        video.muted = true;
        video.preload = 'auto';

        video.addEventListener('loadedmetadata', async () => {
            const duration = video.duration;
            const targetFPS = 2;
            const interval = 1 / targetFPS;
            const totalFrames = Math.floor(duration * targetFPS);

            setStatus(`Video-Dauer: ${duration.toFixed(1)}s → ${totalFrames} Frames bei 2 FPS`);

            const tmpCanvas = document.createElement('canvas');
            const tmpCtx = tmpCanvas.getContext('2d');

            let currentTime = 0;
            let frameIndex = 0;
            frames = [];

            const seekHandler = () => {
                // Capture frame
                tmpCanvas.width = video.videoWidth;
                tmpCanvas.height = video.videoHeight;
                tmpCtx.drawImage(video, 0, 0);
                const imageData = tmpCtx.getImageData(0, 0, tmpCanvas.width, tmpCanvas.height);
                
                frames.push({
                    imageData,
                    width: tmpCanvas.width,
                    height: tmpCanvas.height,
                    detections: null
                });

                frameIndex++;
                const progress = Math.round((frameIndex / totalFrames) * 60) + 5;
                setProgress(progress, `Frame ${frameIndex} / ${totalFrames} extrahiert...`);

                currentTime += interval;

                if (currentTime <= duration) {
                    video.currentTime = currentTime;
                } else {
                    finalize();
                }
            };

            const finalize = async () => {
                video.removeEventListener('seeked', seekHandler);
                setStatus(`${frames.length} Frames extrahiert. KI-Modell wird geladen...`);
                setProgress(70, 'TensorFlow.js COCO-SSD Modell wird geladen...');
                
                await loadModel();
                
                setProgress(80, 'Personen werden erkannt...');
                await detectAllPersons();
                
                setProgress(100, 'Fertig!');
                setTimeout(() => {
                    hideProgress();
                    showPlayer();
                    resolve();
                }, 500);
            };

            video.addEventListener('seeked', seekHandler);
            video.currentTime = currentTime;
        });

        video.addEventListener('error', () => {
            setStatus('Fehler beim Laden des Videos', true);
            hideProgress();
            reject();
        });
    });
}

// Model Loading
async function loadModel() {
    if (cocoModel && poseDetector) return;
    if (modelLoading) return;
    
    modelLoading = true;
    try {
        // Wait for TensorFlow.js to be ready
        await tf.ready();
        console.log('TensorFlow.js ready');
        
        // Load COCO-SSD for person detection
        if (!cocoModel) {
            cocoModel = await cocoSsd.load();
            console.log('COCO-SSD Model loaded successfully');
        }
        
        // Load MoveNet for body part detection
        if (!poseDetector) {
            const detectorConfig = {
                modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING
            };
            poseDetector = await poseDetection.createDetector(
                poseDetection.SupportedModels.MoveNet,
                detectorConfig
            );
            console.log('MoveNet Pose Detection Model loaded successfully');
        }
    } catch (error) {
        setStatus('Fehler beim Laden des KI-Modells: ' + error.message, true);
        console.error('Model loading error:', error);
        throw error;
    }
}

// Person Detection with Body Parts
async function detectAllPersons() {
    if (!cocoModel || !poseDetector) return;
    
    const tmpCanvas = document.createElement('canvas');
    const tmpCtx = tmpCanvas.getContext('2d');
    
    for (let i = 0; i < frames.length; i++) {
        const frame = frames[i];
        tmpCanvas.width = frame.width;
        tmpCanvas.height = frame.height;
        tmpCtx.putImageData(frame.imageData, 0, 0);
        
        try {
            // Schritt 1: Detect person bounding box
            const predictions = await cocoModel.detect(tmpCanvas);
            const persons = predictions.filter(pred => pred.class === 'person');
            
            // Nur die Person mit der höchsten Wahrscheinlichkeit behalten
            if (persons.length > 0) {
                const bestPerson = persons.reduce((best, current) =>
                    current.score > best.score ? current : best
                );
                
                frame.detections = [bestPerson];
                
                // Schritt 2: Crop the person region and detect body parts only in that region
                const [x, y, width, height] = bestPerson.bbox;
                
                // Create a cropped canvas with the person region
                const croppedCanvas = document.createElement('canvas');
                const croppedCtx = croppedCanvas.getContext('2d');
                croppedCanvas.width = width;
                croppedCanvas.height = height;
                
                // Draw the cropped person region
                croppedCtx.putImageData(
                    tmpCtx.getImageData(x, y, width, height),
                    0, 0
                );
                
                // Detect body parts only in the cropped region
                const poses = await poseDetector.estimatePoses(croppedCanvas);
                
                if (poses.length > 0) {
                    // Adjust keypoint coordinates back to original canvas coordinates
                    const adjustedPose = {
                        ...poses[0],
                        keypoints: poses[0].keypoints.map(kp => ({
                            ...kp,
                            x: kp.x + x,
                            y: kp.y + y
                        }))
                    };
                    frame.poses = [adjustedPose];
                } else {
                    frame.poses = [];
                }
            } else {
                frame.detections = [];
                frame.poses = [];
            }
        } catch (error) {
            console.error('Detection error:', error);
            frame.detections = [];
            frame.poses = [];
        }
        
        const progress = 80 + Math.round((i / frames.length) * 18);
        setProgress(progress, `Körperteile werden erkannt: Frame ${i + 1} / ${frames.length}`);
    }
}

// Player Functions
function showPlayer() {
    if (!frames.length) return;
    
    const firstFrame = frames[0];
    mainCanvas.width = firstFrame.width;
    mainCanvas.height = firstFrame.height;
    overlayCanvas.width = firstFrame.width;
    overlayCanvas.height = firstFrame.height;
    
    playerSection.style.display = 'block';
    statTotal.textContent = frames.length;
    scrubber.max = frames.length - 1;
    
    setStatus(`Bereit · ${frames.length} Frames · 2 FPS`);
    renderFrame(0);
}

function renderFrame(index) {
    if (!frames[index]) return;
    
    const frame = frames[index];
    
    // Draw main frame
    ctx.putImageData(frame.imageData, 0, 0);
    
    // Clear overlay
    octx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    
    // Get detections
    const detections = frame.detections || [];
    
    // Update stats
    statPersons.textContent = detections.length > 0 ? 'Ja' : 'Nein';
    statFrame.textContent = index + 1;
    scrubber.value = index;
    
    // Update time
    const totalSeconds = frames.length / 2;
    const currentSeconds = index / 2;
    timeLabel.textContent = `${formatTime(currentSeconds)} / ${formatTime(totalSeconds)}`;
    
    // Draw bounding box for the best person
    detections.forEach((detection) => {
        const [x, y, width, height] = detection.bbox;
        const confidence = Math.round(detection.score * 100);
        
        // Draw box
        octx.strokeStyle = '#1D9E75';
        octx.lineWidth = Math.max(3, overlayCanvas.width / 300);
        octx.strokeRect(x, y, width, height);
        
        // Draw label background
        const fontSize = Math.max(14, overlayCanvas.width / 50);
        octx.font = `600 ${fontSize}px sans-serif`;
        const label = `Person  ${confidence}%`;
        const textWidth = octx.measureText(label).width;
        const padding = 8;
        const labelY = Math.max(0, y - fontSize - padding * 2);
        
        octx.fillStyle = '#1D9E75';
        octx.fillRect(x - 2, labelY, textWidth + padding * 2, fontSize + padding * 2);
        
        // Draw label text
        octx.fillStyle = '#ffffff';
        octx.fillText(label, x + padding - 2, labelY + fontSize + padding * 0.8);
    });
    
    // Draw body parts (pose keypoints)
    const poses = frame.poses || [];
    if (poses.length > 0) {
        const pose = poses[0];
        drawPoseKeypoints(pose);
        drawPoseSkeleton(pose);
    }
}

// Körperteil-Namen auf Deutsch
const keypointNames = {
    'nose': 'Nase',
    'left_eye': 'Linkes Auge',
    'right_eye': 'Rechtes Auge',
    'left_ear': 'Linkes Ohr',
    'right_ear': 'Rechtes Ohr',
    'left_shoulder': 'Linke Schulter',
    'right_shoulder': 'Rechte Schulter',
    'left_elbow': 'Linker Ellbogen',
    'right_elbow': 'Rechter Ellbogen',
    'left_wrist': 'Linkes Handgelenk',
    'right_wrist': 'Rechtes Handgelenk',
    'left_hip': 'Linke Hüfte',
    'right_hip': 'Rechte Hüfte',
    'left_knee': 'Linkes Knie',
    'right_knee': 'Rechtes Knie',
    'left_ankle': 'Linker Knöchel',
    'right_ankle': 'Rechter Knöchel'
};

function drawPoseKeypoints(pose) {
    const keypoints = pose.keypoints;
    const minConfidence = 0.3;
    
    keypoints.forEach((keypoint) => {
        if (keypoint.score > minConfidence) {
            const { x, y, name } = keypoint;
            
            // Draw keypoint circle
            octx.beginPath();
            octx.arc(x, y, 6, 0, 2 * Math.PI);
            octx.fillStyle = '#FF6B6B';
            octx.fill();
            octx.strokeStyle = '#ffffff';
            octx.lineWidth = 2;
            octx.stroke();
            
            // Draw label
            const germanName = keypointNames[name] || name;
            const fontSize = Math.max(10, overlayCanvas.width / 80);
            octx.font = `500 ${fontSize}px sans-serif`;
            octx.fillStyle = '#FF6B6B';
            octx.strokeStyle = '#ffffff';
            octx.lineWidth = 3;
            octx.strokeText(germanName, x + 10, y - 5);
            octx.fillText(germanName, x + 10, y - 5);
        }
    });
}

function drawPoseSkeleton(pose) {
    const keypoints = pose.keypoints;
    const minConfidence = 0.3;
    
    // Define skeleton connections
    const connections = [
        ['nose', 'left_eye'],
        ['nose', 'right_eye'],
        ['left_eye', 'left_ear'],
        ['right_eye', 'right_ear'],
        ['left_shoulder', 'right_shoulder'],
        ['left_shoulder', 'left_elbow'],
        ['left_elbow', 'left_wrist'],
        ['right_shoulder', 'right_elbow'],
        ['right_elbow', 'right_wrist'],
        ['left_shoulder', 'left_hip'],
        ['right_shoulder', 'right_hip'],
        ['left_hip', 'right_hip'],
        ['left_hip', 'left_knee'],
        ['left_knee', 'left_ankle'],
        ['right_hip', 'right_knee'],
        ['right_knee', 'right_ankle']
    ];
    
    // Draw connections
    connections.forEach(([startName, endName]) => {
        const startPoint = keypoints.find(kp => kp.name === startName);
        const endPoint = keypoints.find(kp => kp.name === endName);
        
        if (startPoint && endPoint &&
            startPoint.score > minConfidence &&
            endPoint.score > minConfidence) {
            
            octx.beginPath();
            octx.moveTo(startPoint.x, startPoint.y);
            octx.lineTo(endPoint.x, endPoint.y);
            octx.strokeStyle = '#4ECDC4';
            octx.lineWidth = Math.max(2, overlayCanvas.width / 400);
            octx.stroke();
        }
    });
}

function togglePlay() {
    if (!frames.length) return;
    
    playing = !playing;
    
    if (playing) {
        playIcon.className = 'ti ti-player-pause';
        playText.textContent = 'Pausieren';
        
        playInterval = setInterval(() => {
            if (currentFrame >= frames.length - 1) {
                currentFrame = 0;
            } else {
                currentFrame++;
            }
            renderFrame(currentFrame);
        }, 500); // 2 FPS = 500ms per frame
    } else {
        playIcon.className = 'ti ti-player-play';
        playText.textContent = 'Abspielen';
        clearInterval(playInterval);
    }
}

function resetAll() {
    // Stop playback
    clearInterval(playInterval);
    playing = false;
    
    // Reset state
    frames = [];
    currentFrame = 0;
    
    // Reset UI
    playerSection.style.display = 'none';
    statusSection.style.display = 'none';
    progressSection.style.display = 'none';
    fileInput.value = '';
    
    playIcon.className = 'ti ti-player-play';
    playText.textContent = 'Abspielen';
}

// Initialize
console.log('Video FPS Reducer + Person Tracker initialized');

// Made with Bob

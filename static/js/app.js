// Handwritten Digit Recognition - Interactive Frontend Engine

let currentModel = 'cnn';
let isDrawing = false;
let lastX = 0;
let lastY = 0;
let activeTab = 'draw';
let debounceTimer = null;

const canvas = document.getElementById('drawingCanvas');
const ctx = canvas.getContext('2d');
const brushSizeSlider = document.getElementById('brushSize');
const brushSizeVal = document.getElementById('brushSizeVal');

// Canvas Initialization
function initCanvas() {
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = '#FFFFFF';
    ctx.lineWidth = parseInt(brushSizeSlider.value, 10);
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
}

brushSizeSlider.addEventListener('input', (e) => {
    const val = e.target.value;
    brushSizeVal.innerText = `${val}px`;
    ctx.lineWidth = parseInt(val, 10);
});

// Event Listeners for Drawing
function getCanvasPos(e) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    
    if (e.touches && e.touches.length > 0) {
        return {
            x: (e.touches[0].clientX - rect.left) * scaleX,
            y: (e.touches[0].clientY - rect.top) * scaleY
        };
    }
    return {
        x: (e.clientX - rect.left) * scaleX,
        y: (e.clientY - rect.top) * scaleY
    };
}

function startDrawing(e) {
    e.preventDefault();
    isDrawing = true;
    const pos = getCanvasPos(e);
    lastX = pos.x;
    lastY = pos.y;
    
    // Draw initial dot
    ctx.beginPath();
    ctx.arc(lastX, lastY, ctx.lineWidth / 2, 0, Math.PI * 2);
    ctx.fillStyle = '#FFFFFF';
    ctx.fill();
    ctx.beginPath();
}

function draw(e) {
    if (!isDrawing) return;
    e.preventDefault();
    const pos = getCanvasPos(e);
    
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();
    
    lastX = pos.x;
    lastY = pos.y;
    
    // Trigger real-time live prediction with slight debounce
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
        classifyCurrent(false);
    }, 280);
}

function stopDrawing(e) {
    if (!isDrawing) return;
    isDrawing = false;
    clearTimeout(debounceTimer);
    classifyCurrent(false);
}

canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseleave', stopDrawing);

canvas.addEventListener('touchstart', startDrawing, { passive: false });
canvas.addEventListener('touchmove', draw, { passive: false });
canvas.addEventListener('touchend', stopDrawing, { passive: false });

function clearCanvas() {
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    resetOutputs();
}

function resetOutputs() {
    document.getElementById('digitDisplay').innerText = '—';
    document.getElementById('confidenceDisplay').innerText = '0.0%';
    document.getElementById('confBarFill').style.width = '0%';
    document.getElementById('processedImgPreview').src = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='56' height='56' viewBox='0 0 28 28'%3E%3Crect width='28' height='28' fill='%23111827'/%3E%3C/svg%3E";
    updateProbabilities(new Array(10).fill(0));
}

// Probability Bars Generator
function renderEmptyProbabilities() {
    const container = document.getElementById('probBarsContainer');
    container.innerHTML = '';
    for (let i = 0; i < 10; i++) {
        const row = document.createElement('div');
        row.className = 'prob-row';
        row.id = `prob-row-${i}`;
        row.innerHTML = `
            <span class="prob-digit">${i}</span>
            <div class="prob-bar-track">
                <div class="prob-bar-inner" id="prob-bar-${i}" style="width: 0%;"></div>
            </div>
            <span class="prob-percent" id="prob-pct-${i}">0.0%</span>
        `;
        container.appendChild(row);
    }
}

function updateProbabilities(probs, winningDigit = -1) {
    for (let i = 0; i < 10; i++) {
        const row = document.getElementById(`prob-row-${i}`);
        const bar = document.getElementById(`prob-bar-${i}`);
        const pctText = document.getElementById(`prob-pct-${i}`);
        
        const pct = (probs[i] * 100).toFixed(1);
        bar.style.width = `${pct}%`;
        pctText.innerText = `${pct}%`;
        
        if (i === winningDigit && probs[i] > 0.1) {
            row.classList.add('active');
        } else {
            row.classList.remove('active');
        }
    }
}

// Model Switching
function selectModel(modelName) {
    currentModel = modelName;
    document.getElementById('btnModelCNN').classList.toggle('active', modelName === 'cnn');
    document.getElementById('btnModelMLP').classList.toggle('active', modelName === 'mlp');
    classifyCurrent(false);
}

// Tab Switching
function switchTab(tab) {
    activeTab = tab;
    document.getElementById('tabDraw').classList.toggle('active', tab === 'draw');
    document.getElementById('tabUpload').classList.toggle('active', tab === 'upload');
    document.getElementById('drawContainer').classList.toggle('active', tab === 'draw');
    document.getElementById('uploadContainer').classList.toggle('active', tab === 'upload');
}

// File Upload & Drag-and-drop
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');

['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.add('dragover');
    });
});

['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.remove('dragover');
    });
});

dropZone.addEventListener('drop', (e) => {
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        handleFile(e.dataTransfer.files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
});

let currentUploadedImageBase64 = null;

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please upload an image file (PNG, JPG, JPEG)');
        return;
    }
    const reader = new FileReader();
    reader.onload = (e) => {
        currentUploadedImageBase64 = e.target.result;
        document.getElementById('uploadPreviewImg').src = currentUploadedImageBase64;
        dropZone.style.display = 'none';
        document.getElementById('uploadPreviewWrapper').style.display = 'flex';
        sendPredictionRequest(currentUploadedImageBase64);
    };
    reader.readAsDataURL(file);
}

function resetUpload() {
    currentUploadedImageBase64 = null;
    fileInput.value = '';
    document.getElementById('uploadPreviewWrapper').style.display = 'none';
    dropZone.style.display = 'flex';
    resetOutputs();
}

// Classification Dispatcher
function classifyCurrent(showVisualFeedback = true) {
    let imagePayload = null;
    if (activeTab === 'draw') {
        imagePayload = canvas.toDataURL('image/png');
    } else if (activeTab === 'upload' && currentUploadedImageBase64) {
        imagePayload = currentUploadedImageBase64;
    }
    
    if (!imagePayload) return;
    sendPredictionRequest(imagePayload, showVisualFeedback);
}

// API Call
async function sendPredictionRequest(base64Data, showFeedback = false) {
    const startTime = performance.now();
    const statusBadge = document.getElementById('statusBadge');
    statusBadge.innerText = 'Predicting...';
    
    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                image: base64Data,
                model: currentModel
            })
        });
        
        const data = await response.json();
        const latency = Math.round(performance.now() - startTime);
        document.getElementById('latencyVal').innerText = `${latency} ms`;
        statusBadge.innerText = 'Ready';
        
        if (data.status === 'success') {
            document.getElementById('digitDisplay').innerText = data.prediction;
            document.getElementById('confidenceDisplay').innerText = `${(data.confidence * 100).toFixed(1)}%`;
            document.getElementById('confBarFill').style.width = `${(data.confidence * 100).toFixed(1)}%`;
            
            if (data.preprocessed_image) {
                document.getElementById('processedImgPreview').src = `data:image/png;base64,${data.preprocessed_image}`;
            }
            
            updateProbabilities(data.probabilities, data.prediction);
        } else if (data.status === 'empty') {
            resetOutputs();
        }
    } catch (err) {
        console.error('Inference error:', err);
        statusBadge.innerText = 'Error';
    }
}

// Preset Quick Sample Digits Loader
function initSampleTray() {
    const tray = document.getElementById('samplesList');
    tray.innerHTML = '';
    for (let digit = 0; digit <= 9; digit++) {
        const chip = document.createElement('button');
        chip.type = 'button';
        chip.className = 'sample-chip';
        chip.innerText = `Digit ${digit}`;
        chip.onclick = () => loadSampleDigit(digit);
        tray.appendChild(chip);
    }
}

async function loadSampleDigit(digit) {
    try {
        const res = await fetch(`/api/sample/${digit}`);
        const data = await res.json();
        if (data.status === 'success') {
            // Draw onto canvas
            const img = new Image();
            img.onload = () => {
                ctx.fillStyle = '#000000';
                ctx.fillRect(0, 0, canvas.width, canvas.height);
                ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                switchTab('draw');
                classifyCurrent(false);
            };
            img.src = `data:image/png;base64,${data.image}`;
        }
    } catch (e) {
        console.error('Failed to load sample digit', e);
    }
}

// Window load init
window.addEventListener('DOMContentLoaded', () => {
    initCanvas();
    renderEmptyProbabilities();
    initSampleTray();
});

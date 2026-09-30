// Web-Only TensorFlow.js Inference Engine for GitHub Pages Deployment

let model = null;
let isDrawing = false;
let lastX = 0;
let lastY = 0;
let activeTab = 'draw';
let debounceTimer = null;
let currentUploadedImage = null;

const canvas = document.getElementById('drawingCanvas');
const ctx = canvas.getContext('2d');
const brushSizeSlider = document.getElementById('brushSize');
const brushSizeVal = document.getElementById('brushSizeVal');

// Initialize Drawing Canvas
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
    
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => classifyCurrent(), 150);
}

function stopDrawing(e) {
    if (!isDrawing) return;
    isDrawing = false;
    clearTimeout(debounceTimer);
    classifyCurrent();
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

// Probabilities UI
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

function switchTab(tab) {
    activeTab = tab;
    document.getElementById('tabDraw').classList.toggle('active', tab === 'draw');
    document.getElementById('tabUpload').classList.toggle('active', tab === 'upload');
    document.getElementById('drawContainer').classList.toggle('active', tab === 'draw');
    document.getElementById('uploadContainer').classList.toggle('active', tab === 'upload');
}

// Upload & Drag-and-drop
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');

['dragenter', 'dragover'].forEach(name => {
    dropZone.addEventListener(name, (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });
});
['dragleave', 'drop'].forEach(name => {
    dropZone.addEventListener(name, (e) => {
        e.preventDefault();
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

function handleFile(file) {
    if (!file.type.startsWith('image/')) return;
    const reader = new FileReader();
    reader.onload = (e) => {
        const img = new Image();
        img.onload = () => {
            currentUploadedImage = img;
            document.getElementById('uploadPreviewImg').src = e.target.result;
            dropZone.style.display = 'none';
            document.getElementById('uploadPreviewWrapper').style.display = 'flex';
            classifyCurrent();
        };
        img.src = e.target.result;
    };
    reader.readAsDataURL(file);
}

function resetUpload() {
    currentUploadedImage = null;
    fileInput.value = '';
    document.getElementById('uploadPreviewWrapper').style.display = 'none';
    dropZone.style.display = 'flex';
    resetOutputs();
}

// Model Loading & Construction in TensorFlow.js
async function loadBrowserCNNModel() {
    const statusBadge = document.getElementById('statusBadge');
    statusBadge.innerText = 'Loading Model...';

    try {
        // Construct CNN Architecture
        const cnn = tf.sequential();
        
        // Block 1
        cnn.add(tf.layers.conv2d({
            inputShape: [28, 28, 1],
            filters: 32,
            kernelSize: 3,
            padding: 'same',
            activation: 'relu'
        }));
        cnn.add(tf.layers.batchNormalization());
        cnn.add(tf.layers.conv2d({
            filters: 32,
            kernelSize: 3,
            padding: 'same',
            activation: 'relu'
        }));
        cnn.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));
        cnn.add(tf.layers.dropout({ rate: 0.25 }));

        // Block 2
        cnn.add(tf.layers.conv2d({
            filters: 64,
            kernelSize: 3,
            padding: 'same',
            activation: 'relu'
        }));
        cnn.add(tf.layers.batchNormalization());
        cnn.add(tf.layers.conv2d({
            filters: 64,
            kernelSize: 3,
            padding: 'same',
            activation: 'relu'
        }));
        cnn.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));
        cnn.add(tf.layers.dropout({ rate: 0.25 }));

        // Classification Head
        cnn.add(tf.layers.flatten());
        cnn.add(tf.layers.dense({ units: 128, activation: 'relu' }));
        cnn.add(tf.layers.batchNormalization());
        cnn.add(tf.layers.dropout({ rate: 0.40 }));
        cnn.add(tf.layers.dense({ units: 10, activation: 'softmax' }));

        // Load Manifest and Binary Weights
        const manifestRes = await fetch('web_model/weights_manifest.json');
        const manifest = await manifestRes.json();
        
        const binRes = await fetch('web_model/weights.bin');
        const binBuffer = await binRes.arrayBuffer();

        const tensors = [];
        let byteOffset = 0;
        for (const item of manifest) {
            const numElements = item.shape.reduce((a, b) => a * b, 1);
            const byteLength = numElements * 4;
            const slice = binBuffer.slice(byteOffset, byteOffset + byteLength);
            const floatData = new Float32Array(slice);
            tensors.push(tf.tensor(floatData, item.shape, 'float32'));
            byteOffset += byteLength;
        }

        cnn.setWeights(tensors);
        model = cnn;

        statusBadge.innerText = 'Ready';
        statusBadge.classList.add('live');
        console.log("TensorFlow.js CNN Model loaded and weights initialized successfully!");
    } catch (err) {
        console.error("Failed to load browser model:", err);
        statusBadge.innerText = 'Model Error';
    }
}

// Center of Mass & MNIST Preprocessor in JavaScript
function preprocessCanvasToMNIST(sourceCanvasOrImg) {
    // 1. Draw onto temp canvas to extract grayscale pixels
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 280;
    tempCanvas.height = 280;
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.drawImage(sourceCanvasOrImg, 0, 0, 280, 280);

    const imgData = tempCtx.getImageData(0, 0, 280, 280);
    const data = imgData.data;

    let gray = new Float32Array(280 * 280);
    for (let i = 0; i < data.length; i += 4) {
        // Luminance
        gray[i / 4] = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
    }

    // Check background corners: invert if bright
    const corners = [gray[0], gray[279], gray[280 * 279], gray[280 * 280 - 1]];
    const avgCorner = (corners[0] + corners[1] + corners[2] + corners[3]) / 4;
    if (avgCorner > 127) {
        for (let i = 0; i < gray.length; i++) {
            gray[i] = 255.0 - gray[i];
        }
    }

    // Noise gate
    for (let i = 0; i < gray.length; i++) {
        if (gray[i] < 30) gray[i] = 0;
    }

    // Find bounding box
    let minX = 280, maxX = -1, minY = 280, maxY = -1;
    for (let y = 0; y < 280; y++) {
        for (let x = 0; x < 280; x++) {
            if (gray[y * 280 + x] > 30) {
                if (x < minX) minX = x;
                if (x > maxX) maxX = x;
                if (y < minY) minY = y;
                if (y > maxY) maxY = y;
            }
        }
    }

    if (minX > maxX || minY > maxY) {
        return null; // Empty
    }

    const cropW = maxX - minX + 1;
    const cropH = maxY - minY + 1;

    // Scale to fit 20x20
    let targetW, targetH;
    if (cropH > cropW) {
        targetH = 20;
        targetW = Math.max(1, Math.round(cropW * (20.0 / cropH)));
    } else {
        targetW = 20;
        targetH = Math.max(1, Math.round(cropH * (20.0 / cropW)));
    }

    // Crop onto intermediate canvas
    const cropCanvas = document.createElement('canvas');
    cropCanvas.width = targetW;
    cropCanvas.height = targetH;
    const cropCtx = cropCanvas.getContext('2d');
    cropCtx.drawImage(tempCanvas, minX, minY, cropW, cropH, 0, 0, targetW, targetH);

    const croppedData = cropCtx.getImageData(0, 0, targetW, targetH).data;

    // Pad into 28x28
    const padded = new Float32Array(28 * 28);
    const startX = Math.floor((28 - targetW) / 2);
    const startY = Math.floor((28 - targetH) / 2);

    for (let y = 0; y < targetH; y++) {
        for (let x = 0; x < targetW; x++) {
            const idx = (y * targetW + x) * 4;
            const val = 0.299 * croppedData[idx] + 0.587 * croppedData[idx + 1] + 0.114 * croppedData[idx + 2];
            padded[(startY + y) * 28 + (startX + x)] = avgCorner > 127 ? (255 - val) : val;
        }
    }

    // Center of mass
    let sumVal = 0, sumX = 0, sumY = 0;
    for (let y = 0; y < 28; y++) {
        for (let x = 0; x < 28; x++) {
            const v = padded[y * 28 + x];
            if (v > 25) {
                sumVal += v;
                sumX += x * v;
                sumY += y * v;
            }
        }
    }

    const centered = new Float32Array(28 * 28);
    if (sumVal > 0) {
        const cX = sumX / sumVal;
        const cY = sumY / sumVal;
        const shiftX = Math.round(14 - cX);
        const shiftY = Math.round(14 - cY);

        for (let y = 0; y < 28; y++) {
            for (let x = 0; x < 28; x++) {
                const srcX = x - shiftX;
                const srcY = y - shiftY;
                if (srcX >= 0 && srcX < 28 && srcY >= 0 && srcY < 28) {
                    centered[y * 28 + x] = padded[srcY * 28 + srcX];
                }
            }
        }
    } else {
        centered.set(padded);
    }

    // Normalize to [0, 1]
    const normalized = new Float32Array(28 * 28);
    for (let i = 0; i < 28 * 28; i++) {
        normalized[i] = Math.min(1.0, Math.max(0.0, centered[i] / 255.0));
    }

    // Render 28x28 preview
    const previewCanvas = document.createElement('canvas');
    previewCanvas.width = 28;
    previewCanvas.height = 28;
    const prevCtx = previewCanvas.getContext('2d');
    const prevImgData = prevCtx.createImageData(28, 28);
    for (let i = 0; i < 28 * 28; i++) {
        const v = Math.round(normalized[i] * 255);
        prevImgData.data[i * 4] = v;
        prevImgData.data[i * 4 + 1] = v;
        prevImgData.data[i * 4 + 2] = v;
        prevImgData.data[i * 4 + 3] = 255;
    }
    prevCtx.putImageData(prevImgData, 0, 0);
    document.getElementById('processedImgPreview').src = previewCanvas.toDataURL();

    return normalized;
}

// Run Inference
async function classifyCurrent() {
    if (!model) return;

    let source = null;
    if (activeTab === 'draw') {
        source = canvas;
    } else if (activeTab === 'upload' && currentUploadedImage) {
        source = currentUploadedImage;
    }
    if (!source) return;

    const t0 = performance.now();
    const normalized28 = preprocessCanvasToMNIST(source);

    if (!normalized28) {
        resetOutputs();
        return;
    }

    // Create tensor [1, 28, 28, 1]
    const inputTensor = tf.tensor4d(normalized28, [1, 28, 28, 1]);
    const outputTensor = model.predict(inputTensor);
    const probs = await outputTensor.data();

    inputTensor.dispose();
    outputTensor.dispose();

    const latency = Math.round(performance.now() - t0);
    document.getElementById('latencyVal').innerText = `${latency} ms`;

    let maxIdx = 0;
    let maxVal = probs[0];
    for (let i = 1; i < 10; i++) {
        if (probs[i] > maxVal) {
            maxVal = probs[i];
            maxIdx = i;
        }
    }

    document.getElementById('digitDisplay').innerText = maxIdx;
    document.getElementById('confidenceDisplay').innerText = `${(maxVal * 100).toFixed(1)}%`;
    document.getElementById('confBarFill').style.width = `${(maxVal * 100).toFixed(1)}%`;

    updateProbabilities(probs, maxIdx);
}

// Preset Quick Sample Digits Generator
function initSampleTray() {
    const tray = document.getElementById('samplesList');
    tray.innerHTML = '';
    for (let digit = 0; digit <= 9; digit++) {
        const chip = document.createElement('button');
        chip.type = 'button';
        chip.className = 'sample-chip';
        chip.innerText = `Digit ${digit}`;
        chip.onclick = () => drawPresetDigit(digit);
        tray.appendChild(chip);
    }
}

// Sample Digit Drawing Glyphs for Instant Testing
function drawPresetDigit(digit) {
    switchTab('draw');
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = '#FFFFFF';
    ctx.lineWidth = 22;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    const cx = 140, cy = 140;
    ctx.beginPath();
    
    if (digit === 0) {
        ctx.ellipse(cx, cy, 45, 75, 0, 0, Math.PI * 2);
    } else if (digit === 1) {
        ctx.moveTo(cx - 20, cy - 60);
        ctx.lineTo(cx, cy - 75);
        ctx.lineTo(cx, cy + 75);
    } else if (digit === 2) {
        ctx.arc(cx, cy - 35, 40, Math.PI, 0);
        ctx.lineTo(cx - 40, cy + 70);
        ctx.lineTo(cx + 45, cy + 70);
    } else if (digit === 3) {
        ctx.arc(cx, cy - 35, 40, -Math.PI / 2, Math.PI / 2);
        ctx.arc(cx, cy + 35, 40, -Math.PI / 2, Math.PI / 2);
    } else if (digit === 4) {
        ctx.moveTo(cx + 25, cy + 70);
        ctx.lineTo(cx + 25, cy - 70);
        ctx.lineTo(cx - 45, cy + 20);
        ctx.lineTo(cx + 55, cy + 20);
    } else if (digit === 5) {
        ctx.moveTo(cx + 40, cy - 70);
        ctx.lineTo(cx - 35, cy - 70);
        ctx.lineTo(cx - 40, cy - 10);
        ctx.arc(cx, cy + 25, 45, -Math.PI / 2, Math.PI / 2);
    } else if (digit === 6) {
        ctx.arc(cx, cy + 30, 42, 0, Math.PI * 2);
        ctx.moveTo(cx - 42, cy + 30);
        ctx.quadraticCurveTo(cx - 30, cy - 50, cx + 25, cy - 70);
    } else if (digit === 7) {
        ctx.moveTo(cx - 45, cy - 70);
        ctx.lineTo(cx + 45, cy - 70);
        ctx.lineTo(cx - 15, cy + 75);
    } else if (digit === 8) {
        ctx.arc(cx, cy - 35, 34, 0, Math.PI * 2);
        ctx.moveTo(cx + 42, cy + 35);
        ctx.arc(cx, cy + 35, 42, 0, Math.PI * 2);
    } else if (digit === 9) {
        ctx.arc(cx, cy - 30, 42, 0, Math.PI * 2);
        ctx.moveTo(cx + 42, cy - 30);
        ctx.quadraticCurveTo(cx + 30, cy + 50, cx - 25, cy + 70);
    }
    ctx.stroke();
    classifyCurrent();
}

window.addEventListener('DOMContentLoaded', () => {
    initCanvas();
    renderEmptyProbabilities();
    initSampleTray();
    loadBrowserCNNModel();
});

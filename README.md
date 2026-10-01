# AI-Based Handwritten Digit Recognition System

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-3.x-red)](https://keras.io/)
[![Flask](https://img.shields.io/badge/Flask-3.x-green)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

An end-to-end Computer Vision & Deep Learning major project that recognizes handwritten digits (0–9) using Convolutional Neural Networks (CNNs). Includes training pipelines, baseline MLP comparisons, comprehensive evaluation metrics, and a modern, interactive web application for real-time drawing and image classification.

### 🌐 Web Application Access
- 🚀 **Online GitHub Pages Demo (Live in Browser):**  
  👉 **[https://vinila-velmala.github.io/AI-Based-Handwritten-Digit-Recognition-System/](https://vinila-velmala.github.io/AI-Based-Handwritten-Digit-Recognition-System/)**

- 💻 **Local Web Access (Running on Your Machine):**  
  1. Run `python app.py` in your terminal.  
  2. Open: [http://localhost:5000](http://localhost:5000/) or [http://127.0.0.1:5000](http://127.0.0.1:5000/)  
  *(Local Network Wi-Fi Access: `http://192.168.1.102:5000`)*

---

## 📌 Project Overview

Handwritten digit recognition plays a critical role in postal automated mail sorting, banking check verification, and Optical Character Recognition (OCR). This project develops and evaluates high-accuracy neural networks on the benchmark **MNIST dataset (70,000 images)**.

### Key Goals:
- **Build & Train CNN:** Construct a deep convolutional architecture achieving **>99.2% test accuracy**.
- **Benchmark Comparison:** Compare against a Multilayer Perceptron (MLP) baseline to demonstrate the power of convolutional feature extraction.
- **Diagnostics & Error Analysis:** Generate Confusion Matrices, per-digit Precision/Recall/F1-scores, and visualize misclassified edge cases.
- **Interactive Web Interface:** Provide a real-time drawing canvas and image uploader powered by a custom **center-of-mass centering & bounding box normalizer** that mirrors the MNIST training distribution.

---

## 📁 Repository Structure

```
Handwritten Digit/
│
├── data/                                 # Cached dataset storage
├── models/
│   ├── digit_cnn_model.keras            # Saved high-accuracy CNN model
│   └── digit_mlp_model.keras            # Saved baseline MLP model
│
├── outputs/                             # Visualizations and metrics
│   ├── dataset_samples.png              # Samples from classes 0–9
│   ├── training_history_cnn.png         # CNN loss & accuracy curves
│   ├── training_history_mlp.png         # MLP loss & accuracy curves
│   ├── confusion_matrix_cnn.png         # High-resolution confusion matrix
│   ├── model_comparison.png             # CNN vs MLP benchmark chart
│   ├── misclassified_samples.png        # Breakdown of model errors
│   ├── correct_predictions.png          # High-confidence correct samples
│   ├── classification_report.txt        # Full per-class metrics
│   └── metrics_summary.json             # Structured evaluation metrics
│
├── static/                              # Web application assets
│   ├── css/
│   │   └── style.css                    # Dark glassmorphism styling
│   └── js/
│       └── app.js                       # Canvas engine, API caller, live charts
│
├── templates/
│   └── index.html                       # Responsive HTML5 web interface
│
├── model_architecture.py                # CNN and MLP model definitions
├── preprocess.py                        # Dataset loader & center-of-mass preprocessor
├── train.py                             # Complete model training script
├── evaluate.py                          # Evaluation, confusion matrix & error analysis
├── app.py                               # Flask web server & inference API
├── handwritten_digit_recognition.ipynb  # Interactive, step-by-step Jupyter Notebook
├── requirements.txt                     # Python package dependencies
├── REPORT.md                            # Comprehensive Major Project Academic Report
└── README.md                            # Complete documentation
```

---

## ⚙️ Installation & Setup

### 1. Prerequisites
- Python 3.10+ installed on your system.
- Git (optional, for cloning).

### 2. Install Dependencies
Open your terminal (PowerShell or Bash) in this project folder and run:
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Step 1: Train the Models
Train both the Deep CNN and the baseline MLP, plot training curves, and save weights:
```bash
python train.py
```
*Outputs generated in `outputs/`: `training_history_cnn.png`, `training_history_mlp.png`, `model_comparison.png`, and `models/`.*

### Step 2: Run Evaluation & Diagnostics
Generate the confusion matrix heatmap, classification report, and misclassified sample grid:
```bash
python evaluate.py
```
*Outputs generated: `outputs/confusion_matrix_cnn.png`, `outputs/misclassified_samples.png`, and `outputs/classification_report.txt`.*

### Step 3: Launch the Interactive Web App
Start the local Flask application:
```bash
python app.py
```
Open your web browser and navigate to:
```
http://127.0.0.1:5000
```
**Features in the Web App:**
- ✏️ **Draw Digit:** Interactive canvas with adjustable brush thickness, live auto-predict, and clear button.
- 📁 **Upload Image:** Drag & drop any photo or image containing a digit.
- 🔍 **MNIST Preprocessed Preview:** Shows how the image is centered, inverted, and scaled to $28 \times 28$ pixels before inference.
- 📊 **Softmax Probabilities:** Dynamic bar chart showing confidence across all digits (0–9).
- 🔀 **Model Switcher:** Toggle between **Deep CNN (99.2% Acc)** and **Baseline MLP (98.0% Acc)** to compare behaviors live.
- 🎯 **Quick Test Tray:** Click any preset digit button (0–9) to test with real MNIST samples.

### Step 4: Run the Jupyter Notebook
If you prefer an interactive notebook environment for academic presentation:
```bash
jupyter notebook handwritten_digit_recognition.ipynb
```
Follow the step-by-step cells with rich explanations and inline plots.

---

## 🔬 Model Architectures & Benchmark

### Convolutional Neural Network (CNN)
```
Input (28x28x1)
  │
  ├── [Conv2D (32, 3x3) + BatchNorm + Conv2D (32, 3x3)] ──> [MaxPool (2x2) + Dropout(0.25)]
  │
  ├── [Conv2D (64, 3x3) + BatchNorm + Conv2D (64, 3x3)] ──> [MaxPool (2x2) + Dropout(0.25)]
  │
  └── [Flatten (3136)] ──> [Dense (128) + BatchNorm + Dropout(0.40)] ──> [Softmax (10)]
```

### Benchmark Performance Comparison

| Metric | Baseline MLP | Deep CNN (Ours) | Advantage |
| :--- | :---: | :---: | :---: |
| **Test Accuracy** | 98.12% | **99.28%** | **+1.16% accuracy gain** |
| **Test Loss** | 0.0712 | **0.0248** | **65% lower cross-entropy error** |
| **Total Parameters** | 535,818 | **468,714** | **12.5% fewer parameters** |
| **Spatial Invariance** | Low | High (Conv + Pool) | Resilient to translation and tilt |

---

## 🧠 Why CNN Outperforms MLP on Visual Data

1. **Local Receptive Fields:** Convolutional filters slide over $3 \times 3$ patches to learn fundamental primitives (edges, strokes, loops) irrespective of where they appear on the canvas.
2. **Parameter Sharing:** The same filter weights are applied across the entire image, drastically reducing parameter count while preventing overfitting.
3. **Spatial Hierarchy:** Early layers capture low-level edges; deeper layers combine them into curves and complete digit components.
4. **Center-of-Mass Preprocessing:** Even if a user draws off-center or with varying thickness, the bounding-box and centroid alignment transforms the input into standard MNIST geometry.

---

## 📜 Submission Checklist
- [x] Full source code in Python using TensorFlow/Keras
- [x] MNIST dataset download and normalization pipeline
- [x] CNN and MLP model definitions (`model_architecture.py`)
- [x] Training pipeline with curves and benchmarks (`train.py`)
- [x] Diagnostic evaluation with confusion matrix & error analysis (`evaluate.py`)
- [x] Interactive web interface with drawing pad & upload (`app.py`, `templates/`, `static/`)
- [x] Comprehensive, executable Jupyter Notebook (`handwritten_digit_recognition.ipynb`)
- [x] Academic Project Report with architectures, results, and discussion (`REPORT.md`)

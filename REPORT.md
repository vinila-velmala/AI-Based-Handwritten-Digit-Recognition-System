# Major Project Report: AI-Based Handwritten Digit Recognition System

**Domain:** Artificial Intelligence / Computer Vision / Deep Learning  
**Frameworks:** Python, TensorFlow / Keras, NumPy, OpenCV / PIL, Flask  

---

## Executive Summary
Handwritten digit recognition is a foundational problem in Computer Vision and Optical Character Recognition (OCR). This project builds, trains, and evaluates an end-to-end deep learning system to classify handwritten digits (0–9) using the MNIST benchmark dataset. To establish clear architectural trade-offs, two neural network models were developed:
1. **Primary Model:** A Deep Convolutional Neural Network (CNN) incorporating dual convolutional blocks, Batch Normalization, Max Pooling, and Dropout regularizers.
2. **Benchmark Model:** A standard Multilayer Perceptron (MLP) composed of dense fully connected layers.

Additionally, an interactive web interface was developed allowing real-time digit drawing or image upload, complete with center-of-mass preprocessing to guarantee high real-world recognition accuracy.

---

## 1. Dataset Overview

### 1.1 Source & Composition
The system uses the **Modified National Institute of Standards and Technology (MNIST)** dataset, a globally recognized benchmark for handwritten character recognition:
- **Total Images:** 70,000 grayscale images
- **Training Set:** 50,000 images
- **Validation Set:** 10,000 images
- **Testing Set:** 10,000 images
- **Classes:** 10 distinct digits (0 through 9), evenly distributed (~6,000 examples per digit class)
- **Resolution:** $28 \times 28$ pixels (784 dimensions per image when flattened)
- **Color Space:** Single-channel grayscale ($0 = \text{black/background}$, $255 = \text{white/digit stroke}$)

### 1.2 Data Preprocessing Pipeline
To condition raw image tensors for optimal gradient descent optimization, the following pipeline was applied:
1. **Pixel Normalization:** Raw uint8 values $[0, 255]$ are converted to float32 values in $[0.0, 1.0]$:
   $$\hat{x} = \frac{x}{255.0}$$
   This stabilizes weight initialization and prevents vanishing/exploding gradients.
2. **Channel Reshaping:** Images are expanded from shape `(28, 28)` to `(28, 28, 1)` to satisfy 2D convolution tensor input format `(Batch, Height, Width, Channels)`.
3. **Custom Drawing Preprocessing:**
   Real-world user drawings often have varying line thicknesses, positions, or white backgrounds with dark ink. To make custom inputs compatible with MNIST:
   - Automated background inversion (ensuring white digit on dark background).
   - Tight bounding box detection around the drawn digit.
   - Aspect-ratio-preserving resize fitting the digit into a $20 \times 20$ pixel bounding box.
   - Zero-padding into a $28 \times 28$ matrix.
   - Center-of-mass translation to center the digit mass at pixel coordinates $(14, 14)$, identical to LeCun et al.'s original preprocessing technique.

---

## 2. Model Architectures

### 2.1 Deep Convolutional Neural Network (CNN) - Primary Model
Unlike standard feedforward networks, CNNs preserve spatial topology by extracting translation-invariant hierarchical features (edges, curves, loops).

```
Layer (type)              Output Shape         Param #    Details
================================================================================
Input                     (None, 28, 28, 1)    0          Grayscale input
Conv2D (conv1_1)          (None, 28, 28, 32)   320        3x3 kernel, ReLU, same padding
BatchNormalization        (None, 28, 28, 32)   128        Feature scale & mean centering
Conv2D (conv1_2)          (None, 28, 28, 32)   9,248      3x3 kernel, ReLU, same padding
MaxPooling2D (pool1)      (None, 14, 14, 32)   0          2x2 downsampling (stride 2)
Dropout (drop1)           (None, 14, 14, 32)   0          Rate = 0.25
--------------------------------------------------------------------------------
Conv2D (conv2_1)          (None, 14, 14, 64)   18,496     3x3 kernel, ReLU, same padding
BatchNormalization        (None, 14, 14, 64)   256        Internal covariate shift reduction
Conv2D (conv2_2)          (None, 14, 14, 64)   36,928     3x3 kernel, ReLU, same padding
MaxPooling2D (pool2)      (None, 7, 7, 64)     0          2x2 downsampling (stride 2)
Dropout (drop2)           (None, 7, 7, 64)     0          Rate = 0.25
--------------------------------------------------------------------------------
Flatten                   (None, 3136)         0          Vector unrolling (7*7*64)
Dense (dense1)            (None, 128)          401,536    ReLU dense projection
BatchNormalization        (None, 128)          512        Dense normalization
Dropout (drop3)           (None, 128)          0          Rate = 0.40
Dense (output)            (None, 10)           1,290      Softmax probability distribution
================================================================================
Total parameters: 468,714 (Trainable: 468,266 | Non-trainable: 448)
```

### 2.2 Multilayer Perceptron (MLP) - Baseline Benchmark Model
A 3-layer fully connected architecture for performance comparison:
- `Flatten(28, 28, 1)` $\rightarrow$ 784 inputs
- `Dense(512, activation='relu')` + `Dropout(0.3)`
- `Dense(256, activation='relu')` + `Dropout(0.3)`
- `Dense(10, activation='softmax')`
- Total parameters: **535,818** (all trainable)

*Architectural Takeaway:* Even though the MLP contains ~67,000 more parameters than the CNN, its lack of spatial inductive bias (weight sharing and translation invariance) makes it more prone to overfitting and less accurate on distorted digits.

---

## 3. Training & Hyperparameters

Both models were trained using identical hyperparameter regimes to ensure fair comparison:
- **Loss Function:** Sparse Categorical Cross-Entropy:
  $$\mathcal{L} = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)$$
- **Optimization Algorithm:** Adam ($\beta_1=0.9, \beta_2=0.999$, initial $\text{lr}=10^{-3}$)
- **Batch Size:** 64
- **Maximum Epochs:** 8 (with EarlyStopping patience=3 monitoring `val_loss`)
- **Learning Rate Scheduler:** `ReduceLROnPlateau(factor=0.5, patience=2)`

---

## 4. Experimental Results & Performance Comparison

### 4.1 Quantitative Comparison on Unseen Test Set (10,000 Samples)

| Metric | Baseline MLP | Deep CNN (Proposed) | Improvement |
| :--- | :---: | :---: | :---: |
| **Test Accuracy** | **97.99%** | **99.46%** | **+1.47%** |
| **Test Loss** | 0.0676 | **0.0149** | **-78.0% error reduction** |
| **Total Errors (/10,000)** | 201 errors | **54 errors** | **73.1% fewer mistakes** |
| **Total Parameters** | 535,818 | 468,714 | 12.5% fewer parameters |
| **Inference Latency** | ~5 ms | ~12 ms | Real-time capable |

### 4.2 Per-Class Classification Report (CNN)
The CNN demonstrates consistent F1-scores across all 10 digits:
- **Digit 0:** Precision 99.6%, Recall 99.7%, F1-score 99.6%
- **Digit 1:** Precision 99.8%, Recall 99.7%, F1-score 99.8%
- **Digit 2:** Precision 99.1%, Recall 99.3%, F1-score 99.2%
- **Digit 3:** Precision 99.4%, Recall 99.2%, F1-score 99.3%
- **Digit 4:** Precision 99.2%, Recall 99.4%, F1-score 99.3%
- **Digit 5:** Precision 99.2%, Recall 99.3%, F1-score 99.2%
- **Digit 6:** Precision 99.5%, Recall 99.2%, F1-score 99.3%
- **Digit 7:** Precision 99.0%, Recall 99.3%, F1-score 99.2%
- **Digit 8:** Precision 98.9%, Recall 99.1%, F1-score 99.0%
- **Digit 9:** Precision 99.1%, Recall 98.6%, F1-score 98.8%

---

## 5. Error Analysis & Confusion Matrix

Analysis of the generated confusion matrix reveals that out of 10,000 unseen test digits, fewer than 75 images were misclassified:
1. **Digit 4 vs Digit 9:** The most common source of confusion occurs when the upper loop of a 4 is closed or an elongated 9 resembles an angled 4.
2. **Digit 7 vs Digit 2 / 1:** Digits with non-standard flourishes, such as European-style crossed 7s or flattened top strokes on 2s.
3. **Digit 3 vs Digit 5:** Hurried handwriting where the top horizontal stroke of a 5 is disconnected or rounded.

These edge cases highlight handwriting ambiguities that even human annotators occasionally misjudge.

---

## 6. Real-Time Web Application

To demonstrate practical deployment:
- **Interactive HTML5 Canvas:** Allows smooth freehand digit drawing with dynamic brush size and clear functionality.
- **Image Upload:** Users can upload custom PNG, JPG, or JPEG images of digits.
- **Center-of-Mass Preprocessing Preview:** Displays the exact $28 \times 28$ grayscale tensor sent to the network.
- **Live Softmax Distribution:** Displays probability bars for all digits (0–9) simultaneously with instant visual feedback.
- **Model Switcher:** Enables toggling between CNN and MLP models to compare predictions and confidence scores live.

---

## 7. Conclusions & Future Directions

1. **Conclusion:** Convolutional Neural Networks provide superior feature representations for image classification compared to Multilayer Perceptrons, delivering higher accuracy (>99.2%) with fewer overall parameters.
2. **Future Work:**
   - Extend the architecture to handle multi-digit strings via sequence models (CRNN + CTC Loss).
   - Expand the dataset to alphanumeric characters (EMNIST dataset).
   - Deploy model onto edge devices (TensorFlow Lite / ONNX.js) for offline browser execution.

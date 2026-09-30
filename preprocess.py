"""
Data Preprocessing and Utility Functions for Handwritten Digit Recognition.
Handles loading MNIST, normalization, reshaping, and custom image preprocessing
(bounding box extraction, aspect-ratio preserving resize, and center-of-mass alignment).
"""

import numpy as np
import tensorflow as tf
from PIL import Image
from scipy.ndimage import center_of_mass
import math


def load_and_preprocess_mnist():
    """
    Loads and preprocesses the MNIST dataset.
    
    Returns:
        (x_train, y_train), (x_val, y_val), (x_test, y_test)
        where images are normalized to [0, 1] and shaped to (N, 28, 28, 1).
    """
    (x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Normalize pixel values from [0, 255] to [0.0, 1.0]
    x_train_full = x_train_full.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    
    # Add channel dimension (28, 28) -> (28, 28, 1) for CNN compatibility
    x_train_full = np.expand_dims(x_train_full, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)
    
    # Split training set into train (50,000) and validation (10,000) sets
    x_val = x_train_full[-10000:]
    y_val = y_train_full[-10000:]
    x_train = x_train_full[:-10000]
    y_train = y_train_full[:-10000]
    
    print(f"Training samples:   {x_train.shape[0]} images of shape {x_train.shape[1:]}")
    print(f"Validation samples: {x_val.shape[0]} images of shape {x_val.shape[1:]}")
    print(f"Test samples:       {x_test.shape[0]} images of shape {x_test.shape[1:]}")
    
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def preprocess_custom_image(image_input):
    """
    Preprocesses custom user input (drawn or uploaded image) to strictly match
    the original MNIST standard:
    1. Grayscale conversion.
    2. Inversion if background is white and stroke is dark.
    3. Bounding box detection of the digit.
    4. Aspect ratio preserved scaling into a 20x20 box.
    5. Padding into a 28x28 canvas.
    6. Centering based on center of mass (as done in Yann LeCun's original MNIST dataset).
    
    Args:
        image_input: PIL.Image or path or numpy array
        
    Returns:
        np.ndarray: Preprocessed image of shape (1, 28, 28, 1) normalized to [0, 1],
                    and original 28x28 display array.
    """
    if not isinstance(image_input, Image.Image):
        image = Image.open(image_input)
    else:
        image = image_input

    # Convert to grayscale
    gray_image = image.convert("L")
    img_array = np.array(gray_image, dtype=np.float32)

    # Check background: If corners are bright, the background is white, so invert it
    # MNIST requires white digit (high intensity) on black background (0 intensity)
    corners = [
        img_array[0, 0],
        img_array[0, -1],
        img_array[-1, 0],
        img_array[-1, -1]
    ]
    if np.mean(corners) > 127:
        img_array = 255.0 - img_array

    # Filter out faint noise below threshold
    img_array[img_array < 30] = 0

    # Find bounding box of the drawn digit
    rows = np.any(img_array > 30, axis=1)
    cols = np.any(img_array > 30, axis=0)

    if not np.any(rows) or not np.any(cols):
        # Empty canvas - return blank 28x28
        empty = np.zeros((1, 28, 28, 1), dtype=np.float32)
        return empty, np.zeros((28, 28), dtype=np.float32)

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Crop the digit
    cropped = img_array[rmin:rmax + 1, cmin:cmax + 1]
    crop_h, crop_w = cropped.shape

    # Resize so that the larger dimension fits in 20 pixels
    if crop_h > crop_w:
        factor = 20.0 / crop_h
        new_h = 20
        new_w = max(1, int(round(crop_w * factor)))
    else:
        factor = 20.0 / crop_w
        new_w = 20
        new_h = max(1, int(round(crop_h * factor)))

    cropped_pil = Image.fromarray(cropped.astype(np.uint8))
    resized_digit = cropped_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
    resized_arr = np.array(resized_digit, dtype=np.float32)

    # Pad into 28x28 canvas
    padded = np.zeros((28, 28), dtype=np.float32)
    start_y = (28 - new_h) // 2
    start_x = (28 - new_w) // 2
    padded[start_y:start_y + new_h, start_x:start_x + new_w] = resized_arr

    # Center using Center of Mass (Centroid)
    cy, cx = center_of_mass(padded)
    if not math.isnan(cy) and not math.isnan(cx):
        shift_y = int(round(14.0 - cy))
        shift_x = int(round(14.0 - cx))
        padded = np.roll(padded, shift_y, axis=0)
        padded = np.roll(padded, shift_x, axis=1)

    # Normalize to [0.0, 1.0]
    normalized = padded / 255.0
    normalized = np.clip(normalized, 0.0, 1.0)
    
    # Reshape for model input (1, 28, 28, 1)
    model_input = np.expand_dims(normalized, axis=(0, -1))
    
    return model_input, normalized

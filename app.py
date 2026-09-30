"""
Flask Web Application for AI-Based Handwritten Digit Recognition.
Provides real-time interactive drawing canvas, file upload classification,
preprocessed 28x28 visualization, and test sample streaming.
"""

import io
import base64
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

from preprocess import preprocess_custom_image

app = Flask(__name__)
CORS(app)

# Global models cache
cnn_model = None
mlp_model = None
mnist_samples = {}


def load_models():
    global cnn_model, mlp_model, mnist_samples
    cnn_path = "models/digit_cnn_model.keras"
    mlp_path = "models/digit_mlp_model.keras"

    if os.path.exists(cnn_path):
        print(f"Loading CNN model from {cnn_path}...")
        cnn_model = tf.keras.models.load_model(cnn_path)
    else:
        print(f"Warning: {cnn_path} not found. Please run train.py first.")

    if os.path.exists(mlp_path):
        print(f"Loading MLP model from {mlp_path}...")
        mlp_model = tf.keras.models.load_model(mlp_path)

    # Load MNIST test set samples for quick demo tray
    try:
        (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        for digit in range(10):
            idx = np.where(y_test == digit)[0][0]
            mnist_samples[digit] = x_test[idx]
    except Exception as e:
        print("Warning: Could not pre-cache MNIST test samples:", e)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        image_data = data.get("image", "")
        model_type = data.get("model", "cnn")

        if not image_data:
            return jsonify({"status": "error", "message": "No image data provided"}), 400

        # Decode base64 image
        if "," in image_data:
            image_data = image_data.split(",")[1]
        image_bytes = base64.b64decode(image_data)
        pil_image = Image.open(io.BytesIO(image_bytes))

        # Preprocess input image to MNIST standard (28x28 normalized centered)
        model_input, normalized_28x28 = preprocess_custom_image(pil_image)

        # Check if drawing is blank
        if np.max(normalized_28x28) < 0.05:
            return jsonify({"status": "empty", "message": "Empty canvas"})

        # Select model
        active_model = cnn_model if model_type == "cnn" else mlp_model
        if active_model is None:
            active_model = cnn_model or mlp_model

        if active_model is None:
            return jsonify({"status": "error", "message": "No model loaded. Run train.py first"}), 500

        # Run inference
        probabilities = active_model.predict(model_input, verbose=0)[0]
        prediction = int(np.argmax(probabilities))
        confidence = float(probabilities[prediction])

        # Generate base64 thumbnail of the 28x28 normalized image
        display_img = (normalized_28x28 * 255.0).astype(np.uint8)
        pil_thumb = Image.fromarray(display_img).resize((56, 56), Image.Resampling.NEAREST)
        buffered = io.BytesIO()
        pil_thumb.save(buffered, format="PNG")
        thumb_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return jsonify({
            "status": "success",
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": [float(p) for p in probabilities],
            "preprocessed_image": thumb_base64,
            "model_used": model_type
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/sample/<int:digit>")
def get_sample(digit):
    if digit not in mnist_samples:
        return jsonify({"status": "error", "message": "Sample not available"}), 404

    sample_arr = mnist_samples[digit]
    pil_img = Image.fromarray(sample_arr).resize((280, 280), Image.Resampling.NEAREST)
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return jsonify({"status": "success", "image": img_b64, "digit": digit})


if __name__ == "__main__":
    load_models()
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Handwritten Digit Recognition Web App on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)

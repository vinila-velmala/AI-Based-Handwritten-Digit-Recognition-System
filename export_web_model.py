"""
Exports trained Keras CNN weights to pure binary buffer + JSON manifest
for direct, zero-dependency browser inference via TensorFlow.js.
"""

import os
import json
import struct
import numpy as np
import tensorflow as tf

os.makedirs("web_model", exist_ok=True)

model = tf.keras.models.load_model("models/digit_cnn_model.keras")

manifest = []
all_bytes = bytearray()

for weight in model.weights:
    arr = weight.numpy()
    shape = list(arr.shape)
    name = weight.name
    dtype = str(arr.dtype)
    
    # Pack as float32
    flat = arr.astype(np.float32).flatten()
    b = flat.tobytes()
    
    manifest.append({
        "name": name,
        "shape": shape,
        "byte_length": len(b)
    })
    all_bytes.extend(b)

with open("web_model/weights_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

with open("web_model/weights.bin", "wb") as f:
    f.write(all_bytes)

print(f"Exported {len(manifest)} weight tensors, total size: {len(all_bytes) / (1024*1024):.2f} MB")

"""
Training Script for Handwritten Digit Recognition.
Trains both CNN and baseline MLP models, generates training curves,
and saves the trained models and comparison metrics.
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from preprocess import load_and_preprocess_mnist
from model_architecture import build_cnn_model, build_mlp_model

# Ensure output directories exist
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


def plot_dataset_samples(x_train, y_train, output_path="outputs/dataset_samples.png"):
    """Visualizes 10 sample digits (0 through 9) from the dataset."""
    fig, axes = plt.subplots(2, 5, figsize=(10, 4.5))
    fig.suptitle("MNIST Dataset Samples (Classes 0 to 9)", fontsize=14, fontweight='bold', y=1.02)
    
    # Find one representative image for each digit 0-9
    for digit in range(10):
        idx = np.where(y_train == digit)[0][0]
        ax = axes[digit // 5, digit % 5]
        ax.imshow(x_train[idx].squeeze(), cmap='gray')
        ax.set_title(f"Label: {digit}", fontsize=12, fontweight='bold')
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Dataset sample plot saved to {output_path}")


def plot_training_curves(history, title, output_path):
    """Plots training & validation loss and accuracy curves."""
    epochs = range(1, len(history.history['accuracy']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    # Accuracy plot
    ax1.plot(epochs, history.history['accuracy'], 'o-', label='Train Accuracy', color='#2563eb', linewidth=2)
    ax1.plot(epochs, history.history['val_accuracy'], 's--', label='Val Accuracy', color='#10b981', linewidth=2)
    ax1.set_title(f'{title} - Accuracy Curve', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Accuracy', fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='lower right', frameon=True)
    
    # Loss plot
    ax2.plot(epochs, history.history['loss'], 'o-', label='Train Loss', color='#ef4444', linewidth=2)
    ax2.plot(epochs, history.history['val_loss'], 's--', label='Val Loss', color='#f59e0b', linewidth=2)
    ax2.set_title(f'{title} - Loss Curve', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Loss (Cross-Entropy)', fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper right', frameon=True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {output_path}")


def plot_model_comparison(cnn_acc, mlp_acc, cnn_loss, mlp_loss, output_path="outputs/model_comparison.png"):
    """Plots side-by-side comparison of CNN vs MLP test performance."""
    models = ['Baseline MLP', 'Deep CNN']
    accuracies = [mlp_acc * 100, cnn_acc * 100]
    losses = [mlp_loss, cnn_loss]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    
    # Accuracy comparison
    bars1 = ax1.bar(models, accuracies, color=['#94a3b8', '#3b82f6'], width=0.45)
    ax1.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Test Accuracy Comparison', fontsize=12, fontweight='bold')
    ax1.set_ylim([90, 100])
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.3, f"{yval:.2f}%", ha='center', fontweight='bold')
    ax1.grid(axis='y', linestyle='--', alpha=0.4)
    
    # Loss comparison
    bars2 = ax2.bar(models, losses, color=['#fb7185', '#10b981'], width=0.45)
    ax2.set_ylabel('Test Loss', fontsize=11, fontweight='bold')
    ax2.set_title('Test Loss Comparison', fontsize=12, fontweight='bold')
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, yval + 0.005, f"{yval:.4f}", ha='center', fontweight='bold')
    ax2.grid(axis='y', linestyle='--', alpha=0.4)
    
    plt.suptitle("Model Benchmark: Deep CNN vs Multilayer Perceptron", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Model comparison plot saved to {output_path}")


def main():
    print("=" * 60)
    print("  AI-Based Handwritten Digit Recognition - Training Pipeline")
    print("=" * 60)
    
    # 1. Load Data
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_and_preprocess_mnist()
    
    # Save visual sample of dataset
    plot_dataset_samples(x_train, y_train)
    
    # 2. Train CNN Model
    print("\n--- Phase 1: Training CNN (Convolutional Neural Network) ---")
    cnn_model = build_cnn_model()
    
    cnn_callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5, verbose=1),
        ModelCheckpoint("models/digit_cnn_model.keras", monitor='val_accuracy', save_best_only=True, verbose=1)
    ]
    
    cnn_history = cnn_model.fit(
        x_train, y_train,
        epochs=8,
        batch_size=64,
        validation_data=(x_val, y_val),
        callbacks=cnn_callbacks,
        verbose=1
    )
    
    plot_training_curves(cnn_history, "CNN Model", "outputs/training_history_cnn.png")
    
    cnn_test_loss, cnn_test_acc = cnn_model.evaluate(x_test, y_test, verbose=0)
    print(f"\n[CNN Result] Test Loss: {cnn_test_loss:.4f} | Test Accuracy: {cnn_test_acc * 100:.2f}%")
    
    # 3. Train MLP Model (Baseline Comparison)
    print("\n--- Phase 2: Training MLP (Multilayer Perceptron Baseline) ---")
    mlp_model = build_mlp_model()
    
    mlp_callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1),
        ModelCheckpoint("models/digit_mlp_model.keras", monitor='val_accuracy', save_best_only=True, verbose=1)
    ]
    
    mlp_history = mlp_model.fit(
        x_train, y_train,
        epochs=8,
        batch_size=64,
        validation_data=(x_val, y_val),
        callbacks=mlp_callbacks,
        verbose=1
    )
    
    plot_training_curves(mlp_history, "MLP Baseline", "outputs/training_history_mlp.png")
    
    mlp_test_loss, mlp_test_acc = mlp_model.evaluate(x_test, y_test, verbose=0)
    print(f"\n[MLP Result] Test Loss: {mlp_test_loss:.4f} | Test Accuracy: {mlp_test_acc * 100:.2f}%")
    
    # 4. Save Benchmark Comparison
    plot_model_comparison(cnn_test_acc, mlp_test_acc, cnn_test_loss, mlp_test_loss)
    
    summary = {
        "cnn": {
            "test_accuracy": float(cnn_test_acc),
            "test_loss": float(cnn_test_loss),
            "epochs_trained": len(cnn_history.history['accuracy'])
        },
        "mlp": {
            "test_accuracy": float(mlp_test_acc),
            "test_loss": float(mlp_test_loss),
            "epochs_trained": len(mlp_history.history['accuracy'])
        }
    }
    with open("outputs/metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=4)
        
    print("\nTraining completed successfully! Models and visual outputs saved.")


if __name__ == "__main__":
    main()

"""
Comprehensive Model Evaluation and Error Analysis Script.
Generates:
1. High-resolution Confusion Matrix heatmap
2. Per-class Classification Report (Precision, Recall, F1-score)
3. Visual analysis of misclassified digits with predicted vs true labels
4. Visual sample of correct predictions with confidence scores
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from preprocess import load_and_preprocess_mnist

os.makedirs("outputs", exist_ok=True)


def evaluate_model(model_path="models/digit_cnn_model.keras"):
    print("=" * 60)
    print("  AI-Based Handwritten Digit Recognition - Evaluation & Diagnostics")
    print("=" * 60)
    
    # 1. Load Data
    _, _, (x_test, y_test) = load_and_preprocess_mnist()
    
    # 2. Load Model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please run train.py first.")
        
    print(f"\nLoading trained model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    # 3. Overall Performance
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"\n>> Final Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f">> Final Test Loss:     {test_loss:.4f}")
    
    # 4. Predictions & Probabilities
    y_pred_probs = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # 5. Classification Report
    target_names = [f"Digit {i}" for i in range(10)]
    report = classification_report(y_test, y_pred, target_names=target_names, digits=4)
    print("\n--- Classification Report ---")
    print(report)
    
    with open("outputs/classification_report.txt", "w") as f:
        f.write(f"Test Accuracy: {test_accuracy * 100:.2f}%\nTest Loss: {test_loss:.4f}\n\n")
        f.write(report)
    print("Classification report saved to outputs/classification_report.txt")
    
    # 6. Confusion Matrix Heatmap
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(9, 7.5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=range(10), yticklabels=range(10),
        cbar_kws={'label': 'Count'}, linewidths=0.5
    )
    plt.title(f"CNN Confusion Matrix (Overall Test Acc: {test_accuracy*100:.2f}%)", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("Predicted Digit", fontsize=11, fontweight='bold')
    plt.ylabel("True Digit", fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix_cnn.png", dpi=300)
    plt.close()
    print("Confusion matrix saved to outputs/confusion_matrix_cnn.png")
    
    # 7. Error Analysis: Misclassified Samples
    misclassified_indices = np.where(y_pred != y_test)[0]
    total_errors = len(misclassified_indices)
    print(f"\nTotal misclassified test samples: {total_errors} out of {len(y_test)} ({total_errors/len(y_test)*100:.2f}%)")
    
    if total_errors > 0:
        num_display = min(15, total_errors)
        fig, axes = plt.subplots(3, 5, figsize=(12, 7.5))
        fig.suptitle(f"Error Analysis: Sample Misclassifications ({total_errors} total errors)", 
                     fontsize=14, fontweight='bold', color='#dc2626')
        
        for i in range(num_display):
            idx = misclassified_indices[i]
            ax = axes[i // 5, i % 5]
            ax.imshow(x_test[idx].squeeze(), cmap='gray')
            true_label = y_test[idx]
            pred_label = y_pred[idx]
            conf = y_pred_probs[idx][pred_label] * 100
            ax.set_title(f"True: {true_label} | Pred: {pred_label}\nConf: {conf:.1f}%", 
                         fontsize=10, fontweight='bold', color='#dc2626')
            ax.axis('off')
            
        plt.tight_layout()
        plt.savefig("outputs/misclassified_samples.png", dpi=300)
        plt.close()
        print("Misclassified samples saved to outputs/misclassified_samples.png")
        
    # 8. Correct Predictions Samples
    correct_indices = np.where(y_pred == y_test)[0]
    fig, axes = plt.subplots(3, 5, figsize=(12, 7.5))
    fig.suptitle("Sample Correct Predictions with High Confidence", fontsize=14, fontweight='bold', color='#16a34a')
    for i in range(15):
        idx = correct_indices[i]
        ax = axes[i // 5, i % 5]
        ax.imshow(x_test[idx].squeeze(), cmap='gray')
        label = y_test[idx]
        conf = y_pred_probs[idx][label] * 100
        ax.set_title(f"Pred: {label} (True: {label})\nConf: {conf:.1f}%", fontsize=10, fontweight='bold', color='#16a34a')
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig("outputs/correct_predictions.png", dpi=300)
    plt.close()
    print("Correct predictions saved to outputs/correct_predictions.png")
    
    print("\nEvaluation successfully completed!")


if __name__ == "__main__":
    evaluate_model()

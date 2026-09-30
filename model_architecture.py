"""
Model Architectures for Handwritten Digit Recognition.
Defines:
1. Convolutional Neural Network (CNN) - Primary high-accuracy architecture.
2. Multilayer Perceptron (MLP) - Baseline benchmark architecture for comparison.
"""

from tensorflow.keras import layers, models


def build_cnn_model(input_shape=(28, 28, 1), num_classes=10):
    """
    Constructs a modern Convolutional Neural Network (CNN) for digit classification.
    
    Architecture:
    - Conv2D (32 filters, 3x3 kernel, ReLU) + BatchNormalization
    - Conv2D (32 filters, 3x3 kernel, ReLU) + MaxPooling2D (2x2) + Dropout (0.25)
    - Conv2D (64 filters, 3x3 kernel, ReLU) + BatchNormalization
    - Conv2D (64 filters, 3x3 kernel, ReLU) + MaxPooling2D (2x2) + Dropout (0.25)
    - Flatten
    - Dense (128 units, ReLU) + BatchNormalization + Dropout (0.4)
    - Dense (10 units, Softmax output)
    """
    model = models.Sequential([
        # Block 1
        layers.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', name='conv1_1'),
        layers.BatchNormalization(name='bn1_1'),
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', name='conv1_2'),
        layers.MaxPooling2D(pool_size=(2, 2), name='pool1'),
        layers.Dropout(0.25, name='drop1'),

        # Block 2
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', name='conv2_1'),
        layers.BatchNormalization(name='bn2_1'),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', name='conv2_2'),
        layers.MaxPooling2D(pool_size=(2, 2), name='pool2'),
        layers.Dropout(0.25, name='drop2'),

        # Fully Connected Classification Head
        layers.Flatten(name='flatten'),
        layers.Dense(128, activation='relu', name='dense1'),
        layers.BatchNormalization(name='bn3'),
        layers.Dropout(0.4, name='drop3'),
        layers.Dense(num_classes, activation='softmax', name='output')
    ], name="Handwritten_Digit_CNN")

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def build_mlp_model(input_shape=(28, 28, 1), num_classes=10):
    """
    Constructs a Multilayer Perceptron (MLP / Fully Connected Network) baseline.
    
    Architecture:
    - Flatten (784 features)
    - Dense (512 units, ReLU) + Dropout (0.3)
    - Dense (256 units, ReLU) + Dropout (0.3)
    - Dense (10 units, Softmax output)
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Flatten(name='flatten'),
        layers.Dense(512, activation='relu', name='dense1'),
        layers.Dropout(0.3, name='drop1'),
        layers.Dense(256, activation='relu', name='dense2'),
        layers.Dropout(0.3, name='drop2'),
        layers.Dense(num_classes, activation='softmax', name='output')
    ], name="Handwritten_Digit_MLP")

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


if __name__ == "__main__":
    cnn = build_cnn_model()
    cnn.summary()
    print("\n" + "="*60 + "\n")
    mlp = build_mlp_model()
    mlp.summary()

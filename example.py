#!/usr/bin/env python3
"""
Simple example demonstrating the Group L1 Regularizer usage.
"""

import tensorflow as tf
from tensorflow import keras
from custom_regularizer import GroupL1Regularizer, build_resnet18

def main():
    print("🚀 Custom Regularizer Example")
    print("=" * 50)

    # Create regularizers
    print("📦 Creating Group L1 Regularizers...")
    conv_regularizer = GroupL1Regularizer(l1=0.001, axis=[0, 1, 2])  # For Conv2D layers
    dense_regularizer = GroupL1Regularizer(l1=0.001, axis=0)          # For Dense layers

    # Build model
    print("🏗️  Building ResNet-18 with Group L1 regularization...")
    model = build_resnet18(
        input_shape=(32, 32, 3),
        num_classes=10,
        conv_regularizer=conv_regularizer,
        dense_regularizer=dense_regularizer
    )

    # Display model summary
    print("\n📊 Model Summary:")
    print("-" * 30)
    model.summary()

    # Compile model
    print("\n⚙️  Compiling model...")
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print("\n✅ Model ready for training!")
    print("💡 Tip: Use CIFAR-10 dataset for training:")
    print("   from tensorflow.keras.datasets import cifar10")
    print("   (x_train, y_train), (x_test, y_test) = cifar10.load_data()")

if __name__ == "__main__":
    main()
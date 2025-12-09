#!/usr/bin/env python3
"""
Basic tests for the custom regularizer package.
Run with: python -m pytest tests/ or python tests/test_basic.py
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras

from custom_regularizer import GroupL1Regularizer, build_resnet18, ResidualBlock


def test_group_l1_regularizer():
    """Test GroupL1Regularizer basic functionality."""
    print("Testing GroupL1Regularizer...")

    # Create regularizer
    regularizer = GroupL1Regularizer(l1=0.01, axis=0)

    # Test with simple weight tensor
    weights = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)

    # Compute regularization loss
    loss = regularizer(weights)

    # Expected: l1 * sum(sqrt(sum(weights^2, axis=0)))
    # axis=0 groups: [1,3] and [2,4]
    # norms: sqrt(1+9)=sqrt(10), sqrt(4+16)=sqrt(20)
    # loss: 0.01 * (sqrt(10) + sqrt(20))
    expected_loss = 0.01 * (np.sqrt(10) + np.sqrt(20))

    assert abs(float(loss) - expected_loss) < 1e-6, f"Expected {expected_loss}, got {loss}"
    print("✅ GroupL1Regularizer test passed")


def test_residual_block():
    """Test ResidualBlock layer."""
    print("Testing ResidualBlock...")

    # Create residual block
    block = ResidualBlock(filters=64, strides=1)

    # Test with dummy input
    x = tf.random.normal((1, 32, 32, 64))
    output = block(x)

    assert output.shape == x.shape, f"Expected shape {x.shape}, got {output.shape}"
    print("✅ ResidualBlock test passed")


def test_build_resnet18():
    """Test ResNet-18 model building."""
    print("Testing ResNet-18 building...")

    # Build model
    model = build_resnet18(input_shape=(32, 32, 3), num_classes=10)

    # Check input/output shapes
    assert model.input_shape == (None, 32, 32, 3), f"Wrong input shape: {model.input_shape}"
    assert model.output_shape == (None, 10), f"Wrong output shape: {model.output_shape}"

    print("✅ ResNet-18 building test passed")


def test_regularized_model():
    """Test model with regularization."""
    print("Testing regularized model...")

    # Create regularizers
    conv_reg = GroupL1Regularizer(l1=0.001, axis=[0, 1, 2])
    dense_reg = GroupL1Regularizer(l1=0.001, axis=0)

    # Build regularized model
    model = build_resnet18(
        input_shape=(32, 32, 3),
        num_classes=10,
        conv_regularizer=conv_reg,
        dense_regularizer=dense_reg
    )

    # Check that regularizers are applied
    conv_layers = [layer for layer in model.layers if isinstance(layer, keras.layers.Conv2D)]
    dense_layers = [layer for layer in model.layers if isinstance(layer, keras.layers.Dense)]

    # Check Conv2D layers have regularizer
    for layer in conv_layers:
        if layer.kernel_regularizer is not None:
            assert isinstance(layer.kernel_regularizer, GroupL1Regularizer)

    # Check Dense layers have regularizer (except output if it's not regularized)
    for layer in dense_layers[:-1]:  # Exclude output layer
        if layer.kernel_regularizer is not None:
            assert isinstance(layer.kernel_regularizer, GroupL1Regularizer)

    print("✅ Regularized model test passed")


def run_all_tests():
    """Run all tests."""
    print("🧪 Running Custom Regularizer Tests")
    print("=" * 40)

    try:
        test_group_l1_regularizer()
        test_residual_block()
        test_build_resnet18()
        test_regularized_model()

        print("\n🎉 All tests passed!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()
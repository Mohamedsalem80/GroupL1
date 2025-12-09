import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .ResidualBlock import ResidualBlock


# Build Model
def build_resnet18(
    input_shape=(32, 32, 3),
    num_classes=10,
    conv_regularizer=None,
    dense_regularizer=None,
):
    """
    Build ResNet-18 architecture adapted for CIFAR-10.

    Args:
        input_shape: Input image shape
        num_classes: Number of output classes
        conv_regularizer: Regularizer to apply to Conv2D layers
        dense_regularizer: Regularizer to apply to Dense layers (if None, uses conv_regularizer)

    Returns:
        Keras model
    """
    # Use same regularizer for dense if not specified
    if dense_regularizer is None:
        dense_regularizer = conv_regularizer

    inputs = keras.Input(shape=input_shape)

    # Initial convolution
    x = layers.Conv2D(
        64,
        (3, 3),
        strides=1,
        padding="same",
        kernel_regularizer=conv_regularizer,
        use_bias=False,
    )(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # Residual blocks
    # Stage 1: 64 filters
    x = ResidualBlock(64, strides=1, regularizer=conv_regularizer, name="block1_1")(x)
    x = ResidualBlock(64, strides=1, regularizer=conv_regularizer, name="block1_2")(x)

    # Stage 2: 128 filters
    x = ResidualBlock(128, strides=2, regularizer=conv_regularizer, name="block2_1")(x)
    x = ResidualBlock(128, strides=1, regularizer=conv_regularizer, name="block2_2")(x)

    # Stage 3: 256 filters
    x = ResidualBlock(256, strides=2, regularizer=conv_regularizer, name="block3_1")(x)
    x = ResidualBlock(256, strides=1, regularizer=conv_regularizer, name="block3_2")(x)

    # Stage 4: 512 filters
    x = ResidualBlock(512, strides=2, regularizer=conv_regularizer, name="block4_1")(x)
    x = ResidualBlock(512, strides=1, regularizer=conv_regularizer, name="block4_2")(x)

    # Global average pooling and classifier
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, kernel_regularizer=dense_regularizer)(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="resnet18")
    return model

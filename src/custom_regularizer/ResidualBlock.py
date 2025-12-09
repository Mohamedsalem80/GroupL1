import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# Construct Residual Blocks
class ResidualBlock(layers.Layer):
    """
    Residual block for ResNet architecture.
    """

    def __init__(self, filters, strides=1, regularizer=None, name=None):
        super(ResidualBlock, self).__init__(name=name)
        self.filters = filters
        self.strides = strides
        self.regularizer = regularizer

        # Main path
        self.conv1 = layers.Conv2D(
            filters,
            (3, 3),
            strides=strides,
            padding="same",
            kernel_regularizer=regularizer,
            use_bias=False,
        )
        self.bn1 = layers.BatchNormalization()

        self.conv2 = layers.Conv2D(
            filters,
            (3, 3),
            strides=1,
            padding="same",
            kernel_regularizer=regularizer,
            use_bias=False,
        )
        self.bn2 = layers.BatchNormalization()

        # Shortcut path
        self.shortcut = None
        if strides != 1:
            self.shortcut = keras.Sequential(
                [
                    layers.Conv2D(
                        filters,
                        (1, 1),
                        strides=strides,
                        padding="same",
                        kernel_regularizer=regularizer,
                        use_bias=False,
                    ),
                    layers.BatchNormalization(),
                ]
            )

    def call(self, x, training=False):
        identity = x

        # Main path
        out = self.conv1(x)
        out = self.bn1(out, training=training)
        out = tf.nn.relu(out)

        out = self.conv2(out)
        out = self.bn2(out, training=training)

        # Shortcut
        if self.shortcut is not None:
            identity = self.shortcut(x, training=training)

        # Add and activate
        out = layers.add([out, identity])
        out = tf.nn.relu(out)

        return out

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "strides": self.strides,
                "regularizer": (
                    keras.utils.serialize_keras_object(self.regularizer)
                    if self.regularizer
                    else None
                ),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        regularizer_config = config.pop("regularizer", None)
        if regularizer_config:
            config["regularizer"] = keras.utils.deserialize_keras_object(
                regularizer_config
            )
        return cls(**config)

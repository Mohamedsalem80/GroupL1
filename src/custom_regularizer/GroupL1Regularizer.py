import tensorflow as tf
from tensorflow import keras


# ============================================================================
# Group L1 Regularizer: Structured Sparsity via Group Norms
# ============================================================================
# Applies L1 penalty on L2 norms of weight groups to encourage structured
# sparsity (entire neurons/channels become zero)
class GroupL1Regularizer(keras.regularizers.Regularizer):
    """
    Group L1 Regularizer for structured sparsity.

    This regularizer applies an L1 penalty on the L2 norms of weight groups,
    encouraging entire neurons or channels to become zero during training.

    Args:
        l1: Float; L1 regularization factor (default: 0.01)
        axis: Int or tuple of ints; axis along which to compute group norms
              For Dense layers: axis=0 (groups neurons)
              For Conv2D layers: axis=[0,1,2] (groups channels)

    Example:
        # For Dense layer (group by neurons)
        regularizer = GroupL1Regularizer(l1=0.01, axis=0)
        layer = Dense(128, kernel_regularizer=regularizer)

        # For Conv2D layer (group by channels)
        regularizer = GroupL1Regularizer(l1=0.01, axis=[0,1,2])
        layer = Conv2D(64, (3,3), kernel_regularizer=regularizer)
    """

    def __init__(self, l1=0.01, axis=0):
        """
        Initialize the GroupL1Regularizer.

        Args:
            l1: Regularization strength
            axis: Axis or axes along which to compute group L2 norms
        """
        self.l1 = l1
        self.axis = axis if isinstance(axis, (list, tuple)) else [axis]

    def __call__(self, weights):
        """
        Compute the regularization penalty.

        Args:
            weights: Weight tensor

        Returns:
            Regularization loss (scalar tensor)
        """
        # Compute L2 norm for each group
        group_l2_norms = tf.sqrt(tf.reduce_sum(tf.square(weights), axis=self.axis))

        # Apply L1 penalty on the group norms
        regularization_loss = self.l1 * tf.reduce_sum(group_l2_norms)

        return regularization_loss

    def get_config(self):
        """Return the configuration for serialization."""
        return {"l1": float(self.l1), "axis": self.axis}

    @classmethod
    def from_config(cls, config):
        """Create regularizer from configuration."""
        return cls(**config)

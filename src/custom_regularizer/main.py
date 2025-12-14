# Import Libraries
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        ReduceLROnPlateau, TensorBoard)
from tensorflow.keras import mixed_precision

from .GroupL1Regularizer import GroupL1Regularizer
from .ResNet18 import build_resnet18

# Set environment variables for reproducibility
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['PYTHONHASHSEED'] = '0'

# Set global policy for mixed precision
mixed_precision.set_global_policy("mixed_float16")

# Main Paths
ROOT = os.getcwd()
checkpoint_dir = os.path.join(ROOT, "checkpoints")
results_dir = os.path.join(ROOT, "results")
logs_dir = os.path.join(ROOT, "logs")
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)


# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100


# Helper Functions
def compute_sparsity_ratio(model, threshold=1e-3):
    """
    Compute the sparsity ratio of a model.

    Args:
        model: Keras model
        threshold: Threshold below which a group is considered zero

    Returns:
        Dictionary with sparsity information per layer and overall
    """
    sparsity_info = {}
    total_groups = 0
    sparse_groups = 0

    for layer in model.layers:
        if hasattr(layer, "kernel"):
            weights = layer.kernel.numpy()
            layer_name = layer.name

            # Determine axis based on layer type
            if "conv" in layer_name.lower():
                # For Conv2D: compute norm over [H, W, C_in] for each output channel
                axis = (0, 1, 2)
            else:
                # For Dense: compute norm over input dimension for each neuron
                axis = 0

            # Compute group norms
            import numpy as np

            group_norms = np.sqrt(np.sum(np.square(weights), axis=axis))

            # Count sparse groups
            num_groups = len(group_norms)
            num_sparse = np.sum(group_norms < threshold)

            total_groups += num_groups
            sparse_groups += num_sparse

            sparsity_info[layer_name] = {
                "num_groups": num_groups,
                "num_sparse": num_sparse,
                "sparsity_ratio": num_sparse / num_groups if num_groups > 0 else 0,
                "group_norms": group_norms,
            }

    sparsity_info["overall"] = {
        "total_groups": total_groups,
        "sparse_groups": sparse_groups,
        "sparsity_ratio": sparse_groups / total_groups if total_groups > 0 else 0,
    }

    return sparsity_info


def print_sparsity_report(model, threshold=1e-3):
    """
    Print a formatted sparsity report for the model.

    Args:
        model: Keras model
        threshold: Threshold for considering a group as sparse
    """
    info = compute_sparsity_ratio(model, threshold)

    print("\n" + "=" * 70)
    print("SPARSITY REPORT")
    print("=" * 70)

    for layer_name, layer_info in info.items():
        if layer_name == "overall":
            continue
        print(f"\n{layer_name}:")
        print(f"  Total groups: {layer_info['num_groups']}")
        print(f"  Sparse groups: {layer_info['num_sparse']}")
        print(f"  Sparsity ratio: {layer_info['sparsity_ratio']:.2%}")

    print("\n" + "-" * 70)
    print(f"OVERALL SPARSITY: {info['overall']['sparsity_ratio']:.2%}")
    print(
        f"  ({info['overall']['sparse_groups']}/{info['overall']['total_groups']} groups)"
    )
    print("=" * 70 + "\n")


def load_and_preprocess_cifar10():
    """
    Load and preprocess CIFAR-10 dataset.

    Returns:
        (x_train, y_train), (x_test, y_test)
    """
    # Load data
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # Normalize pixel values to [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Convert labels to integers
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    print(f"Training samples: {len(x_train)}")
    print(f"Test samples: {len(x_test)}")
    print(f"Image shape: {x_train.shape[1:]}")

    return (x_train, y_train), (x_test, y_test)


def create_train_dataset(x_train, y_train, batch_size=128):
    """Build a tf.data pipeline with light augmentation and prefetch."""
    augmenter = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomTranslation(0.1, 0.1),
            layers.RandomRotation(0.04),
        ]
    )

    ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    ds = ds.shuffle(buffer_size=20000, seed=42, reshuffle_each_iteration=True)
    ds = ds.map(
        lambda img, label: (augmenter(img, training=True), label),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def create_eval_dataset(x, y, batch_size=128):
    """Evaluation pipeline without augmentation."""
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# Training Functions
def train_model(
    model,
    x_train,
    y_train,
    x_test,
    y_test,
    epochs=100,
    batch_size=128,
    model_name="model",
):
    """
    Train a model with callbacks and data augmentation.

    Args:
        model: Keras model to train
        x_train, y_train: Training data
        x_test, y_test: Test data
        epochs: Number of epochs
        batch_size: Batch size
        model_name: Name for saving checkpoints

    Returns:
        Training history
    """

    # Compile model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    # Build datasets
    train_ds = create_train_dataset(x_train, y_train, batch_size)
    val_ds = create_eval_dataset(x_test, y_test, batch_size)

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            f"{checkpoint_dir}/{model_name}_best.keras",
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
        ),
    ]

    # Train
    print(f"\nTraining {model_name}...")
    print("=" * 70)

    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1,
    )

    return history


def train_model_with_tensorboard(
    model,
    x_train,
    y_train,
    x_test,
    y_test,
    epochs=100,
    batch_size=128,
    model_name="model",
):
    """
    Train a model with callbacks including TensorBoard.

    Args:
        model: Keras model to train
        x_train, y_train: Training data
        x_test, y_test: Test data
        epochs: Number of epochs
        batch_size: Batch size
        model_name: Name for saving checkpoints and logs

    Returns:
        Training history
    """
    # Compile model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    # Build datasets
    train_ds = create_train_dataset(x_train, y_train, batch_size)
    val_ds = create_eval_dataset(x_test, y_test, batch_size)

    # Callbacks
    tb_callback = create_tensorboard_callback(model_name, logs_dir)
    weight_logger = WeightHistogramLogger(tb_callback.log_dir)
    # image_logger = SampleImageLogger(tb_callback.log_dir, x_train[:8])
    callbacks = [
        ModelCheckpoint(
            f"{checkpoint_dir}/{model_name}_best.keras",
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
        ),
        tb_callback,
        weight_logger,
        # image_logger,
    ]

    # Train
    print(f"\nTraining {model_name}...")
    print("=" * 70)

    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1,
    )

    return history


# Evaluation Funcitons
def evaluate_model(model, x_test, y_test, model_name="model"):
    """
    Evaluate model and print results.

    Args:
        model: Keras model
        x_test, y_test: Test data
        model_name: Model name for display

    Returns:
        Dictionary with evaluation metrics
    """
    print(f"\nEvaluating {model_name}...")
    print("=" * 70)

    # Evaluate
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Compute sparsity
    sparsity_info = compute_sparsity_ratio(model)
    print_sparsity_report(model)

    # Count parameters
    total_params = model.count_params()
    print(f"Total Parameters: {total_params:,}")

    results = {
        "model_name": model_name,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "total_params": int(total_params),
        "overall_sparsity": float(sparsity_info["overall"]["sparsity_ratio"]),
    }

    return results


def plot_training_history(
    histories, model_names, save_path=os.path.join(results_dir, "training_curves.png")
):
    """
    Plot training histories for multiple models.

    Args:
        histories: List of training histories
        model_names: List of model names
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot accuracy
    for history, name in zip(histories, model_names):
        axes[0].plot(
            history.history["accuracy"],
            label=f"{name} (train)",
            linestyle="--",
            alpha=0.7,
        )
        axes[0].plot(
            history.history["val_accuracy"], label=f"{name} (val)", linewidth=2
        )

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Model Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot loss
    for history, name in zip(histories, model_names):
        axes[1].plot(
            history.history["loss"], label=f"{name} (train)", linestyle="--", alpha=0.7
        )
        axes[1].plot(history.history["val_loss"], label=f"{name} (val)", linewidth=2)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Model Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Training curves saved to {save_path}")
    plt.close()


def plot_weight_distributions(
    models, model_names, save_path=os.path.join(results_dir, "weight_distributions.png")
):
    """
    Plot weight norm distributions for multiple models.

    Args:
        models: List of Keras models
        model_names: List of model names
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(len(models), 1, figsize=(12, 4 * len(models)))
    if len(models) == 1:
        axes = [axes]

    for idx, (model, name) in enumerate(zip(models, model_names)):
        sparsity_info = compute_sparsity_ratio(model)

        # Collect all group norms
        all_norms = []
        for layer_name, layer_info in sparsity_info.items():
            if layer_name != "overall" and "group_norms" in layer_info:
                all_norms.extend(layer_info["group_norms"])

        # Plot histogram
        axes[idx].hist(all_norms, bins=50, alpha=0.7, color="blue", edgecolor="black")
        axes[idx].set_xlabel("Group L2 Norm")
        axes[idx].set_ylabel("Count")
        axes[idx].set_title(
            f"{name} - Weight Group Norms Distribution\n"
            f'Sparsity: {sparsity_info["overall"]["sparsity_ratio"]:.2%}'
        )
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_yscale("log")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Weight distributions saved to {save_path}")
    plt.close()


# TensorBoard CallBack
def create_tensorboard_callback(model_name, log_dir=logs_dir):
    """
    Create TensorBoard callback with custom log directory.

    Args:
        model_name: Name of the model for logging
        log_dir: Base directory for TensorBoard logs

    Returns:
        TensorBoard callback
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(log_dir, f"{model_name}_{timestamp}")

    tensorboard_callback = TensorBoard(
        log_dir=log_path,
        histogram_freq=1,  # Log weight histograms every epoch
        write_graph=True,
        write_images=True,
        update_freq="epoch",
        profile_batch="10,20",  # Profile batches 10-20
        embeddings_freq=0,  # Disable embeddings logging (no embedding layers)
    )

    print(f"TensorBoard logs will be saved to: {log_path}")
    return tensorboard_callback


class WeightHistogramLogger(tf.keras.callbacks.Callback):
    """Log per-layer weight histograms with explicit tags."""

    def __init__(self, log_dir):
        super().__init__()
        self.writer = tf.summary.create_file_writer(
            os.path.join(log_dir, "weights_manual")
        )

    def on_epoch_end(self, epoch, logs=None):
        with self.writer.as_default():
            for layer in self.model.layers:
                for weight in layer.weights:
                    tag = f"{layer.name}/{weight.name.replace(':', '_')}"
                    tf.summary.histogram(tag, weight, step=epoch)
            self.writer.flush()


class SampleImageLogger(tf.keras.callbacks.Callback):
    """Log a fixed batch of training images for TensorBoard images tab."""

    def __init__(self, log_dir, sample_images):
        super().__init__()
        self.images = tf.convert_to_tensor(sample_images, dtype=tf.float32)
        self.writer = tf.summary.create_file_writer(os.path.join(log_dir, "images_manual"))

    def on_epoch_end(self, epoch, logs=None):
        with self.writer.as_default():
            tf.summary.image(
                "train_samples",
                self.images,
                step=epoch,
                max_outputs=tf.shape(self.images)[0],
            )
            self.writer.flush()


# Visualization Functions
def plot_comprehensive_metrics(histories, model_names, save_dir=results_dir):
    """
    Create comprehensive visualization of training metrics.

    Args:
        histories: List of training histories
        model_names: List of model names
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)

    # 1. Training curves (accuracy and loss)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Training accuracy
    for history, name in zip(histories, model_names):
        axes[0, 0].plot(history.history["accuracy"], label=name, linewidth=2)
    axes[0, 0].set_xlabel("Epoch", fontsize=12)
    axes[0, 0].set_ylabel("Training Accuracy", fontsize=12)
    axes[0, 0].set_title("Training Accuracy Comparison", fontsize=14, fontweight="bold")
    axes[0, 0].legend(loc="best", fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)

    # Validation accuracy
    for history, name in zip(histories, model_names):
        axes[0, 1].plot(history.history["val_accuracy"], label=name, linewidth=2)
    axes[0, 1].set_xlabel("Epoch", fontsize=12)
    axes[0, 1].set_ylabel("Validation Accuracy", fontsize=12)
    axes[0, 1].set_title(
        "Validation Accuracy Comparison", fontsize=14, fontweight="bold"
    )
    axes[0, 1].legend(loc="best", fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)

    # Training loss
    for history, name in zip(histories, model_names):
        axes[1, 0].plot(history.history["loss"], label=name, linewidth=2)
    axes[1, 0].set_xlabel("Epoch", fontsize=12)
    axes[1, 0].set_ylabel("Training Loss", fontsize=12)
    axes[1, 0].set_title("Training Loss Comparison", fontsize=14, fontweight="bold")
    axes[1, 0].legend(loc="best", fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)

    # Validation loss
    for history, name in zip(histories, model_names):
        axes[1, 1].plot(history.history["val_loss"], label=name, linewidth=2)
    axes[1, 1].set_xlabel("Epoch", fontsize=12)
    axes[1, 1].set_ylabel("Validation Loss", fontsize=12)
    axes[1, 1].set_title("Validation Loss Comparison", fontsize=14, fontweight="bold")
    axes[1, 1].legend(loc="best", fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "comprehensive_metrics.png"),
        dpi=300,
        bbox_inches="tight",
    )
    print(f"Comprehensive metrics saved to {save_dir}/comprehensive_metrics.png")
    plt.close()


def plot_sparsity_comparison(models, model_names, save_dir=results_dir):
    """
    Visualize sparsity patterns across different models.

    Args:
        models: List of trained models
        model_names: List of model names
        save_dir: Directory to save plots
    """
    sparsity_data = []

    for model, name in zip(models, model_names):
        sparsity_info = compute_sparsity_ratio(model)
        for layer_name, layer_info in sparsity_info.items():
            if layer_name != "overall":
                sparsity_data.append(
                    {
                        "Model": name,
                        "Layer": layer_name,
                        "Sparsity": layer_info["sparsity_ratio"],
                    }
                )

    df = pd.DataFrame(sparsity_data)
    if df.empty:
        print("No sparsity data to plot.")
        return

    # Align layers across models via pivot to avoid shape mismatches
    pivot = df.pivot(index="Layer", columns="Model", values="Sparsity").fillna(0)

    ax = pivot.plot(kind="bar", figsize=(16, 8))
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Sparsity Ratio", fontsize=12)
    ax.set_title("Layer-wise Sparsity Comparison", fontsize=14, fontweight="bold")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    plt.savefig(
        os.path.join(save_dir, "sparsity_comparison.png"), dpi=300, bbox_inches="tight"
    )
    print(f"Sparsity comparison saved to {save_dir}/sparsity_comparison.png")
    plt.close()


def plot_weight_distribution_grid(models, model_names, save_dir=results_dir):
    """
    Create a grid of weight distribution plots for all models.

    Args:
        models: List of trained models
        model_names: List of model names
        save_dir: Directory to save plots
    """
    n_models = len(models)
    fig, axes = plt.subplots(n_models, 2, figsize=(14, 4 * n_models))

    if n_models == 1:
        axes = axes.reshape(1, -1)

    for idx, (model, name) in enumerate(zip(models, model_names)):
        sparsity_info = compute_sparsity_ratio(model)

        # Collect all group norms
        all_norms = []
        conv_norms = []
        dense_norms = []

        for layer_name, layer_info in sparsity_info.items():
            if layer_name != "overall" and "group_norms" in layer_info:
                norms = layer_info["group_norms"]
                all_norms.extend(norms)
                if "conv" in layer_name.lower():
                    conv_norms.extend(norms)
                elif "dense" in layer_name.lower():
                    dense_norms.extend(norms)

        # Plot all weights histogram
        axes[idx, 0].hist(
            all_norms, bins=50, alpha=0.7, color="steelblue", edgecolor="black"
        )
        axes[idx, 0].axvline(
            1e-3, color="red", linestyle="--", linewidth=2, label="Sparsity Threshold"
        )
        axes[idx, 0].set_xlabel("Group L2 Norm", fontsize=11)
        axes[idx, 0].set_ylabel("Count (log scale)", fontsize=11)
        axes[idx, 0].set_title(
            f"{name} - All Layers Weight Distribution\n"
            f'Overall Sparsity: {sparsity_info["overall"]["sparsity_ratio"]:.2%}',
            fontsize=12,
            fontweight="bold",
        )
        axes[idx, 0].set_yscale("log")
        axes[idx, 0].legend()
        axes[idx, 0].grid(True, alpha=0.3)

        # Plot separated conv and dense
        if conv_norms:
            axes[idx, 1].hist(
                conv_norms,
                bins=40,
                alpha=0.6,
                color="green",
                edgecolor="black",
                label="Conv Layers",
            )
        if dense_norms:
            axes[idx, 1].hist(
                dense_norms,
                bins=20,
                alpha=0.6,
                color="orange",
                edgecolor="black",
                label="Dense Layers",
            )
        axes[idx, 1].axvline(
            1e-3, color="red", linestyle="--", linewidth=2, label="Threshold"
        )
        axes[idx, 1].set_xlabel("Group L2 Norm", fontsize=11)
        axes[idx, 1].set_ylabel("Count (log scale)", fontsize=11)
        axes[idx, 1].set_title(
            f"{name} - Conv vs Dense Layers", fontsize=12, fontweight="bold"
        )
        axes[idx, 1].set_yscale("log")
        axes[idx, 1].legend()
        axes[idx, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "weight_distributions_grid.png"),
        dpi=300,
        bbox_inches="tight",
    )
    print(f"Weight distribution grid saved to {save_dir}/weight_distributions_grid.png")
    plt.close()


def plot_confusion_matrices(models, model_names, x_test, y_test, save_dir=results_dir):
    """
    Generate confusion matrices for all models.

    Args:
        models: List of trained models
        model_names: List of model names
        x_test: Test images
        y_test: Test labels
        save_dir: Directory to save plots
    """
    cifar10_classes = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    n_models = len(models)
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    axes = axes.flatten()

    for idx, (model, name) in enumerate(zip(models, model_names)):
        # Get predictions
        predictions = model.predict(x_test, verbose=0)
        y_pred = np.argmax(predictions, axis=1)

        # Compute confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        # Plot
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=cifar10_classes,
            yticklabels=cifar10_classes,
            ax=axes[idx],
            cbar_kws={"label": "Normalized Count"},
        )
        axes[idx].set_xlabel("Predicted Label", fontsize=11)
        axes[idx].set_ylabel("True Label", fontsize=11)
        axes[idx].set_title(f"{name} Confusion Matrix", fontsize=12, fontweight="bold")
        axes[idx].tick_params(axis="both", labelsize=9)

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "confusion_matrices.png"), dpi=300, bbox_inches="tight"
    )
    print(f"Confusion matrices saved to {save_dir}/confusion_matrices.png")
    plt.close()


def plot_performance_summary(results, save_dir=results_dir):
    """
    Create a summary visualization of model performance metrics.

    Args:
        results: List of result dictionaries
        save_dir: Directory to save plots
    """
    df = pd.DataFrame(results)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Accuracy comparison
    axes[0].bar(df["model_name"], df["test_accuracy"], color="steelblue", alpha=0.8)
    axes[0].set_ylabel("Test Accuracy", fontsize=12)
    axes[0].set_title("Test Accuracy Comparison", fontsize=14, fontweight="bold")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].grid(True, alpha=0.3, axis="y")
    for i, v in enumerate(df["test_accuracy"]):
        axes[0].text(i, v + 0.01, f"{v:.4f}", ha="center", va="bottom", fontsize=10)

    # Sparsity comparison
    axes[1].bar(df["model_name"], df["overall_sparsity"], color="green", alpha=0.8)
    axes[1].set_ylabel("Sparsity Ratio", fontsize=12)
    axes[1].set_title("Overall Sparsity Comparison", fontsize=14, fontweight="bold")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].grid(True, alpha=0.3, axis="y")
    for i, v in enumerate(df["overall_sparsity"]):
        axes[1].text(i, v + 0.01, f"{v:.2%}", ha="center", va="bottom", fontsize=10)

    # Accuracy vs Sparsity scatter
    axes[2].scatter(
        df["overall_sparsity"], df["test_accuracy"], s=200, alpha=0.6, c=range(len(df))
    )
    for i, name in enumerate(df["model_name"]):
        axes[2].annotate(
            name,
            (df["overall_sparsity"][i], df["test_accuracy"][i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
        )
    axes[2].set_xlabel("Sparsity Ratio", fontsize=12)
    axes[2].set_ylabel("Test Accuracy", fontsize=12)
    axes[2].set_title("Accuracy vs Sparsity Trade-off", fontsize=14, fontweight="bold")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "performance_summary.png"), dpi=300, bbox_inches="tight"
    )
    print(f"Performance summary saved to {save_dir}/performance_summary.png")
    plt.close()


def generate_classification_reports(
    models, model_names, x_test, y_test, save_dir=results_dir
):
    """
    Generate detailed classification reports for all models.

    Args:
        models: List of trained models
        model_names: List of model names
        x_test: Test images
        y_test: Test labels
        save_dir: Directory to save reports
    """
    cifar10_classes = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    reports = {}

    for model, name in zip(models, model_names):
        predictions = model.predict(x_test, verbose=0)
        y_pred = np.argmax(predictions, axis=1)

        report = classification_report(
            y_test, y_pred, target_names=cifar10_classes, output_dict=True
        )
        reports[name] = report

        # Save text report
        report_text = classification_report(
            y_test, y_pred, target_names=cifar10_classes
        )
        with open(
            os.path.join(save_dir, f"{name}_classification_report.txt"), "w"
        ) as f:
            f.write(f"Classification Report for {name}\n")
            f.write("=" * 70 + "\n")
            f.write(report_text)

    print(f"Classification reports saved to {save_dir}/")
    return reports


print("=" * 70)
print("Group Sparsity Regularizer - CIFAR-10 Experiment")
print("=" * 70)


# Load CIFAR-10 dataset
print("\nLoading CIFAR-10 dataset...")
(x_train, y_train), (x_test, y_test) = load_and_preprocess_cifar10()


# Training parameters
EPOCHS = 100
BATCH_SIZE = 512 # Increased batch size for faster training


# Build models with different regularizations
print("\nBuilding models...")

# 1. Baseline (no regularization)
print("  - Baseline model (no regularization)")
model_baseline = build_resnet18(conv_regularizer=None, dense_regularizer=None)

# 2. L1 regularizer
print("  - L1 regularized model")
l1_reg = keras.regularizers.l1(0.0001)
model_l1 = build_resnet18(conv_regularizer=l1_reg, dense_regularizer=l1_reg)

# 3. L2 regularization
print("  - L2 regularized model")
l2_reg = keras.regularizers.l2(0.0001)
model_l2 = build_resnet18(conv_regularizer=l2_reg, dense_regularizer=l2_reg)


# 4. Group L1 regularization
print("  - Group L1 regularized model")
conv_group_regularizer = GroupL1Regularizer(l1=0.001, axis=[0, 1, 2])
dense_group_regularizer = GroupL1Regularizer(l1=0.001, axis=0)
model_group = build_resnet18(
    conv_regularizer=conv_group_regularizer, dense_regularizer=dense_group_regularizer
)


# Model Architecture
print("\nModel Architecture:")
model_baseline.summary()


# Train models
histories = []

print("\n" + "=" * 70)
print("TRAINING PHASE")
print("=" * 70)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices("GPU")))
print("=" * 70)


# Baseline
history_baseline = train_model_with_tensorboard(
    model_baseline,
    x_train,
    y_train,
    x_test,
    y_test,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    model_name="baseline",
)
histories.append(history_baseline)


# L1 regularizer
history_l1 = train_model_with_tensorboard(
    model_l1,
    x_train,
    y_train,
    x_test,
    y_test,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    model_name="l1_reg",
)
histories.append(history_l1)


# L2 Regularizer
history_l2 = train_model_with_tensorboard(
    model_l2,
    x_train,
    y_train,
    x_test,
    y_test,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    model_name="l2_reg",
)
histories.append(history_l2)


# Group L1 Regularizer
history_group = train_model_with_tensorboard(
    model_group,
    x_train,
    y_train,
    x_test,
    y_test,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    model_name="group_l1",
)
histories.append(history_group)

# Save training histories
with open(os.path.join(results_dir, "all_histories.json"), "w") as f:
    json.dump(histories, f, indent=2)

# Evaluate models
print("\n" + "=" * 70)
print("EVALUATION PHASE")
print("=" * 70)

results = []
results.append(evaluate_model(model_baseline, x_test, y_test, "Baseline"))
results.append(evaluate_model(model_l1, x_test, y_test, "L1 Regularized"))
results.append(evaluate_model(model_l2, x_test, y_test, "L2 Regularized"))
results.append(evaluate_model(model_group, x_test, y_test, "Group L1 Regularized"))


# Save results
with open(os.path.join(results_dir, "results.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {os.path.join(results_dir, 'results.json')}\n\n")
# Generate visualizations
print("\nGenerating visualizations...")
plot_training_history(
    histories,
    ["Baseline", "L1 Reg", "L2 Reg", "Group L1"],
    save_path=os.path.join(results_dir, "training_curves.png"),
)

plot_weight_distributions(
    [model_baseline, model_l1, model_l2, model_group],
    ["Baseline", "L1 Reg", "L2 Regularized", "Group L1 Regularized"],
    save_path=os.path.join(results_dir, "weight_distributions.png"),
)


# Summary table
print("\n" + "=" * 70)
print("FINAL RESULTS SUMMARY")
print("=" * 70)
print(f"{'Model':<25} {'Accuracy':<12} {'Sparsity':<12} {'Params'}")
print("-" * 70)
for result in results:
    print(
        f"{result['model_name']:<25} "
        f"{result['test_accuracy']:<12.4f} "
        f"{result['overall_sparsity']:<12.2%} "
        f"{result['total_params']:,}"
    )
print("=" * 70)


# Comprehensive Visualizations
print("\n" + "=" * 70)
print("GENERATING COMPREHENSIVE VISUALIZATIONS")
print("=" * 70)

model_list = [model_baseline, model_l1, model_l2, model_group]
model_names_list = [
    "Baseline",
    "L1 Regularized",
    "L2 Regularized",
    "Group L1 Regularized",
]

## 1. Comprehensive training metrics
print("\n1. Creating comprehensive training metrics plot...")
plot_comprehensive_metrics(histories, model_names_list)

## 2. Sparsity comparison across layers
print("\n2. Creating layer-wise sparsity comparison...")
plot_sparsity_comparison(model_list, model_names_list)

## 3. Weight distribution grid
print("\n3. Creating weight distribution grid...")
plot_weight_distribution_grid(model_list, model_names_list)

## 4. Confusion matrices
print("\n4. Generating confusion matrices...")
plot_confusion_matrices(model_list, model_names_list, x_test, y_test)

## 5. Performance summary
print("\n5. Creating performance summary...")
plot_performance_summary(results)

## 6. Classification reports
classification_reports = generate_classification_reports(
    model_list, model_names_list, x_test, y_test
)


print("\n" + "=" * 70)
print("ALL VISUALIZATIONS COMPLETED")
print("=" * 70)

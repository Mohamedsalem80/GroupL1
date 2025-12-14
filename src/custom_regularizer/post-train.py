
import os
import sys
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix

# Allow running as a script (no package context)
ROOT = os.getcwd()
sys.path.append(os.path.join(ROOT, "src"))
try:
    from .GroupL1Regularizer import GroupL1Regularizer
    from .ResNet18 import build_resnet18
except ImportError:
    from custom_regularizer.GroupL1Regularizer import GroupL1Regularizer
    from custom_regularizer.ResNet18 import build_resnet18

checkpoint_dir = os.path.join(ROOT, "checkpoints")
results_dir = os.path.join(ROOT, "results")
os.makedirs(results_dir, exist_ok=True)

# Threshold for considering a group effectively zeroed-out
SPARSITY_THRESHOLD = 1e-2


def compute_sparsity_ratio(model, threshold=SPARSITY_THRESHOLD):
    sparsity_info = {}
    total_groups = 0
    sparse_groups = 0
    for layer in model.layers:
        if hasattr(layer, "kernel"):
            weights = layer.kernel.numpy()
            axis = (0, 1, 2) if "conv" in layer.name.lower() else 0
            group_norms = np.sqrt(np.sum(np.square(weights), axis=axis))
            num_groups = len(group_norms)
            num_sparse = np.sum(group_norms < threshold)
            total_groups += num_groups
            sparse_groups += num_sparse
            sparsity_info[layer.name] = {
                "num_groups": num_groups,
                "num_sparse": num_sparse,
                "sparsity_ratio": num_sparse / num_groups if num_groups else 0,
                "group_norms": group_norms,
            }
    sparsity_info["overall"] = {
        "total_groups": total_groups,
        "sparse_groups": sparse_groups,
        "sparsity_ratio": sparse_groups / total_groups if total_groups else 0,
    }
    return sparsity_info


def plot_sparsity_comparison(
    models, model_names, save_dir=results_dir, threshold=SPARSITY_THRESHOLD
):
    """
    Visualize sparsity patterns across different models.

    Args:
        models: List of trained models
        model_names: List of model names
        save_dir: Directory to save plots
    """
    sparsity_data = []

    for model, name in zip(models, model_names):
        sparsity_info = compute_sparsity_ratio(model, threshold=threshold)
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
            SPARSITY_THRESHOLD,
            color="red",
            linestyle="--",
            linewidth=2,
            label="Sparsity Threshold",
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
            SPARSITY_THRESHOLD,
            color="red",
            linestyle="--",
            linewidth=2,
            label="Threshold",
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


def load_and_preprocess_cifar10():
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    return (x_train, y_train), (x_test, y_test)


def build_models():
    models = {}
    models["baseline"] = build_resnet18(conv_regularizer=None, dense_regularizer=None)
    l1_reg = keras.regularizers.l1(0.0001)
    models["l1_reg"] = build_resnet18(conv_regularizer=l1_reg, dense_regularizer=l1_reg)
    l2_reg = keras.regularizers.l2(0.0001)
    models["l2_reg"] = build_resnet18(conv_regularizer=l2_reg, dense_regularizer=l2_reg)
    conv_group_regularizer = GroupL1Regularizer(l1=0.001, axis=[0, 1, 2])
    dense_group_regularizer = GroupL1Regularizer(l1=0.001, axis=0)
    models["group_l1"] = build_resnet18(
        conv_regularizer=conv_group_regularizer,
        dense_regularizer=dense_group_regularizer,
    )
    return models


def load_weights_if_available(models_dict, ckpt_dir=checkpoint_dir):
    loaded = {}
    for name, model in models_dict.items():
        path = os.path.join(ckpt_dir, f"{name}_best.keras")
        if os.path.exists(path):
            model.load_weights(path)
            loaded[name] = model
            print(f"Loaded weights for {name} from {path}")
        else:
            print(f"Checkpoint not found for {name}: {path}")
    return loaded


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


def compile_model(model):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


def main():
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_cifar10()

    models_map = build_models()
    models_map = load_weights_if_available(models_map)
    if not models_map:
        print("No checkpoints loaded; aborting.")
        return

    # Compile for evaluation
    for m in models_map.values():
        compile_model(m)

    model_names = list(models_map.keys())
    models_list = [models_map[k] for k in model_names]

    # Evaluate
    results = []
    for name, model in zip(model_names, models_list):
        print(f"Evaluating {name}...")
        print_sparsity_report(model)
        test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
        sparsity_info = compute_sparsity_ratio(model)
        total_params = model.count_params()
        results.append(
            {
                "model_name": name,
                "test_loss": float(test_loss),
                "test_accuracy": float(test_accuracy),
                "total_params": int(total_params),
                "overall_sparsity": float(sparsity_info["overall"]["sparsity_ratio"]),
            }
        )

    with open(os.path.join(results_dir, "post_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved evaluation to {os.path.join(results_dir, 'post_results.json')}")

    # Plots (no training curves because histories are unavailable post-hoc)
    plot_weight_distribution_grid(models_list, model_names)
    plot_sparsity_comparison(models_list, model_names)
    plot_confusion_matrices(models_list, model_names, x_test, y_test)
    plot_performance_summary(results)
    generate_classification_reports(models_list, model_names, x_test, y_test)


if __name__ == "__main__":
    main()
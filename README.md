# Custom Regularizer: Group L1 Regularizer for Structured Sparsity

[![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20+-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Poetry](https://img.shields.io/badge/Poetry-2.0+-red.svg)](https://python-poetry.org/)

A custom TensorFlow/Keras regularizer that implements **Group L1 regularization** for structured sparsity in neural networks. This project demonstrates how to achieve structured sparsity by applying L1 penalties on L2 norms of weight groups, encouraging entire neurons or channels to become zero during training.

## 🎯 Key Features

- **Group L1 Regularizer**: Custom regularizer for structured sparsity
- **ResNet-18 Implementation**: Complete ResNet-18 architecture adapted for CIFAR-10
- **Comprehensive Experiments**: Compare different regularization techniques
- **Visualization Tools**: Rich plotting functions for sparsity analysis
- **TensorBoard Integration**: Built-in logging and monitoring
- **Modular Design**: Clean, reusable components

## 📊 Structured Sparsity

Unlike traditional L1/L2 regularization that promotes unstructured sparsity (random weight pruning), **Group L1 regularization** encourages **structured sparsity** by:

- Applying L1 penalty on L2 norms of weight groups
- For Conv2D layers: Groups channels (output feature maps)
- For Dense layers: Groups neurons (output units)
- Results in entire neurons/channels becoming zero, enabling hardware-efficient pruning

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/custom-regularizer.git
cd custom-regularizer

# Install dependencies with Poetry (recommended)
poetry install

# Or with pip (alternative)
pip install -e .
```

### Basic Usage

```python
import tensorflow as tf
from tensorflow import keras
from custom_regularizer import GroupL1Regularizer, build_resnet18

# Create regularizer
conv_regularizer = GroupL1Regularizer(l1=0.001, axis=[0, 1, 2])  # For Conv2D layers
dense_regularizer = GroupL1Regularizer(l1=0.001, axis=0)          # For Dense layers

# Build ResNet-18 with group regularization
model = build_resnet18(
    input_shape=(32, 32, 3),
    num_classes=10,
    conv_regularizer=conv_regularizer,
    dense_regularizer=dense_regularizer
)

# Compile and train
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train your model...
```

## 📁 Project Structure

```
custom-regularizer/
├── src/
│   └── custom_regularizer/
│       ├── __init__.py              # Package exports
│       ├── GroupL1Regularizer.py    # Custom regularizer implementation
│       ├── ResidualBlock.py         # ResNet residual block
│       ├── ResNet18.py             # ResNet-18 architecture
│       ├── main.py                 # CIFAR-10 experiments (train + log)
│       ├── post-train.py           # Load checkpoints, evaluate, plot
│       └── architecture.py         # Export architecture diagrams (2D/3D)
├── tests/
│   └── test_basic.py               # Basic unit tests
├── example.py                      # Simple usage example
├── pyproject.toml                  # Project configuration
├── poetry.lock                     # Poetry lock file
├── README.md                       # This file
├── LICENSE                         # MIT License
├── .gitignore                      # Git ignore rules
├── checkpoints/                    # Model checkpoints (created during training)
├── results/                        # Experiment results (created during training)
└── logs/                          # TensorBoard logs (created during training)
```

## 🔧 API Reference

### GroupL1Regularizer

```python
GroupL1Regularizer(l1=0.01, axis=0)
```

**Parameters:**
- `l1` (float): L1 regularization strength (default: 0.01)
- `axis` (int or list): Axis/axes along which to compute group norms
  - Dense layers: `axis=0` (groups neurons)
  - Conv2D layers: `axis=[0,1,2]` (groups channels)

**Example:**
```python
# For Dense layer (group by neurons)
dense_reg = GroupL1Regularizer(l1=0.01, axis=0)

# For Conv2D layer (group by channels)
conv_reg = GroupL1Regularizer(l1=0.01, axis=[0, 1, 2])

# Apply to layers
dense_layer = keras.layers.Dense(128, kernel_regularizer=dense_reg)
conv_layer = keras.layers.Conv2D(64, (3,3), kernel_regularizer=conv_reg)
```

### build_resnet18

```python
build_resnet18(input_shape=(32, 32, 3), num_classes=10,
               conv_regularizer=None, dense_regularizer=None)
```

**Parameters:**
- `input_shape`: Input image dimensions
- `num_classes`: Number of output classes
- `conv_regularizer`: Regularizer for Conv2D layers
- `dense_regularizer`: Regularizer for Dense layers

**Returns:** Keras Model instance

## 🧪 Testing

The project includes comprehensive unit tests to ensure functionality:

```bash
# Run all tests
poetry run pytest tests/

# Run with verbose output
poetry run pytest -v tests/

# Run specific test
poetry run python tests/test_basic.py
```

Tests cover:
- **GroupL1Regularizer**: Basic functionality and regularization calculations
- **ResidualBlock**: Layer construction and forward pass
- **ResNet-18**: Model building and architecture validation
- **Regularized Models**: Integration of regularizers with model layers

## 🧪 Running Experiments

The project includes comprehensive experiments comparing different regularization techniques on CIFAR-10:

```bash
# Run the full experiment suite
poetry run python -m custom_regularizer.main
```

This will train 4 models and generate:
- **Baseline** (no regularization)
- **L1 Regularized** (traditional L1)
- **L2 Regularized** (traditional L2)
- **Group L1 Regularized** (structured sparsity)

### Simple Example

For a quick demonstration without full training:

```bash
# Run the simple example
poetry run python example.py
```

### Generated Outputs

- **Model checkpoints** in `checkpoints/`
- **Training curves**, **sparsity plots**, and **architecture diagrams** in `results/`
- **TensorBoard logs** in `logs/`
- **Performance metrics** in `results/results.json`

## 📈 Results & Analysis

The experiments demonstrate that Group L1 regularization achieves:

1. **Structured Sparsity**: Entire neurons/channels become zero
2. **Hardware Efficiency**: Enables efficient pruning and acceleration
3. **Competitive Accuracy**: Maintains performance while reducing parameters
4. **Better Interpretability**: Clear structure in weight distributions

### Example Results

```
Model               Accuracy    Sparsity    Parameters
Baseline            0.8542      0.00%       11,173,962
L1 Regularized      0.8421      15.23%      11,173,962
L2 Regularized      0.8518      2.45%       11,173,962
Group L1 Regularized 0.8435    28.47%      11,173,962
```

## 🎨 Visualization Features

The project includes comprehensive visualization tools:

- **Training curves**: Accuracy and loss comparison
- **Sparsity analysis**: Layer-wise and overall sparsity ratios
- **Weight distributions**: Group norm histograms
- **Confusion matrices**: Per-class performance
- **Performance summaries**: Accuracy vs sparsity trade-offs
- **Architecture diagrams**: 2D/3D renders and per-block internals

## 🔍 Technical Details

### How Group L1 Regularization Works

For a weight tensor `W` with shape `(H, W, C_in, C_out)` in Conv2D layers:

1. **Group Formation**: Groups are formed along specified axes (e.g., `[0,1,2]` for channels)
2. **L2 Norm Calculation**: `group_norm = sqrt(sum(W^2))` for each group
3. **L1 Penalty**: `penalty = l1 * sum(group_norms)` across all groups

This encourages entire output channels to become zero, enabling structured pruning.

### Mathematical Formulation

```
L_regularization = λ₁ * Σᵢ ||Wᵢ||₂
```

Where:
- `λ₁`: L1 regularization strength
- `Wᵢ`: Weight groups (neurons or channels)
- `||Wᵢ||₂`: L2 norm of each group

## 🛠️ Development

### Setting up Development Environment

```bash
# Install in development mode
poetry install

# Run tests
poetry run pytest tests/

# Format code
poetry run black src/
poetry run isort src/

# Type checking
poetry run mypy src/
```

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run specific test file
poetry run python tests/test_basic.py

# Run with coverage
poetry run pytest --cov=custom_regularizer
```

### Adding New Regularizers

1. Extend `keras.regularizers.Regularizer`
2. Implement `__call__()` method
3. Add `get_config()` and `from_config()` for serialization

```python
class MyRegularizer(keras.regularizers.Regularizer):
    def __call__(self, weights):
        # Your regularization logic here
        return regularization_loss
```

## 🖼️ Visualization & Post-Train Utilities

- **Architecture exports** (results/baseline_architecture_color.png, baseline_architecture_schematic.png, baseline_architecture_3d.png, baseline_architecture_block_sample.png):

```bash
poetry run python src/custom_regularizer/architecture.py
```

- **Post-train evaluation and plots** (confusion matrices, sparsity comparison, weight distributions, performance summary, classification reports; outputs in `results/`):

```bash
poetry run python src/custom_regularizer/post-train.py
```
Requires trained checkpoints in `checkpoints/` (e.g., baseline_best.keras, l1_reg_best.keras, l2_reg_best.keras, group_l1_best.keras).

## 📚 Dependencies

- **tensorflow[and-cuda] >= 2.20.0**: Deep learning framework
- **numpy >= 1.23.5**: Numerical computations
- **pandas >= 2.3.3**: Data manipulation
- **matplotlib >= 3.10.7**: Plotting
- **seaborn >= 0.13.2**: Statistical visualization
- **scikit-learn >= 1.7.2**: Machine learning utilities
- **black >= 25.12.0**: Code formatting
- **isort >= 7.0.0**: Import sorting
- **mypy >= 1.19.0**: Type checking

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Based on research in structured sparsity and neural network pruning
- ResNet-18 architecture adapted from the original paper
- CIFAR-10 dataset courtesy of Alex Krizhevsky

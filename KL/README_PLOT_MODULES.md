# WM Transform Plotting Modules

This directory contains modularized plotting functionality for WM (Watermarking) transformation analysis. The plotting functions have been separated from the core WM transformation logic for better maintainability.

## Module Overview

### Core Module
- **`wm_simple.py`**: Core WM transformation functions without plotting
  - `w(x, a)`: Exponential transformation
  - `m(x, a)`: Logarithmic transformation
  - `f(x, a)`: Rounding function
  - `create_normalized_vector()`: Generate normalized probability vectors
  - `analyze_transformation()`: Complete WM transformation analysis

### Plotting Modules

#### 1. `plot_kl_single.py` - Single Vector KL Plots
**Function**: `plot_kl_vs_a(X, a_range, filename)`

**Purpose**: Plot KL divergence vs parameter a for a single input vector

**Usage**:
```python
from plot_kl_single import plot_kl_vs_a, create_normalized_vector

X = create_normalized_vector(size=5, seed=42)
plot_kl_vs_a(X, filename="single_kl_plot.jpg")
```

#### 2. `plot_kl_comparison.py` - Multi-Vector KL Comparison
**Function**: `plot_multiple_kl_vs_a(X_list, labels, a_range, filename, kl_type)`

**Purpose**: Compare KL divergence curves for multiple vectors/distributions

**Usage**:
```python
from plot_kl_comparison import plot_multiple_kl_vs_a, create_normalized_vector

X1 = create_normalized_vector(size=5, distribution='uniform')
X2 = create_normalized_vector(size=5, distribution='exponential')
X_list = [X1, X2]
labels = ['Uniform', 'Exponential']

plot_multiple_kl_vs_a(X_list, labels, kl_type="X_Y", filename="comparison.jpg")
```

#### 3. `plot_optimal_a_distribution.py` - Optimal a Distribution
**Function**: `analyze_optimal_a_distribution(num_vectors, vector_size, a_range, distributions, filename)`

**Purpose**: Analyze the distribution of optimal a values across different random strategies

**Usage**:
```python
from plot_optimal_a_distribution import analyze_optimal_a_distribution

results = analyze_optimal_a_distribution(
    num_vectors=30,
    distributions=['uniform', 'exponential', 'beta', 'gamma', 'normal'],
    filename="optimal_a_dist.jpg"
)
```

#### 4. `plot_size_migration.py` - Size Migration Analysis
**Function**: `analyze_optimal_a_vs_size_multiple_distributions(sizes, num_vectors_per_size, a_range, filename)`

**Purpose**: Analyze how optimal a values change with vector size across different distributions

**Usage**:
```python
from plot_size_migration import analyze_optimal_a_vs_size_multiple_distributions

results = analyze_optimal_a_vs_size_multiple_distributions(
    sizes=[10, 50, 100, 200, 500],
    num_vectors_per_size=5,
    filename="size_migration.jpg"
)
```

## Key Features

### GPU Acceleration
All modules automatically detect and utilize GPU if available:
- Automatic device detection (`cuda` vs `cpu`)
- GPU memory management for tensor operations
- Optimized for large-scale computations

### Error Handling
- Robust NaN/inf detection and handling
- Graceful fallbacks for computation errors
- Informative error messages

### Visualization
- High-resolution plots (300 DPI)
- Automatic axis scaling and labeling
- Multiple color schemes for different distributions
- Clean, publication-ready output

## Dependencies

- PyTorch
- Matplotlib
- NumPy (via PyTorch)
- SciPy (optional, for density plots)

## Usage Examples

### Quick Start
```python
# Import core functions
from wm_simple import create_normalized_vector, analyze_transformation

# Create test vector
X = create_normalized_vector(size=10, distribution='uniform')

# Analyze transformation
results = analyze_transformation(X, a=3.5)
print(f"KL divergence: {results['KL_X_Y']:.4f}")
```

### Advanced Analysis
```python
# Multi-distribution comparison
from plot_kl_comparison import plot_multiple_kl_vs_a

vectors = []
labels = []
for dist in ['uniform', 'exponential', 'beta']:
    X = create_normalized_vector(size=20, distribution=dist, seed=42)
    vectors.append(X)
    labels.append(dist.upper())

plot_multiple_kl_vs_a(vectors, labels, filename="multi_dist_comparison.jpg")
```

## Output Files

All plotting functions save high-quality JPG files:
- Single plots: ~300KB
- Multi-distribution plots: ~400-500KB
- Migration analysis: ~300-400KB

## Performance Notes

- GPU acceleration provides significant speedup for large vectors
- Memory usage scales with vector size and number of computations
- Plotting functions are optimized for batch processing

## Troubleshooting

1. **CUDA not available**: Functions automatically fall back to CPU
2. **Memory errors**: Reduce vector size or num_vectors parameter
3. **Plot display issues**: Ensure matplotlib backend is properly configured
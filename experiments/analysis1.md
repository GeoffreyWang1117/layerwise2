# DeepFM Teacher Model Gradient Analysis

## Introduction

This document describes the gradient analysis methodology implemented in `analysis1.py`, which focuses on analyzing the layer-wise contributions of gradients and weights in a DeepFM teacher model trained on the Frappe dataset. The analysis helps understand which components of the model contribute most significantly to learning and prediction.

## Methodology

### Mathematical Principles

#### Gradient Analysis

The core methodology is based on measuring parameter importance through gradient analysis. For a neural network with parameters θ, the importance of a parameter θᵢ can be estimated using:

1. **Gradient Magnitude**: The absolute gradient value |∂L/∂θᵢ| indicates how sensitive the loss function L is to changes in the parameter.

2. **Parameter Norm**: The norm of the parameter ||θᵢ|| provides information about its scale.

3. **Parameter Importance Score**: Combining these two metrics gives us the importance score:

   $$\text{Importance}(\theta_i) = \left|\frac{\partial L}{\partial \theta_i}\right| \times \|\theta_i\|$$

   This is conceptually similar to the principles behind pruning techniques like magnitude-based and gradient-based pruning.

#### Layer-wise Contribution

For each layer l with parameters θᵢ ∈ l, the average importance across its parameters is calculated as:

$$\text{Layer\_Importance}(l) = \frac{1}{|l|} \sum_{\theta_i \in l} \text{Importance}(\theta_i)$$

where |l| is the number of parameters in layer l.

## Implementation Details

### Key Components

1. **Model Loading**: The code loads a pre-trained DeepFM model and the Frappe dataset.

2. **Gradient Tracking**: 
   - Processes small batches of samples through the model
   - Computes loss using binary cross-entropy
   - Performs backward propagation to calculate gradients
   - Records gradient norms for all parameters

3. **Parameter Importance Calculation**:
   - Calculates importance scores as gradient norm × parameter norm
   - Aggregates scores by layer type (embedding, DNN, FM, linear)

4. **Visualization**:
   - Plots gradient norm trends for DNN layers
   - Plots gradient norm trends for embedding layers
   - Highlights differences in gradient behavior across layers

### DeepFM Architecture

The DeepFM model combines:

1. **Factorization Machine (FM)** component: Models low-order feature interactions
   $$y_{FM} = w_0 + \sum_{i=1}^{n} w_i x_i + \sum_{i=1}^{n} \sum_{j=i+1}^{n} \langle \mathbf{v}_i, \mathbf{v}_j \rangle x_i x_j$$

   where w₀ is bias, wᵢ are weights, and $\langle \mathbf{v}_i, \mathbf{v}_j \rangle$ is the dot product of feature embeddings.

2. **Deep Neural Network (DNN)** component: Captures high-order feature interactions through multiple fully connected layers.
   $$y_{DNN} = \sigma(\mathbf{W}^{(L)} \cdot \sigma(\mathbf{W}^{(L-1)} \cdot ... \sigma(\mathbf{W}^{(1)} \cdot \mathbf{a}^{(0)} + \mathbf{b}^{(1)})... + \mathbf{b}^{(L-1)}) + \mathbf{b}^{(L)})$$

   where $\mathbf{a}^{(0)}$ is the concatenated feature embedding vector, $\mathbf{W}^{(l)}$ and $\mathbf{b}^{(l)}$ are weights and biases for layer $l$, and $\sigma$ is the activation function.

3. **Feature Embeddings**: Shared between FM and DNN components.

4. **Final Prediction**: Combines both components
   $$\hat{y} = \sigma(y_{FM} + y_{DNN})$$

   where $\sigma$ is the sigmoid function for binary classification.

## Results and Insights

The analysis reveals:

1. **Feature Importance Ranking**: Identifies which features in the Frappe dataset (user, item, context) contribute most to the model's predictions.

2. **Layer Contribution Distribution**: Shows whether embedding layers or DNN layers are more important for the model's accuracy.

3. **Gradient Behavior Patterns**: Visualizes how gradients evolve across different layers, indicating the learning dynamics.

4. **Optimization Insights**: Provides suggestions for model refinement, such as:
   - Which layers might benefit from focused attention
   - Which features could potentially be pruned in lighter models
   - Whether the learning rates should be adjusted for specific layers

## Applications

This analysis can be applied to:

1. **Model Compression**: Identify less important parameters that can be pruned.
2. **Knowledge Distillation**: Focus distillation on the most important aspects of the teacher model.
3. **Architecture Optimization**: Redesign model architecture based on contribution patterns.
4. **Training Improvement**: Adjust learning strategies for different components based on gradient behavior.
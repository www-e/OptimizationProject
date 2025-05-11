# Experiment Types and Parameters Documentation

This document provides detailed information about the different experiment types available in the Neural Network Optimization project and explains the purpose and functionality of each parameter in the Weight Optimization section.

## Experiment Types

The Neural Network Optimization project offers three distinct types of experiments, each designed to optimize different aspects of neural networks:

### 1. Weight Optimization

**Purpose:** Directly optimize the weights of a neural network using evolutionary algorithms instead of traditional gradient-based methods like backpropagation.

**How it works:**
- The neural network architecture (number of layers and neurons) is fixed based on your selection
- The optimization algorithm (GA or PSO) searches for optimal weight values
- Each individual in the population (GA) or particle (PSO) represents a complete set of weights for the neural network
- The fitness/objective function evaluates how well the network performs with those weights on the training data
- Through evolution or particle movement, the algorithm converges toward optimal weight values

**Benefits:**
- Can potentially escape local minima that gradient-based methods might get stuck in
- Does not require computing gradients, which can be advantageous for certain types of networks or activation functions
- Can optimize non-differentiable objective functions

**Use cases:**
- When traditional training methods fail to converge
- For neural networks with non-differentiable components
- When exploring alternative training methods for comparison

### 2. Feature Selection

**Purpose:** Identify the most relevant features in your dataset that contribute most to model performance, while eliminating redundant or noisy features.

**How it works:**
- Each individual in the population (GA) or particle (PSO) represents a binary mask of features to include/exclude
- The algorithm searches for the optimal subset of features that maximizes model performance
- A neural network is trained using only the selected features and evaluated
- The fitness/objective function measures the performance of the model with the selected features

**Benefits:**
- Reduces dimensionality and complexity of the model
- Can improve model performance by removing noisy or irrelevant features
- Reduces overfitting by eliminating redundant information
- Improves model interpretability by focusing on the most important features
- Reduces computational requirements for training and inference

**Use cases:**
- Datasets with a large number of features
- When feature importance analysis is needed
- When model simplification is desired
- For improving model generalization

### 3. Hyperparameter Tuning

**Purpose:** Find the optimal hyperparameters for a neural network architecture to maximize its performance.

**How it works:**
- Each individual in the population (GA) or particle (PSO) represents a set of hyperparameters
- The algorithm searches for the optimal combination of hyperparameters
- For each set of hyperparameters, a neural network is created, trained, and evaluated
- The fitness/objective function measures the performance of the model with those hyperparameters

**Benefits:**
- Automates the process of finding optimal hyperparameters
- Can explore a much larger hyperparameter space than manual tuning
- Often finds better hyperparameter combinations than manual or grid search methods
- Reduces the need for expert knowledge in hyperparameter selection

**Use cases:**
- When designing new neural network architectures
- For maximizing model performance
- When comparing different model configurations
- For research purposes to understand the impact of different hyperparameters

## Weight Optimization Parameters

The Weight Optimization section in the experiments page contains parameters that control the neural network architecture that will be optimized. Here's a detailed explanation of each parameter:

### Hidden Layers

**Purpose:** Defines the architecture of the neural network by specifying the number of layers and neurons in each layer.

**Options:**
- **16 neurons (1 layer)**: A simple network with a single hidden layer containing 16 neurons
- **32 neurons (1 layer)**: A single hidden layer with 32 neurons, providing more capacity than the 16-neuron option
- **64 neurons (1 layer)**: A single hidden layer with 64 neurons, offering even more capacity
- **32 → 16 neurons (2 layers)**: A two-layer network with 32 neurons in the first hidden layer and 16 in the second
- **64 → 32 neurons (2 layers)**: A two-layer network with 64 neurons in the first hidden layer and 32 in the second
- **128 → 64 neurons (2 layers)**: A two-layer network with 128 neurons in the first hidden layer and 64 in the second
- **64 → 32 → 16 neurons (3 layers)**: A three-layer network with decreasing neuron counts (64, 32, 16)

**Impact:**
- More neurons and layers increase the network's capacity to learn complex patterns
- Deeper networks (more layers) can learn more abstract features
- Wider networks (more neurons per layer) can capture more information at each level of abstraction
- However, more complex networks require more weights to optimize, making the search space larger

**When to use different options:**
- Simple problems: Use simpler architectures (16 or 32 neurons in 1 layer)
- Complex problems with many features: Use deeper and wider networks
- If you're unsure: Start with "64 → 32 neurons (2 layers)" as a good balance

### Activation Function

**Purpose:** Determines the non-linear function applied to the output of each neuron in the hidden layers.

**Options:**
- **ReLU (Rectified Linear Unit)**: f(x) = max(0, x)
  - Fast to compute
  - Helps mitigate the vanishing gradient problem
  - Most commonly used in modern neural networks
  - Works well for many problems

- **Tanh (Hyperbolic Tangent)**: f(x) = tanh(x)
  - Output range: [-1, 1]
  - Zero-centered, which can help with learning
  - Smoother gradient than ReLU
  - Can suffer from vanishing gradient in deep networks

- **Sigmoid**: f(x) = 1 / (1 + e^(-x))
  - Output range: [0, 1]
  - Historically popular but less used in hidden layers now
  - Can suffer from vanishing gradient
  - Useful for binary classification output layers

- **ELU (Exponential Linear Unit)**: f(x) = x if x > 0, α * (e^x - 1) if x ≤ 0
  - Smoother than ReLU
  - Can produce negative outputs, unlike ReLU
  - Can help reduce the "dying ReLU" problem
  - Often performs better than ReLU in some cases

**Impact:**
- Different activation functions can significantly affect how well and how quickly the network learns
- Some activation functions work better for certain types of data or problems
- The choice can affect the optimization landscape that the GA or PSO algorithm needs to navigate

**When to use different options:**
- ReLU: Good default choice for most problems
- Tanh: When zero-centered activations are important
- Sigmoid: Rarely used in hidden layers in modern networks
- ELU: When you want to try something potentially better than ReLU for deep networks

## Optimization Algorithm Considerations

When running weight optimization experiments, the choice of optimization algorithm (GA or PSO) and their parameters can significantly impact the results:

### Genetic Algorithm (GA) for Weight Optimization

- **Population Size**: Larger populations can explore more of the weight space but require more computation
- **Mutation Rate**: Higher rates increase exploration but may disrupt good solutions
- **Mutation Type**: Different mutation types affect how the weights are perturbed
  - **Bit-flip**: Good for binary representations
  - **Inversion**: Can preserve some weight relationships
  - **Swap**: Maintains the same weights but in different positions
  - **Scramble**: More disruptive than swap, rearranges a subset of weights

### Particle Swarm Optimization (PSO) for Weight Optimization

- **Number of Particles**: Similar to population size in GA
- **Inertia Weight**: Controls how much particles maintain their current trajectory
- **Cognitive and Social Coefficients**: Balance between individual exploration and swarm convergence

## Best Practices

1. **Start Simple**: Begin with simpler network architectures before trying more complex ones
2. **Compare Algorithms**: Run experiments with both GA and PSO to see which performs better for your specific problem
3. **Iterative Refinement**: Use the results of initial experiments to guide parameter choices in subsequent runs
4. **Monitor Convergence**: Check if the algorithm is converging to a solution or getting stuck
5. **Save Results**: Always save promising results for later comparison and analysis

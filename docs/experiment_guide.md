# Experiment Guide

This document explains what happens when you select different experiment types and toggle various hyperparameter tuning parameters in the Optimization Project.

## Experiment Types

### 1. Weight Optimization

**What Happens:**
- The system uses either Genetic Algorithm (GA) or Particle Swarm Optimization (PSO) to find the optimal weights for a neural network.
- Instead of using traditional gradient descent, the optimization algorithm directly searches for the best weight values.
- The process starts with random weights (either as chromosomes in GA or particles in PSO).
- Through multiple generations/iterations, the weights evolve toward values that minimize the loss function.

**Output:**
- Convergence graph showing how the fitness (accuracy) improves over generations/iterations
- Comparison between GA and PSO performance if both are selected
- Final accuracy on test data
- Training time comparison
- Visualization of weight distributions

**Behind the Scenes:**
- The neural network architecture remains fixed
- Only the weights connecting neurons are optimized
- The system uses a fitness function that evaluates each set of weights by training the model and measuring validation accuracy
- Early stopping is implemented to prevent overfitting

### 2. Feature Selection

**What Happens:**
- The system determines which input features are most important for predicting the target variable.
- It uses binary encoding where each bit represents whether a feature is included (1) or excluded (0).
- The optimization algorithm searches for the optimal subset of features that maximizes model performance.
- This helps reduce dimensionality and improve model interpretability.

**Output:**
- List of selected features ranked by importance
- Performance comparison between using all features vs. selected features
- Visualization showing the impact of feature selection on model accuracy
- Training time reduction statistics
- Feature importance graph

**Behind the Scenes:**
- Each chromosome (GA) or particle position (PSO) represents a binary mask for feature selection
- The system trains models using only the selected features
- It balances the trade-off between model complexity (number of features) and performance
- The fitness function rewards both high accuracy and smaller feature subsets

### 3. Hyperparameter Tuning

**What Happens:**
- The system searches for the optimal combination of hyperparameters for the neural network.
- Instead of manually testing different configurations, the optimization algorithm automatically explores the hyperparameter space.
- It evaluates various combinations of hyperparameters to find those that yield the best performance.
- This process helps automate the tedious task of hyperparameter selection.

**Output:**
- Best hyperparameter configuration found
- Performance comparison between different hyperparameter settings
- Convergence graph showing improvement over generations/iterations
- Visualization of hyperparameter importance
- Training and inference time statistics

**Behind the Scenes:**
- The system encodes hyperparameters into chromosomes (GA) or particle positions (PSO)
- It trains multiple models with different hyperparameter configurations
- A subset of training data is used for faster evaluation during optimization
- The final model is trained with the best hyperparameters on the full training set
- Early stopping is used to prevent wasting computation on poor configurations

## Hyperparameter Tuning Parameters

### Hidden Layers

**What Actually Happens:**
- When toggled ON: The system will test different neural network architectures by varying the number and size of hidden layers.
- The optimization algorithm will search through predefined configurations like [16], [32], [64], [32, 16], [64, 32, 16], etc.
- Each configuration represents a different network depth and width.
- The system evaluates how these different architectures affect model performance.

**When toggled OFF:**
- The system uses a default hidden layer configuration (typically [64, 32]) and does not vary the network architecture.

### Learning Rate

**What Actually Happens:**
- When toggled ON: The system tests different learning rate values (typically 0.0001, 0.001, 0.01, 0.1).
- The learning rate controls how quickly the model adapts to the problem.
- Too high: The model may converge quickly but miss the optimal solution.
- Too low: The model may take too long to train or get stuck in local minima.
- The optimization algorithm searches for the learning rate that provides the best balance.

**When toggled OFF:**
- The system uses a default learning rate (typically 0.01) for all evaluations.

### Activation Function

**What Actually Happens:**
- When toggled ON: The system tests different activation functions (ReLU, tanh, sigmoid, ELU) for the hidden layers.
- Each activation function has different properties:
  - ReLU: Fast computation, helps with vanishing gradient, but can suffer from "dying ReLU" problem
  - tanh: Outputs between -1 and 1, zero-centered, but can suffer from vanishing gradient
  - sigmoid: Outputs between 0 and 1, good for binary classification outputs, but can suffer from vanishing gradient
  - ELU: Similar to ReLU but handles negative inputs differently, can help with the "dying ReLU" problem
- The optimization algorithm finds which activation function works best for your specific data.

**When toggled OFF:**
- The system uses a default activation function (typically ReLU) for all evaluations.

### Batch Size

**What Actually Happens:**
- When toggled ON: The system tests different batch sizes (typically 16, 32, 64, 128, 256).
- Batch size affects:
  - Training speed: Larger batches generally allow faster training but require more memory
  - Generalization: Smaller batches often provide better generalization but with more noisy gradients
  - Memory usage: Larger batches require more GPU/CPU memory
- The optimization algorithm finds the batch size that provides the best balance for your data.

**When toggled OFF:**
- The system uses a default batch size (typically 32) for all evaluations.

### Dropout Rate

**What Actually Happens:**
- When toggled ON: The system tests different dropout rates (typically 0.0, 0.1, 0.2, 0.3, 0.5).
- Dropout randomly deactivates a percentage of neurons during training to prevent overfitting.
- Higher dropout rates:
  - Increase regularization strength
  - Help prevent overfitting
  - May require longer training time
- Lower dropout rates:
  - Allow faster convergence
  - May lead to overfitting on smaller datasets
- The optimization algorithm finds the dropout rate that provides the best regularization for your data.

**When toggled OFF:**
- The system uses a default dropout rate (typically 0.0, meaning no dropout) for all evaluations.

### Optimizer

**What Actually Happens:**
- When toggled ON: The system tests different optimizers (Adam, SGD, RMSprop, Adagrad).
- Each optimizer has different properties:
  - Adam: Adaptive learning rates for each parameter, generally works well for most problems
  - SGD: Simple but may require careful tuning of learning rate and momentum
  - RMSprop: Good for RNNs and handles non-stationary objectives well
  - Adagrad: Adapts learning rates based on parameter frequency, good for sparse data
- The optimization algorithm finds which optimizer works best for your specific data and model.

**When toggled OFF:**
- The system uses a default optimizer (typically Adam) for all evaluations.

## Practical Tips

1. **Start Simple**: Begin with a single experiment type and a limited set of hyperparameters to tune.

2. **Computational Resources**: Hyperparameter tuning is computationally intensive. If you have limited resources:
   - Toggle OFF parameters that are less likely to impact your specific problem
   - Reduce the population size and number of generations in the settings
   - Use a smaller subset of your data for initial experiments

3. **Experiment Combinations**: Different experiment types can be combined for more comprehensive optimization:
   - Feature Selection → Hyperparameter Tuning: First find the best features, then optimize hyperparameters
   - Hyperparameter Tuning → Weight Optimization: First find the best architecture, then optimize weights

4. **Interpreting Results**: Pay attention to not just the final accuracy but also:
   - Convergence speed: How quickly did the algorithm find good solutions?
   - Consistency: Did multiple runs produce similar results?
   - Generalization: Does performance on validation data match test data?

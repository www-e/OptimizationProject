# How Optimization Algorithms Work with Your Data

## Data Flow Overview

The optimization algorithms (Genetic Algorithm and Particle Swarm Optimization) don't run directly on the CSV file. Instead, they operate on processed data extracted from your dataset. Here's the complete flow:

## 1. Data Loading Process

When you upload a CSV file through the web interface:

- The `DataLoader` class parses your file
- It extracts:
  - **Features (X)**: The input variables from your dataset
  - **Target (y)**: The output variable you're trying to predict
- Feature names and target name are identified
- Data dimensions are calculated (number of samples, number of features)

## 2. Data Preprocessing

Your raw data is then processed and split into:

- **Training set**: Used to train the neural network model
- **Validation set**: Used during optimization to evaluate fitness of solutions
- **Test set**: Used for final evaluation after optimization is complete

This splitting ensures that we can properly evaluate model performance on unseen data.

## 3. What the Algorithms Actually Optimize

Depending on the experiment type, the algorithms optimize different aspects:

### Weight Optimization
- The algorithms search for optimal neural network weights
- Each "individual" or "particle" represents a complete set of weights for the network
- The goal is to find weights that minimize error or maximize accuracy

### Feature Selection
- The algorithms determine which input features are most important
- Each solution represents a binary mask of features to include/exclude
- The goal is to find the minimal set of features that maintains high performance

### Hyperparameter Tuning
- The algorithms find the best neural network architecture and learning parameters
- Each solution represents a specific configuration (number of layers, neurons, etc.)
- The goal is to find the most efficient and effective network structure

## 4. The Fitness Function

At the heart of both GA and PSO is the fitness function:

```python
def _fitness_function(self, weights):
    # Set weights to the neural network
    self.nn.set_weights_flat(weights)
    
    # Evaluate on validation set
    predictions = self.nn.predict(self.X_val)
    
    # Calculate accuracy or other metrics
    # ...
    
    return accuracy  # or other fitness metric
```

This function:
- Takes a potential solution (weights, feature mask, or hyperparameters)
- Applies it to the neural network
- Evaluates how well the network performs on the validation data
- Returns a fitness score (usually accuracy or error rate)

## 5. Optimization Process

1. The algorithm starts with random solutions
2. It evaluates each solution using the fitness function
3. It iteratively improves solutions through:
   - **GA**: Selection, crossover, and mutation
   - **PSO**: Particle movement based on personal and global best positions
4. The process continues for a specified number of generations/iterations
5. The best solution found is returned and applied to the model

## 6. Final Evaluation

After optimization:
- The best solution is applied to the neural network
- The network is evaluated on the test set (data it hasn't seen before)
- Performance metrics are calculated and displayed
- Visualizations are generated to help interpret the results

## Summary

The algorithms run on the neural network's parameters (weights, architecture, etc.) and use your dataset (originally from the CSV) to evaluate how good each potential solution is. The goal is to find the optimal parameters that make the neural network perform best on your specific data.

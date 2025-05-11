# Optimization Algorithms Documentation

This document provides detailed information about the optimization algorithms implemented in the project, including their inputs, outputs, and functionality.

## 1. Genetic Algorithm (GA)

### Overview
The Genetic Algorithm is an evolutionary optimization algorithm inspired by natural selection. It evolves a population of candidate solutions through selection, crossover, and mutation operations to find optimal or near-optimal solutions to complex problems.

### Implementation
Located in: `algorithms/genetic_algorithm.py`

### Parameters

#### Population Parameters
- **Population Size**: Number of individuals in the population
  - Type: Integer
  - Default: 50
  - Effect: Larger populations provide more diversity but require more computational resources

- **Number of Generations**: Number of generations to evolve
  - Type: Integer
  - Default: 100
  - Effect: More generations allow for more evolution but increase computation time

- **Genotype**: Type of encoding for chromosomes
  - Type: String (options: 'binary', 'real', 'integer')
  - Default: 'real'
  - Effect: Determines how solutions are represented and how genetic operators work

#### Genetic Operators
- **Selection Method**: Method for selecting parents
  - Type: String (options: 'tournament', 'roulette', 'rank')
  - Default: 'tournament'
  - Effect: Determines how parents are selected for reproduction

- **Mutation Rate**: Probability of mutation for each gene
  - Type: Float (0.0 to 1.0)
  - Default: 0.2
  - Effect: Higher rates increase exploration but may disrupt good solutions

- **Mutation Type**: Type of mutation operation
  - Type: String (options: 'bit-flip', 'inversion', 'swap', 'scramble')
  - Default: 'bit-flip'
  - Effect: Determines how mutations occur
  - Details:
    - **Bit-flip**: Flips individual bits/values with a probability equal to the mutation rate
    - **Inversion**: Selects two random points and inverts the sequence between them
    - **Swap**: Swaps two random genes
    - **Scramble**: Randomly shuffles a subset of genes

- **Crossover Type**: Type of crossover operation
  - Type: String (options: 'single_point', 'two_point', 'uniform')
  - Default: 'single_point'
  - Effect: Determines how genetic material is exchanged between parents

- **Elitism**: Whether to preserve best individuals
  - Type: Boolean
  - Default: True
  - Effect: Ensures the best solutions are preserved across generations

- **Elite Size**: Number of elite individuals to preserve
  - Type: Integer
  - Default: 2
  - Effect: Determines how many top individuals are preserved

#### Advanced Options
- **Adaptive Mutation**: Adjust mutation rate dynamically
  - Type: Boolean
  - Default: False
  - Effect: Allows the mutation rate to change during evolution

### Input
- **Fitness Function**: Function that evaluates the quality of a solution
  - Type: Function
  - Requirements: Must accept a chromosome and return a fitness score (higher is better)

- **Chromosome Length**: Length of each chromosome
  - Type: Integer
  - Requirements: Must be positive

### Output
- **Best Chromosome**: The best solution found during evolution
  - Type: Array/List
  - Format: Depends on chromosome type (binary, real, or integer)

- **Best Fitness**: The fitness value of the best solution
  - Type: Float
  - Interpretation: Higher values indicate better solutions

- **Fitness History**: Record of fitness values across generations
  - Type: Dictionary with lists
  - Contents: Average and best fitness values for each generation

### Algorithm Process
1. **Initialization**: Create an initial population of random chromosomes
2. **Evaluation**: Calculate fitness for each chromosome
3. **Selection**: Select parents based on fitness using the specified selection method
4. **Crossover**: Create offspring by combining genetic material from parents
5. **Mutation**: Introduce random changes to maintain diversity
6. **Replacement**: Form a new population from offspring and possibly elite individuals
7. **Termination**: Stop after a specified number of generations or when convergence criteria are met

### Usage Example
```python
from algorithms.genetic_algorithm import GeneticAlgorithm

# Define a fitness function
def fitness_function(chromosome):
    # Example: maximize the sum of elements
    return sum(chromosome)

# Create and run the GA
ga = GeneticAlgorithm(
    fitness_function=fitness_function,
    chromosome_length=10,
    population_size=50,
    num_generations=100,
    mutation_rate=0.2,
    mutation_type='bit-flip',
    selection_method='tournament',
    elitism=True,
    elite_size=2,
    chromosome_type='real',
    crossover_type='single_point'
)

# Evolve the population
best_solution, best_fitness = ga.evolve(verbose=True)

# Get the evolution history
history = ga.get_history()
```

## 2. Particle Swarm Optimization (PSO)

### Overview
Particle Swarm Optimization is a population-based optimization algorithm inspired by social behavior of bird flocking or fish schooling. It optimizes a problem by moving particles (candidate solutions) in the search space according to mathematical formulas for the particle's position and velocity.

### Implementation
Located in: `algorithms/particle_swarm.py`

### Parameters

#### Swarm Parameters
- **Number of Particles**: Number of particles in the swarm
  - Type: Integer
  - Default: 50
  - Effect: More particles provide better exploration but require more computation

- **Number of Iterations**: Number of iterations to run
  - Type: Integer
  - Default: 100
  - Effect: More iterations allow for more refinement but increase computation time

- **Value Range**: Range of values for particle positions
  - Type: Tuple (min, max)
  - Default: (-1, 1)
  - Effect: Defines the search space boundaries

- **Discrete**: Whether to use discrete (integer) positions
  - Type: Boolean
  - Default: False
  - Effect: Determines if positions are continuous or discrete

#### Movement Parameters
- **Inertia Weight**: Weight of particle's velocity
  - Type: Float
  - Default: 0.7
  - Effect: Controls the influence of the previous velocity

- **Cognitive Coefficient**: Weight of particle's personal best
  - Type: Float
  - Default: 1.5
  - Effect: Controls the influence of the particle's memory

- **Social Coefficient**: Weight of swarm's global best
  - Type: Float
  - Default: 1.5
  - Effect: Controls the influence of the swarm's knowledge

- **Maximum Velocity**: Maximum allowed velocity
  - Type: Float
  - Default: 1.0
  - Effect: Prevents particles from moving too quickly

#### Advanced Options
- **Decreasing Inertia**: Whether to decrease inertia weight over time
  - Type: Boolean
  - Default: False
  - Effect: Allows for more exploration early and more exploitation later

- **Final Inertia Weight**: Final value for inertia if decreasing
  - Type: Float
  - Default: 0.4
  - Effect: Determines the minimum inertia weight

- **Use Neighborhood**: Whether to use neighborhood topology
  - Type: Boolean
  - Default: False
  - Effect: Particles are influenced by neighbors instead of global best

- **Neighborhood Size**: Size of neighborhood if used
  - Type: Integer
  - Default: 3
  - Effect: Determines how many neighbors influence each particle

### Input
- **Objective Function**: Function that evaluates the quality of a solution
  - Type: Function
  - Requirements: Must accept a particle position and return a fitness score (higher is better)

- **Dimensions**: Number of dimensions in the search space
  - Type: Integer
  - Requirements: Must be positive

### Output
- **Best Position**: The best solution found during optimization
  - Type: Array/List
  - Format: Real or integer values depending on the 'discrete' parameter

- **Best Fitness**: The fitness value of the best solution
  - Type: Float
  - Interpretation: Higher values indicate better solutions

- **Fitness History**: Record of fitness values across iterations
  - Type: Dictionary with lists
  - Contents: Average and best fitness values for each iteration

### Algorithm Process
1. **Initialization**: Create particles with random positions and velocities
2. **Evaluation**: Calculate fitness for each particle
3. **Update Personal Best**: Update each particle's personal best if current position is better
4. **Update Global Best**: Update the swarm's global best
5. **Update Velocity**: Update each particle's velocity based on inertia, cognitive, and social components
6. **Update Position**: Move particles according to their velocities
7. **Termination**: Stop after a specified number of iterations or when convergence criteria are met

### Usage Example
```python
from algorithms.particle_swarm import ParticleSwarmOptimization

# Define an objective function
def objective_function(position):
    # Example: maximize the sum of elements
    return sum(position)

# Create and run the PSO
pso = ParticleSwarmOptimization(
    objective_function=objective_function,
    dimensions=10,
    num_particles=50,
    num_iterations=100,
    inertia_weight=0.7,
    cognitive_coefficient=1.5,
    social_coefficient=1.5,
    max_velocity=1.0,
    value_range=(-1, 1),
    discrete=False
)

# Optimize
best_position, best_fitness = pso.optimize(verbose=True)

# Get the optimization history
history = pso.get_history()
```

## Experiment Types

The project implements several experiment types that use the optimization algorithms:

### 1. Weight Optimization
- **Purpose**: Optimize the weights of a neural network
- **Implementation**: `experiments/weight_optimization.py`
- **Process**: Uses GA or PSO to find optimal weights for a fixed neural network architecture
- **Inputs**: Dataset, neural network architecture, optimization algorithm parameters
- **Outputs**: Optimized neural network weights, performance metrics

### 2. Feature Selection
- **Purpose**: Select the most relevant features for a machine learning task
- **Implementation**: `experiments/feature_selection.py`
- **Process**: Uses GA or PSO to find an optimal subset of features
- **Inputs**: Dataset with features, optimization algorithm parameters
- **Outputs**: Selected features, performance metrics with the selected features

### 3. Hyperparameter Tuning
- **Purpose**: Find optimal hyperparameters for a neural network
- **Implementation**: `experiments/hyperparameter_tuning.py`
- **Process**: Uses GA or PSO to search the hyperparameter space
- **Inputs**: Dataset, hyperparameter ranges, optimization algorithm parameters
- **Outputs**: Optimal hyperparameters, performance metrics with those hyperparameters

"""
Particle Swarm Optimization implementation for neural network optimization.
"""

import numpy as np
from tqdm import tqdm


class Particle:
    """
    Represents a single particle in the PSO algorithm.
    """
    
    def __init__(self, dimensions, value_range=(-1, 1)):
        """
        Initialize a particle with random position and velocity.
        
        Args:
            dimensions: Number of dimensions in the search space
            value_range: Range of values for position (min, max)
        """
        self.dimensions = dimensions
        self.value_range = value_range
        
        # Initialize position and velocity efficiently
        min_val, max_val = value_range
        self.position = np.random.uniform(min_val, max_val, dimensions)
        
        # Set initial velocity as a percentage of the position range
        velocity_range = (max_val - min_val) * 0.1
        self.velocity = np.random.uniform(-velocity_range, velocity_range, dimensions)
        
        # Initialize personal best
        self.personal_best_position = self.position.copy()
        self.personal_best_fitness = float('-inf')
        
        # Current fitness
        self.fitness = float('-inf')


class ParticleSwarmOptimization:
    """
    Particle Swarm Optimization for optimization problems.
    Can be used for neural network weight optimization, feature selection, 
    or hyperparameter tuning.
    """
    
    def __init__(self, fitness_function, dimensions, 
                 num_particles=50, num_iterations=100,
                 inertia_weight=0.7, cognitive_coefficient=1.5, social_coefficient=1.5,
                 max_velocity=0.5, value_range=(-1, 1), discrete=False):
        """
        Initialize the PSO algorithm.
        
        Args:
            fitness_function: Function that evaluates fitness of a particle's position
            dimensions: Number of dimensions in the search space
            num_particles: Number of particles in the swarm
            num_iterations: Number of iterations to run
            inertia_weight: Weight of particle's velocity (w)
            cognitive_coefficient: Weight of particle's personal best (c1)
            social_coefficient: Weight of swarm's global best (c2)
            max_velocity: Maximum velocity of particles
            value_range: Range of values for position (min, max)
            discrete: Whether to discretize positions (for binary problems)
        """
        self.fitness_function = fitness_function
        self.dimensions = dimensions
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.inertia_weight = inertia_weight
        self.cognitive_coefficient = cognitive_coefficient
        self.social_coefficient = social_coefficient
        self.max_velocity = max_velocity
        self.value_range = value_range
        self.discrete = discrete
        
        # Initialize particles
        self.particles = [Particle(dimensions, value_range) for _ in range(num_particles)]
        
        # Initialize global best
        self.global_best_position = None
        self.global_best_fitness = float('-inf')
        
        # History for analysis - initialize before first evaluation
        self.fitness_history = []
        self.best_fitness_history = []
        
        # Ensure we have a valid initial position
        self._evaluate_fitness()
        
        # If we still don't have a valid position, initialize with zeros
        if self.global_best_position is None:
            self.global_best_position = np.zeros(dimensions)
            self.global_best_fitness = 0.0
    
    def _evaluate_fitness(self):
        """Evaluate fitness for all particles."""
        # Prepare positions for evaluation
        positions = np.array([p.position for p in self.particles])
        
        # Discretize if needed
        if self.discrete:
            evaluation_positions = self._discretize_positions(positions)
        else:
            evaluation_positions = positions
            
        # Try vectorized evaluation if possible
        try:
            # Try to evaluate all particles at once
            fitness_values = np.array([self.fitness_function(pos) for pos in evaluation_positions])
            
            # Update each particle's fitness
            for i, particle in enumerate(self.particles):
                particle.fitness = fitness_values[i]
                
                # Update personal best
                if particle.fitness > particle.personal_best_fitness:
                    particle.personal_best_fitness = particle.fitness
                    particle.personal_best_position = positions[i].copy()
                
                # Update global best
                if particle.fitness > self.global_best_fitness:
                    self.global_best_fitness = particle.fitness
                    self.global_best_position = positions[i].copy()
        except:
            # Fall back to individual evaluation
            for i, particle in enumerate(self.particles):
                # Get position
                position = evaluation_positions[i]
                
                # Evaluate fitness
                particle.fitness = self.fitness_function(position)
                
                # Update personal best
                if particle.fitness > particle.personal_best_fitness:
                    particle.personal_best_fitness = particle.fitness
                    particle.personal_best_position = positions[i].copy()
                
                # Update global best
                if particle.fitness > self.global_best_fitness:
                    self.global_best_fitness = particle.fitness
                    self.global_best_position = positions[i].copy()
        
        # Store history
        avg_fitness = np.mean([p.fitness for p in self.particles])
        self.fitness_history.append(avg_fitness)
        self.best_fitness_history.append(self.global_best_fitness)
    
    def _discretize_position(self, position):
        """Convert continuous position to binary (for feature selection)."""
        return (position > 0).astype(int)
        
    def _discretize_positions(self, positions):
        """Convert multiple continuous positions to binary efficiently."""
        return (positions > 0).astype(int)
    
    def _update_velocities_and_positions(self):
        """Update velocities and positions of all particles."""
        min_val, max_val = self.value_range
        
        # Vectorized update for all particles
        positions = np.array([p.position for p in self.particles])
        velocities = np.array([p.velocity for p in self.particles])
        personal_bests = np.array([p.personal_best_position for p in self.particles])
        
        # Generate random coefficients for all particles at once
        r1 = np.random.random((self.num_particles, self.dimensions))
        r2 = np.random.random((self.num_particles, self.dimensions))
        
        # Calculate cognitive and social components
        cognitive_component = self.cognitive_coefficient * r1 * (personal_bests - positions)
        
        # Broadcast global best for all particles
        global_best_broadcast = np.tile(self.global_best_position, (self.num_particles, 1))
        social_component = self.social_coefficient * r2 * (global_best_broadcast - positions)
        
        # Update velocities
        new_velocities = (self.inertia_weight * velocities + 
                         cognitive_component + 
                         social_component)
        
        # Clamp velocities
        new_velocities = np.clip(new_velocities, -self.max_velocity, self.max_velocity)
        
        # Update positions
        new_positions = positions + new_velocities
        
        # Clamp positions
        new_positions = np.clip(new_positions, min_val, max_val)
        
        # Update particle objects
        for i, particle in enumerate(self.particles):
            particle.velocity = new_velocities[i]
            particle.position = new_positions[i]
    
    def optimize(self, verbose=True):
        """
        Run the PSO algorithm for the specified number of iterations.
        
        Args:
            verbose: Whether to display progress information
        
        Returns:
            global_best_position: The best position found
            global_best_fitness: The fitness of the best position
        """
        # Reset history for new optimization run
        self.fitness_history = []
        self.best_fitness_history = []
        
        # Initial fitness evaluation
        self._evaluate_fitness()
        
        # Track iterations without improvement for early stopping
        stagnation_counter = 0
        last_best_fitness = self.global_best_fitness
        
        # Optimization loop
        iterator = tqdm(range(self.num_iterations)) if verbose else range(self.num_iterations)
        for iteration in iterator:
            # Update velocities and positions
            self._update_velocities_and_positions()
            
            # Evaluate fitness
            self._evaluate_fitness()
            
            # Check for improvement
            if self.global_best_fitness > last_best_fitness + 1e-6:
                # Reset counter if improvement found
                stagnation_counter = 0
                last_best_fitness = self.global_best_fitness
            else:
                stagnation_counter += 1
            
            # Early stopping if no improvement for a while
            if stagnation_counter >= 20 and iteration > 20:
                if verbose:
                    tqdm.write(f"Early stopping at iteration {iteration + 1} due to no improvement")
                break
            
            # Display progress
            if verbose and (iteration + 1) % 10 == 0:
                tqdm.write(f"Iteration {iteration + 1}/{self.num_iterations}, "
                          f"Best Fitness: {self.global_best_fitness:.4f}, "
                          f"Avg Fitness: {self.fitness_history[-1] if self.fitness_history else 0:.4f}")
        
        # Return best solution
        if self.discrete:
            return self._discretize_position(self.global_best_position), self.global_best_fitness
        return self.global_best_position, self.global_best_fitness
    
    def get_history(self):
        """Get the history of fitness values for analysis."""
        # Ensure history lists exist and have values
        if not hasattr(self, 'fitness_history') or self.fitness_history is None:
            self.fitness_history = []
        if not hasattr(self, 'best_fitness_history') or self.best_fitness_history is None:
            self.best_fitness_history = []
            
        return {
            'avg_fitness': self.fitness_history,
            'best_fitness': self.best_fitness_history
        }

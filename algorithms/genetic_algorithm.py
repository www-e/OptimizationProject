"""
Genetic Algorithm implementation for neural network optimization.
"""

import numpy as np
from tqdm import tqdm
import random


class GeneticAlgorithm:
    """
    Genetic Algorithm for optimization problems.
    Can be used for neural network weight optimization, feature selection, 
    or hyperparameter tuning.
    """
    
    def __init__(self, fitness_function, chromosome_length, 
                 population_size=50, num_generations=100,
                 mutation_rate=0.2, mutation_type='bit-flip',
                 selection_method='tournament', tournament_size=3,
                 elitism=True, elite_size=2, 
                 chromosome_type='binary', value_range=(-1, 1),
                 crossover_type='single_point'):
        """
        Initialize the Genetic Algorithm.
        
        Args:
            fitness_function: Function that evaluates fitness of a chromosome
            chromosome_length: Length of each chromosome
            population_size: Number of individuals in the population
            num_generations: Number of generations to evolve
            mutation_rate: Probability of mutation
            mutation_type: Type of mutation ('bit-flip', 'inversion', 'swap', 'scramble')
            selection_method: Method for parent selection ('roulette', 'tournament', 'rank')
            tournament_size: Size of tournament if tournament selection is used
            elitism: Whether to use elitism (preserving best individuals)
            elite_size: Number of elite individuals to preserve
            chromosome_type: Type of chromosome ('binary', 'real', 'integer')
            value_range: Range of values for real-valued chromosomes (min, max)
            crossover_type: Type of crossover ('single_point', 'two_point', 'uniform')
        """
        self.fitness_function = fitness_function
        self.chromosome_length = chromosome_length
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.mutation_type = mutation_type
        self.selection_method = selection_method
        self.tournament_size = tournament_size
        self.elitism = elitism
        self.elite_size = elite_size
        self.chromosome_type = chromosome_type
        self.value_range = value_range
        self.crossover_type = crossover_type
        
        # Initialize population
        self.population = self._initialize_population()
        self.fitness_scores = np.zeros(self.population_size)
        
        # Keep track of best solution
        self.best_chromosome = None
        self.best_fitness = float('-inf')
        
        # History for analysis
        self.fitness_history = []
        self.best_fitness_history = []
    
    def _initialize_population(self):
        """Initialize the population based on chromosome type."""
        if self.chromosome_type == 'binary':
            return np.random.randint(0, 2, size=(self.population_size, self.chromosome_length))
        elif self.chromosome_type == 'real':
            min_val, max_val = self.value_range
            return np.random.uniform(min_val, max_val, size=(self.population_size, self.chromosome_length))
        elif self.chromosome_type == 'integer':
            min_val, max_val = self.value_range
            return np.random.randint(min_val, max_val + 1, size=(self.population_size, self.chromosome_length))
        else:
            raise ValueError(f"Unsupported chromosome type: {self.chromosome_type}")
    
    def _evaluate_fitness(self):
        """Evaluate fitness for the entire population."""
        # Vectorized fitness evaluation if possible
        try:
            # Try to evaluate fitness for all chromosomes at once
            self.fitness_scores = np.array([self.fitness_function(chrom) for chrom in self.population])
        except:
            # Fall back to individual evaluation if vectorized approach fails
            for i in range(self.population_size):
                self.fitness_scores[i] = self.fitness_function(self.population[i])
        
        # Find best solution
        best_idx = np.argmax(self.fitness_scores)
        if self.fitness_scores[best_idx] > self.best_fitness:
            self.best_fitness = self.fitness_scores[best_idx]
            self.best_chromosome = self.population[best_idx].copy()
        
        # Store history
        self.fitness_history.append(np.mean(self.fitness_scores))
        self.best_fitness_history.append(self.best_fitness)
    
    def _selection(self):
        """Select parents for reproduction based on the selection method."""
        if self.selection_method == 'roulette':
            return self._roulette_wheel_selection()
        elif self.selection_method == 'tournament':
            return self._tournament_selection()
        elif self.selection_method == 'rank':
            return self._rank_selection()
        else:
            raise ValueError(f"Unsupported selection method: {self.selection_method}")
    
    def _roulette_wheel_selection(self):
        """Roulette wheel selection based on fitness proportionate selection."""
        # Adjust fitness scores to be positive for roulette wheel selection
        adjusted_fitness = self.fitness_scores - np.min(self.fitness_scores) + 1e-10
        
        # Calculate selection probabilities
        selection_probs = adjusted_fitness / np.sum(adjusted_fitness)
        
        # Select two parents
        parent_indices = np.random.choice(
            self.population_size, 
            size=2, 
            p=selection_probs, 
            replace=False
        )
        
        return self.population[parent_indices[0]], self.population[parent_indices[1]]
    
    def _tournament_selection(self):
        """Tournament selection."""
        parents = []
        
        for _ in range(2):  # Select two parents
            # Randomly select tournament_size individuals
            tournament_indices = np.random.choice(
                self.population_size, 
                size=self.tournament_size, 
                replace=False
            )
            
            # Find the best individual in the tournament
            tournament_fitness = self.fitness_scores[tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            
            parents.append(self.population[winner_idx])
        
        return parents[0], parents[1]
    
    def _rank_selection(self):
        """Rank-based selection."""
        # Get ranks (higher fitness = higher rank)
        ranks = np.argsort(np.argsort(-self.fitness_scores)) + 1
        
        # Calculate selection probabilities based on ranks
        selection_probs = ranks / np.sum(ranks)
        
        # Select two parents
        parent_indices = np.random.choice(
            self.population_size, 
            size=2, 
            p=selection_probs, 
            replace=False
        )
        
        return self.population[parent_indices[0]], self.population[parent_indices[1]]
    
    def _crossover(self, parent1, parent2):
        """Perform crossover between two parents."""
        # Fixed crossover rate of 0.8
        if np.random.random() > 0.8:
            # No crossover, return copies of parents
            return parent1.copy(), parent2.copy()
        
        # Use the crossover type specified in the parameters
        if self.crossover_type == 'single_point':
            # Single-point crossover
            crossover_point = np.random.randint(1, self.chromosome_length)
            child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
            child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
        
        elif self.crossover_type == 'two_point':
            # Two-point crossover
            points = sorted(np.random.choice(range(1, self.chromosome_length), size=2, replace=False))
            child1 = np.concatenate([parent1[:points[0]], parent2[points[0]:points[1]], parent1[points[1]:]])
            child2 = np.concatenate([parent2[:points[0]], parent1[points[0]:points[1]], parent2[points[1]:]])
        
        else:  # uniform
            # Uniform crossover
            mask = np.random.randint(0, 2, size=self.chromosome_length).astype(bool)
            child1 = np.where(mask, parent1, parent2)
            child2 = np.where(mask, parent2, parent1)
        
        return child1, child2
    
    def _mutation(self, chromosome):
        """Apply mutation to a chromosome based on the mutation type."""
        mutated_chromosome = chromosome.copy()
        
        # Determine if mutation should happen at all based on mutation rate
        if np.random.random() > self.mutation_rate:
            return mutated_chromosome
            
        # Apply mutations based on mutation type and chromosome type
        if self.mutation_type == 'bit-flip':
            # Create a mutation mask for all genes at once
            mutation_mask = np.random.random(self.chromosome_length) < self.mutation_rate
            
            # Skip if no mutations
            if not np.any(mutation_mask):
                return mutated_chromosome
                
            if self.chromosome_type == 'binary':
                # Flip bits where mutation occurs
                mutated_chromosome[mutation_mask] = 1 - mutated_chromosome[mutation_mask]
            
            elif self.chromosome_type == 'real':
                # Add small random values where mutation occurs
                min_val, max_val = self.value_range
                mutation_strength = (max_val - min_val) * 0.1
                
                # Generate random changes only for genes that will mutate
                mutation_changes = np.random.uniform(
                    -mutation_strength, 
                    mutation_strength, 
                    size=np.sum(mutation_mask)
                )
                
                # Apply changes
                mutated_chromosome[mutation_mask] += mutation_changes
                
                # Ensure values are within range
                mutated_chromosome = np.clip(mutated_chromosome, min_val, max_val)
            
            elif self.chromosome_type == 'integer':
                # Change to random integers in range where mutation occurs
                min_val, max_val = self.value_range
                mutation_indices = np.where(mutation_mask)[0]
                
                mutated_chromosome[mutation_indices] = np.random.randint(
                    min_val, 
                    max_val + 1, 
                    size=len(mutation_indices)
                )
                
        elif self.mutation_type == 'inversion':
            # Select two random points and invert the sequence between them
            if self.chromosome_length <= 2:
                return mutated_chromosome
                
            # Select two random distinct points
            point1, point2 = sorted(np.random.choice(self.chromosome_length, 2, replace=False))
            
            # Invert the sequence between the two points
            mutated_chromosome[point1:point2+1] = mutated_chromosome[point1:point2+1][::-1]
            
        elif self.mutation_type == 'swap':
            # Swap two random genes
            if self.chromosome_length <= 1:
                return mutated_chromosome
                
            # Select two random distinct points
            point1, point2 = np.random.choice(self.chromosome_length, 2, replace=False)
            
            # Swap the genes
            mutated_chromosome[point1], mutated_chromosome[point2] = \
                mutated_chromosome[point2], mutated_chromosome[point1]
                
        elif self.mutation_type == 'scramble':
            # Scramble a randomly selected subset of genes
            if self.chromosome_length <= 2:
                return mutated_chromosome
                
            # Select two random distinct points
            point1, point2 = sorted(np.random.choice(self.chromosome_length, 2, replace=False))
            
            # Get the subset to scramble
            subset = mutated_chromosome[point1:point2+1].copy()
            
            # Scramble the subset
            np.random.shuffle(subset)
            
            # Put the scrambled subset back
            mutated_chromosome[point1:point2+1] = subset
        
        return mutated_chromosome
    
    def evolve(self, verbose=True):
        """
        Evolve the population for the specified number of generations.
        
        Args:
            verbose: Whether to display progress information
        
        Returns:
            best_chromosome: The best chromosome found
            best_fitness: The fitness of the best chromosome
        """
        # Initial fitness evaluation
        self._evaluate_fitness()
        
        # Pre-allocate new population array for better performance
        new_population = np.zeros((self.population_size, self.chromosome_length))
        
        # Evolution loop
        iterator = tqdm(range(self.num_generations)) if verbose else range(self.num_generations)
        for generation in iterator:
            new_pop_idx = 0
            
            # Elitism: keep the best individuals
            if self.elitism and self.elite_size > 0:
                elite_indices = np.argsort(self.fitness_scores)[-self.elite_size:]
                for idx in elite_indices:
                    new_population[new_pop_idx] = self.population[idx].copy()
                    new_pop_idx += 1
            
            # Create new individuals through selection, crossover, and mutation
            while new_pop_idx < self.population_size:
                # Selection
                parent1, parent2 = self._selection()
                
                # Crossover (only if random value is less than crossover probability)
                if np.random.random() < 0.8:  # Using fixed 0.8 as crossover rate
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                # Mutation
                child1 = self._mutation(child1)
                
                # Add first child
                new_population[new_pop_idx] = child1
                new_pop_idx += 1
                
                # Add second child if there's space
                if new_pop_idx < self.population_size:
                    child2 = self._mutation(child2)
                    new_population[new_pop_idx] = child2
                    new_pop_idx += 1
            
            # Replace old population
            self.population = new_population.copy()
            
            # Evaluate fitness
            self._evaluate_fitness()
            
            # Early stopping if no improvement for a while
            if len(self.best_fitness_history) > 20 and generation > 20:
                recent_best = self.best_fitness_history[-20:]
                if all(abs(recent_best[0] - val) < 1e-6 for val in recent_best[1:]):
                    if verbose:
                        tqdm.write(f"Early stopping at generation {generation + 1} due to no improvement")
                    break
            
            # Display progress
            if verbose and (generation + 1) % 10 == 0:
                tqdm.write(f"Generation {generation + 1}/{self.num_generations}, "
                          f"Best Fitness: {self.best_fitness:.4f}, "
                          f"Avg Fitness: {np.mean(self.fitness_scores):.4f}")
        
        return self.best_chromosome, self.best_fitness
    
    def get_history(self):
        """Get the history of fitness values for analysis."""
        return {
            'avg_fitness': self.fitness_history,
            'best_fitness': self.best_fitness_history
        }

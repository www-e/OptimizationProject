"""
Enhanced hyperparameter tuning for neural networks using GA and PSO.
This module provides a more modular and flexible approach to hyperparameter tuning.
"""

import numpy as np
import time
import itertools
import torch
from tqdm import tqdm

from models.neural_network import OptimizableNeuralNetwork
from algorithms.genetic_algorithm import GeneticAlgorithm
from algorithms.particle_swarm import ParticleSwarmOptimization
from experiments.base_experiment import BaseExperiment
from utils.visualization import OptimizationVisualizer


class EnhancedHyperparameterTuning(BaseExperiment):
    """
    Enhanced experiment for hyperparameter tuning using GA and PSO.
    Provides a more modular and flexible approach to hyperparameter tuning.
    """
    
    def __init__(self, X, y, test_size=0.2, validation_size=0.1,
                ga_params=None, pso_params=None, hyperparameter_config=None):
        """
        Initialize the experiment.
        
        Args:
            X: Input features
            y: Target variable
            test_size: Proportion of data for testing
            validation_size: Proportion of training data for validation
            ga_params: Parameters for GA
            pso_params: Parameters for PSO
            hyperparameter_config: Configuration for hyperparameter tuning
        """
        self.X = X
        self.y = y
        self.test_size = test_size
        self.validation_size = validation_size
        
        # Split data
        self._split_data()
        
        # Default hyperparameter configuration
        self.default_config = {
            'tune_hidden_layers': True,
            'tune_learning_rate': True,
            'tune_activation': True,
            'tune_batch_size': True,
            'tune_dropout': True,
            'tune_optimizer': False,
            
            # Default ranges for hyperparameters
            'hidden_layers': [
                [16], [32], [64], [128], 
                [32, 16], [64, 32], [128, 64],
                [64, 32, 16], [128, 64, 32]
            ],
            'learning_rate': [0.0001, 0.001, 0.01, 0.1],
            'activation': ['relu', 'tanh', 'sigmoid', 'elu'],
            'batch_size': [16, 32, 64, 128, 256],
            'dropout_rate': [0.0, 0.1, 0.2, 0.3, 0.5],
            'optimizer': ['adam', 'sgd', 'rmsprop', 'adagrad']
        }
        
        # Update with user-provided configuration
        self.config = self.default_config.copy()
        if hyperparameter_config:
            self.config.update(hyperparameter_config)
        
        # Filter out hyperparameters that are not being tuned
        self.hyperparameter_ranges = {}
        if self.config['tune_hidden_layers']:
            self.hyperparameter_ranges['hidden_layers'] = self.config['hidden_layers']
        
        if self.config['tune_learning_rate']:
            self.hyperparameter_ranges['learning_rate'] = self.config['learning_rate']
        
        if self.config['tune_activation']:
            self.hyperparameter_ranges['activation'] = self.config['activation']
        
        if self.config['tune_batch_size']:
            self.hyperparameter_ranges['batch_size'] = self.config['batch_size']
        
        if self.config['tune_dropout']:
            self.hyperparameter_ranges['dropout_rate'] = self.config['dropout_rate']
        
        if self.config['tune_optimizer']:
            self.hyperparameter_ranges['optimizer'] = self.config['optimizer']
        
        # If no hyperparameters are being tuned, add default ones
        if not self.hyperparameter_ranges:
            self.hyperparameter_ranges = {
                'hidden_layers': self.config['hidden_layers'],
                'learning_rate': self.config['learning_rate']
            }
        
        # Calculate the total number of hyperparameter combinations
        self.total_combinations = 1
        for param_values in self.hyperparameter_ranges.values():
            self.total_combinations *= len(param_values)
        
        # Calculate the chromosome length for GA and dimensions for PSO
        # We'll use a one-hot encoding for categorical parameters
        self.chromosome_length = 0
        for param_values in self.hyperparameter_ranges.values():
            self.chromosome_length += len(param_values)
        
        # Default parameters for GA
        self.ga_params = {
            'population_size': 50,
            'num_generations': 100,
            'mutation_rate': 0.2,
            'mutation_type': 'bit-flip',
            'selection_method': 'tournament',
            'elitism': True,
            'elite_size': 2,
            'chromosome_type': 'binary',
            'crossover_type': 'single_point'
        }
        
        # Default parameters for PSO
        self.pso_params = {
            'num_particles': 50,
            'num_iterations': 100,
            'inertia_weight': 0.7,
            'cognitive_coefficient': 1.5,
            'social_coefficient': 1.5,
            'max_velocity': 0.5,
            'value_range': (0, 1),
            'discrete': True
        }
        
        # Update with user-provided parameters
        if ga_params:
            self.ga_params.update(ga_params)
        
        if pso_params:
            self.pso_params.update(pso_params)
        
        # Results
        self.ga_results = None
        self.pso_results = None
        
        # Visualizer
        self.visualizer = OptimizationVisualizer()
    
    def _decode_chromosome(self, chromosome):
        """
        Decode a chromosome or particle position into hyperparameters.
        
        Args:
            chromosome: Binary chromosome or particle position
        
        Returns:
            Dictionary of hyperparameters
        """
        # Ensure chromosome is binary
        binary_chromosome = (chromosome > 0.5).astype(int)
        
        # Initialize hyperparameters with default values
        hyperparams = {
            'hidden_layers': [64, 32],
            'learning_rate': 0.01,
            'activation': 'relu',
            'batch_size': 32,
            'dropout_rate': 0.0,
            'optimizer': 'adam'
        }
        
        # Current position in the chromosome
        pos = 0
        
        # Decode each hyperparameter that is being tuned
        for param_name, param_values in self.hyperparameter_ranges.items():
            param_length = len(param_values)
            param_genes = binary_chromosome[pos:pos + param_length]
            
            # If no value is selected, choose the first one
            if np.sum(param_genes) == 0:
                param_genes[0] = 1
            
            # If multiple values are selected, choose the one with highest activation
            if np.sum(param_genes) > 1:
                max_idx = np.argmax(param_genes)
                param_genes = np.zeros_like(param_genes)
                param_genes[max_idx] = 1
            
            param_idx = np.argmax(param_genes)
            hyperparams[param_name] = param_values[param_idx]
            
            pos += param_length
        
        return hyperparams
    
    def _fitness_function(self, chromosome):
        """
        Fitness function for hyperparameter tuning.
        
        Args:
            chromosome: Binary chromosome representing hyperparameters
            
        Returns:
            Validation accuracy
        """
        # Decode chromosome to get hyperparameters
        hyperparams = self._decode_chromosome(chromosome)
        
        # Create neural network with these hyperparameters
        nn = OptimizableNeuralNetwork(
            input_dim=self.X_train.shape[1],
            hidden_layers=hyperparams['hidden_layers'],
            output_dim=1 if len(np.unique(self.y)) <= 2 else len(np.unique(self.y)),
            activation=hyperparams['activation'],
            learning_rate=hyperparams['learning_rate']
        )
        
        # Set other hyperparameters
        if 'batch_size' in hyperparams:
            nn.batch_size = hyperparams['batch_size']
        if 'dropout_rate' in hyperparams:
            nn.dropout_rate = hyperparams['dropout_rate']
        if 'optimizer' in hyperparams:
            # Create the correct optimizer object based on the optimizer name
            optimizer_name = hyperparams['optimizer']
            if optimizer_name == 'adam':
                nn.optimizer = torch.optim.Adam(nn.model.parameters(), lr=nn.learning_rate)
            elif optimizer_name == 'sgd':
                nn.optimizer = torch.optim.SGD(nn.model.parameters(), lr=nn.learning_rate)
            elif optimizer_name == 'rmsprop':
                nn.optimizer = torch.optim.RMSprop(nn.model.parameters(), lr=nn.learning_rate)
            elif optimizer_name == 'adagrad':
                nn.optimizer = torch.optim.Adagrad(nn.model.parameters(), lr=nn.learning_rate)
        
        # Use a smaller subset of data for faster evaluation during optimization
        # Take 30% of training data for quick evaluation
        train_size = min(int(len(self.X_train) * 0.3), 1000)  # Cap at 1000 samples
        indices = np.random.choice(len(self.X_train), train_size, replace=False)
        X_train_sample = self.X_train[indices]
        y_train_sample = self.y_train[indices]
        
        # Set epochs for faster evaluation
        nn.epochs = 10
        
        # Train the model
        nn.train(
            X_train_sample, y_train_sample,
            self.X_val, self.y_val,
            verbose=0
        )
        
        # Evaluate on validation set
        val_metrics = nn.evaluate(self.X_val, self.y_val)
        
        # Return validation accuracy as fitness
        return val_metrics['accuracy']
    
    def _handle_training_error(self, hyperparams, error):
        """Handle errors during training with specific hyperparameters."""
        print(f"Training failed with hyperparameters {hyperparams}: {str(error)}")
        return 0.0
    
    def run_ga_optimization(self, verbose=True):
        """
        Run hyperparameter tuning using Genetic Algorithm.
        
        Args:
            verbose: Whether to display progress
            
        Returns:
            Dictionary with results
        """
        if verbose:
            print("Starting GA hyperparameter tuning")
            
        start_time = time.time()
        
        # Initialize GA with configurable population and generations
        ga_params = self.ga_params.copy()
        
        # Use default values if not specified in the configuration
        default_population_size = 30
        default_num_generations = 50  # Increased from 30 to 50 for more thorough optimization
        
        # Set population size (can be overridden in the UI)
        if 'population_size' not in ga_params:
            ga_params['population_size'] = default_population_size
            
        # Set number of generations (can be overridden in the UI)
        if 'num_generations' not in ga_params:
            ga_params['num_generations'] = default_num_generations
        
        # Initialize GA
        try:
            ga = GeneticAlgorithm(
                fitness_function=self._fitness_function,
                chromosome_length=self.chromosome_length,
                **ga_params
            )
            
            # Run optimization
            best_chromosome, best_fitness = ga.evolve(verbose=verbose)
        except Exception as e:
            print(f"Error during GA optimization: {str(e)}")
            # Return default values if optimization fails
            return {
                'best_chromosome': np.zeros(self.chromosome_length),
                'best_fitness': 0.0,
                'best_hyperparameters': self._decode_chromosome(np.zeros(self.chromosome_length)),
                'test_metrics': {'accuracy': 0.0},
                'test_accuracy': 0.0,
                'history': {'avg_fitness': [], 'best_fitness': []},
                'training_time': 0.0,
                'error': str(e)
            }
        
        # Decode the best chromosome
        best_hyperparams = self._decode_chromosome(best_chromosome)
        
        # Create and train neural network with best hyperparameters
        nn = OptimizableNeuralNetwork(
            input_dim=self.X_train.shape[1],
            hidden_layers=best_hyperparams['hidden_layers'],
            output_dim=1 if len(np.unique(self.y)) <= 2 else len(np.unique(self.y)),
            activation=best_hyperparams['activation'],
            learning_rate=best_hyperparams['learning_rate']
        )
        
        # Set other hyperparameters
        if 'batch_size' in best_hyperparams:
            nn.batch_size = best_hyperparams['batch_size']
        if 'dropout_rate' in best_hyperparams:
            nn.dropout_rate = best_hyperparams['dropout_rate']
        if 'optimizer' in best_hyperparams:
            # Create the correct optimizer object based on the optimizer name
            optimizer_name = best_hyperparams['optimizer']
            if optimizer_name == 'adam':
                nn.optimizer = torch.optim.Adam(nn.model.parameters(), lr=nn.learning_rate)
            elif optimizer_name == 'sgd':
                nn.optimizer = torch.optim.SGD(nn.model.parameters(), lr=nn.learning_rate)
            elif optimizer_name == 'rmsprop':
                nn.optimizer = torch.optim.RMSprop(nn.model.parameters(), lr=nn.learning_rate)
            elif optimizer_name == 'adagrad':
                nn.optimizer = torch.optim.Adagrad(nn.model.parameters(), lr=nn.learning_rate)
        
        # Set epochs for final training
        nn.epochs = 20  # Reduced for faster execution
        if 'batch_size' in best_hyperparams:
            nn.batch_size = best_hyperparams.get('batch_size', 32)
        
        # Train the model
        nn.train(
            self.X_train, self.y_train,
            self.X_val, self.y_val,
            verbose=0 if not verbose else 1
        )
        
        # Evaluate on test set
        test_metrics = nn.evaluate(self.X_test, self.y_test)
        
        end_time = time.time()
        training_time = end_time - start_time
        
        # Store results
        self.ga_results = {
            'best_chromosome': best_chromosome,
            'best_fitness': best_fitness,
            'best_hyperparameters': best_hyperparams,
            'test_metrics': test_metrics,
            'test_accuracy': test_metrics['accuracy'],
            'history': ga.get_history(),
            'training_time': training_time,
            'model': nn
        }
        
        return self.ga_results
    
    def run_pso_optimization(self, verbose=True):
        """
        Run hyperparameter tuning using Particle Swarm Optimization.
        
        Args:
            verbose: Whether to display progress
            
        Returns:
            Dictionary with results
        """
        if verbose:
            print("Starting PSO hyperparameter tuning")
            
        start_time = time.time()
        
        # Initialize PSO with configurable particles and iterations
        pso_params = self.pso_params.copy()
        
        # Use default values if not specified in the configuration
        default_num_particles = 30
        default_num_iterations = 50  # Increased from 30 to 50 for more thorough optimization
        
        # Set number of particles (can be overridden in the UI)
        if 'num_particles' not in pso_params:
            pso_params['num_particles'] = default_num_particles
            
        # Set number of iterations (can be overridden in the UI)
        if 'num_iterations' not in pso_params:
            pso_params['num_iterations'] = default_num_iterations
        
        # Initialize PSO
        try:
            pso = ParticleSwarmOptimization(
                fitness_function=self._fitness_function,
                dimensions=self.chromosome_length,
                **pso_params
            )
            
            # Run optimization
            best_position, best_fitness = pso.optimize(verbose=verbose)
        except Exception as e:
            print(f"Error during PSO optimization: {str(e)}")
            # Return default values if optimization fails
            return {
                'best_position': np.zeros(self.chromosome_length),
                'best_fitness': 0.0,
                'best_hyperparameters': self._decode_chromosome(np.zeros(self.chromosome_length)),
                'test_metrics': {'accuracy': 0.0},
                'test_accuracy': 0.0,
                'history': {'avg_fitness': [], 'best_fitness': []},
                'training_time': 0.0,
                'error': str(e)
            }
        
        # Decode the best position
        best_hyperparams = self._decode_chromosome(best_position)
        
        # Create and train neural network with best hyperparameters
        nn = OptimizableNeuralNetwork(
            input_dim=self.X_train.shape[1],
            hidden_layers=best_hyperparams['hidden_layers'],
            output_dim=1 if len(np.unique(self.y)) <= 2 else len(np.unique(self.y)),
            activation=best_hyperparams['activation'],
            learning_rate=best_hyperparams['learning_rate']
        )
        
        # Set other hyperparameters
        if 'batch_size' in best_hyperparams:
            nn.batch_size = best_hyperparams['batch_size']
        if 'dropout_rate' in best_hyperparams:
            nn.dropout_rate = best_hyperparams['dropout_rate']
        if 'optimizer' in best_hyperparams:
            # Create the correct optimizer object based on the optimizer name
            optimizer_name = best_hyperparams['optimizer']
            if optimizer_name == 'adam':
                nn.optimizer = torch.optim.Adam(nn.model.parameters(), lr=nn.learning_rate)
            elif optimizer_name == 'sgd':
                nn.optimizer = torch.optim.SGD(nn.model.parameters(), lr=nn.learning_rate)
            elif optimizer_name == 'rmsprop':
                nn.optimizer = torch.optim.RMSprop(nn.model.parameters(), lr=nn.learning_rate)
            elif optimizer_name == 'adagrad':
                nn.optimizer = torch.optim.Adagrad(nn.model.parameters(), lr=nn.learning_rate)
        
        # Set epochs for final training
        nn.epochs = 20  # Reduced for faster execution
        if 'batch_size' in best_hyperparams:
            nn.batch_size = best_hyperparams.get('batch_size', 32)
        
        # Train the model
        nn.train(
            self.X_train, self.y_train,
            self.X_val, self.y_val,
            verbose=0 if not verbose else 1
        )
        
        # Evaluate on test set
        test_metrics = nn.evaluate(self.X_test, self.y_test)
        
        end_time = time.time()
        training_time = end_time - start_time
        
        # Store results
        self.pso_results = {
            'best_position': best_position,
            'best_fitness': best_fitness,
            'best_hyperparameters': best_hyperparams,
            'test_metrics': test_metrics,
            'test_accuracy': test_metrics['accuracy'],
            'history': pso.get_history(),
            'training_time': training_time,
            'model': nn
        }
        
        return self.pso_results
    
    def compare_algorithms(self):
        """
        Compare GA and PSO performance for hyperparameter tuning.
        
        Returns:
            Comparison results and visualization
        """
        if not self.ga_results or not self.pso_results:
            raise ValueError("Both GA and PSO must be run before comparison")
        
        # Create convergence plot
        # The visualization utility expects each history item to be a dictionary with 'best_fitness' and 'avg_fitness' keys
        convergence_plot = self.visualizer.plot_convergence(
            [self.ga_results['history'], self.pso_results['history']],
            ['Genetic Algorithm', 'Particle Swarm Optimization'],
            title='Hyperparameter Tuning Convergence'
        )
        
        # Create comparison plot
        comparison_data = {
            'GA': {
                'Validation Accuracy': self.ga_results['best_fitness'],
                'Test Accuracy': self.ga_results['test_metrics']['accuracy'],
                'Training Time (s)': self.ga_results['training_time']
            },
            'PSO': {
                'Validation Accuracy': self.pso_results['best_fitness'],
                'Test Accuracy': self.pso_results['test_metrics']['accuracy'],
                'Training Time (s)': self.pso_results['training_time']
            }
        }
        
        comparison_plot = self.visualizer.plot_comparison(
            comparison_data,
            title='Hyperparameter Tuning Comparison'
        )
        
        # Create hyperparameter importance plot
        # We'll focus on the best hyperparameters found by GA and PSO
        ga_hyperparams = self.ga_results['best_hyperparameters']
        pso_hyperparams = self.pso_results['best_hyperparameters']
        
        # Format hyperparameters for visualization
        ga_formatted = {}
        pso_formatted = {}
        
        for key, value in ga_hyperparams.items():
            if key == 'hidden_layers':
                ga_formatted[key] = str(value)
            else:
                ga_formatted[key] = value
        
        for key, value in pso_hyperparams.items():
            if key == 'hidden_layers':
                pso_formatted[key] = str(value)
            else:
                pso_formatted[key] = value
        
        param_importance_plot = self.visualizer.plot_hyperparameter_comparison(
            ga_formatted, pso_formatted,
            title='Best Hyperparameters Comparison'
        )
        
        return {
            'ga_results': self.ga_results,
            'pso_results': self.pso_results,
            'convergence_plot': convergence_plot,
            'comparison_plot': comparison_plot,
            'param_importance_plot': param_importance_plot,
            'best_algorithm': 'GA' if self.ga_results['best_fitness'] > self.pso_results['best_fitness'] else 'PSO',
            'improvement_percentage': abs(self.ga_results['best_fitness'] - self.pso_results['best_fitness']) / 
                                     min(self.ga_results['best_fitness'], self.pso_results['best_fitness']) * 100
        }

"""
Feature selection using GA and PSO algorithms.
"""

import numpy as np
import time
from tqdm import tqdm

from models.neural_network import OptimizableNeuralNetwork
from algorithms.genetic_algorithm import GeneticAlgorithm
from algorithms.particle_swarm import ParticleSwarmOptimization
from experiments.base_experiment import BaseExperiment
from utils.visualization import OptimizationVisualizer


class FeatureSelectionExperiment(BaseExperiment):
    """
    Experiment for feature selection using GA and PSO.
    """
    
    def __init__(self, X, y, feature_names=None, test_size=0.2, validation_size=0.1,
                ga_params=None, pso_params=None, feature_selection_params=None):
        """
        Initialize the experiment.
        
        Args:
            X: Input features
            y: Target variable
            feature_names: Names of features
            test_size: Proportion of data for testing
            validation_size: Proportion of training data for validation
            ga_params: Parameters for GA
            pso_params: Parameters for PSO
            feature_selection_params: Parameters for feature selection
        """
        self.X = X
        self.y = y
        self.feature_names = feature_names or [f'Feature_{i}' for i in range(X.shape[1])]
        self.test_size = test_size
        self.validation_size = validation_size
        
        # Split data
        self._split_data()
        
        # Default parameters for feature selection
        self.feature_selection_params = {
            'max_features': X.shape[1],
            'min_features': 3,
            'method': 'wrapper',
            'scoring': 'accuracy'
        }
        
        # Update with user-provided parameters
        if feature_selection_params:
            self.feature_selection_params.update(feature_selection_params)
        
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
    
    def _split_data(self):
        """Split data into train, validation, and test sets."""
        # Use the parent class implementation
        super()._split_data()
    
    def _fitness_function(self, feature_mask):
        """
        Fitness function for feature selection.
        
        Args:
            feature_mask: Binary mask for feature selection
        
        Returns:
            Validation accuracy
        """
        # Ensure feature mask is binary
        binary_mask = (feature_mask > 0.5).astype(int)
        
        # Count selected features
        num_selected = np.sum(binary_mask)
        
        # If too few or too many features are selected, penalize fitness
        if num_selected < self.feature_selection_params['min_features']:
            return 0.0
        
        if self.feature_selection_params['max_features'] is not None and num_selected > self.feature_selection_params['max_features']:
            return 0.0
        
        # Get indices of selected features
        selected_indices = np.where(binary_mask == 1)[0]
        
        # If no features are selected, return zero fitness
        if len(selected_indices) == 0:
            return 0.0
        
        # Extract selected features
        X_train_selected = self.X_train[:, selected_indices]
        X_val_selected = self.X_val[:, selected_indices]
        
        # Create and train neural network
        nn = OptimizableNeuralNetwork(
            input_dim=len(selected_indices),
            hidden_layers=[32, 16],
            output_dim=1 if len(np.unique(self.y)) <= 2 else len(np.unique(self.y))
        )
        
        # Train the model
        nn.train(X_train_selected, self.y_train, X_val_selected, self.y_val, verbose=0)
        
        # Evaluate on validation set
        predictions = nn.predict(X_val_selected)
        
        # For binary classification
        if nn.output_dim == 1:
            accuracy = np.mean((predictions > 0.5).astype(int) == self.y_val)
        else:
            # For multi-class classification
            pred_classes = np.argmax(predictions, axis=1)
            true_classes = np.argmax(self.y_val, axis=1) if len(self.y_val.shape) > 1 else self.y_val
            accuracy = np.mean(pred_classes == true_classes)
        
        # Add a small penalty based on the number of features
        # This encourages using fewer features when accuracy is similar
        feature_penalty = 0.001 * (num_selected / self.X.shape[1])
        
        return accuracy - feature_penalty
    
    def run_ga_feature_selection(self, verbose=True):
        """
        Run feature selection using Genetic Algorithm.
        
        Args:
            verbose: Whether to display progress
        
        Returns:
            Dictionary with results
        """
        start_time = time.time()
        
        # Initialize GA
        ga = GeneticAlgorithm(
            fitness_function=self._fitness_function,
            chromosome_length=self.X.shape[1],
            **self.ga_params
        )
        
        # Run optimization
        best_mask, best_fitness = ga.evolve(verbose=verbose)
        
        # Ensure mask is binary
        best_mask_binary = (best_mask > 0.5).astype(int)
        
        # Get indices of selected features
        selected_indices = np.where(best_mask_binary == 1)[0]
        selected_feature_names = [self.feature_names[i] for i in selected_indices]
        
        # Extract selected features
        X_train_selected = self.X_train[:, selected_indices]
        X_val_selected = self.X_val[:, selected_indices]
        X_test_selected = self.X_test[:, selected_indices]
        
        # Create and train neural network with selected features
        nn = OptimizableNeuralNetwork(
            input_dim=len(selected_indices),
            hidden_layers=[32, 16],
            output_dim=1 if len(np.unique(self.y)) <= 2 else len(np.unique(self.y))
        )
        
        # Train the model
        nn.train(X_train_selected, self.y_train, X_val_selected, self.y_val, verbose=0)
        
        # Evaluate on test set
        test_metrics = nn.evaluate(X_test_selected, self.y_test)
        
        end_time = time.time()
        training_time = end_time - start_time
        
        # Store results
        self.ga_results = {
            'best_mask': best_mask_binary,
            'best_fitness': best_fitness,
            'selected_indices': selected_indices,
            'selected_feature_names': selected_feature_names,
            'num_selected_features': len(selected_indices),
            'test_metrics': test_metrics,
            'history': ga.get_history(),
            'training_time': training_time
        }
        
        return self.ga_results
    
    def run_pso_feature_selection(self, verbose=True):
        """
        Run feature selection using Particle Swarm Optimization.
        
        Args:
            verbose: Whether to display progress
        
        Returns:
            Dictionary with results
        """
        start_time = time.time()
        
        # Initialize PSO
        pso = ParticleSwarmOptimization(
            fitness_function=self._fitness_function,
            dimensions=self.X.shape[1],
            **self.pso_params
        )
        
        # Run optimization
        best_mask, best_fitness = pso.optimize(verbose=verbose)
        
        # Ensure mask is binary
        best_mask_binary = (best_mask > 0.5).astype(int)
        
        # Get indices of selected features
        selected_indices = np.where(best_mask_binary == 1)[0]
        selected_feature_names = [self.feature_names[i] for i in selected_indices]
        
        # Extract selected features
        X_train_selected = self.X_train[:, selected_indices]
        X_val_selected = self.X_val[:, selected_indices]
        X_test_selected = self.X_test[:, selected_indices]
        
        # Create and train neural network with selected features
        nn = OptimizableNeuralNetwork(
            input_dim=len(selected_indices),
            hidden_layers=[32, 16],
            output_dim=1 if len(np.unique(self.y)) <= 2 else len(np.unique(self.y))
        )
        
        # Train the model
        nn.train(X_train_selected, self.y_train, X_val_selected, self.y_val, verbose=0)
        
        # Evaluate on test set
        test_metrics = nn.evaluate(X_test_selected, self.y_test)
        
        end_time = time.time()
        training_time = end_time - start_time
        
        # Store results
        self.pso_results = {
            'best_mask': best_mask_binary,
            'best_fitness': best_fitness,
            'selected_indices': selected_indices,
            'selected_feature_names': selected_feature_names,
            'num_selected_features': len(selected_indices),
            'test_metrics': test_metrics,
            'history': pso.get_history(),
            'training_time': training_time
        }
        
        return self.pso_results
    
    def compare_algorithms(self):
        """
        Compare GA and PSO performance for feature selection.
        
        Returns:
            Comparison results and visualization
        """
        if not self.ga_results or not self.pso_results:
            raise ValueError("Both GA and PSO must be run before comparison")
        
        # Create convergence plot
        convergence_plot = self.visualizer.plot_convergence(
            [self.ga_results['history'], self.pso_results['history']],
            ['Genetic Algorithm', 'Particle Swarm Optimization'],
            title='Feature Selection Convergence'
        )
        
        # Compare feature importance
        ga_feature_importance = self.ga_results['best_mask']
        pso_feature_importance = self.pso_results['best_mask']
        
        # Plot feature importance for GA
        ga_feature_importance_plot = self.visualizer.plot_feature_importance(
            ga_feature_importance, self.feature_names, 
            title='Feature Importance (Genetic Algorithm)'
        )
        
        # Plot feature importance for PSO
        pso_feature_importance_plot = self.visualizer.plot_feature_importance(
            pso_feature_importance, self.feature_names, 
            title='Feature Importance (Particle Swarm Optimization)'
        )
        
        # Create comparison plot
        comparison_data = {
            'GA': {
                'Validation Accuracy': self.ga_results['best_fitness'],
                'Test Accuracy': self.ga_results['test_metrics']['accuracy'],
                'Training Time (s)': self.ga_results['training_time'],
                'Features Selected': self.ga_results['num_selected_features']
            },
            'PSO': {
                'Validation Accuracy': self.pso_results['best_fitness'],
                'Test Accuracy': self.pso_results['test_metrics']['accuracy'],
                'Training Time (s)': self.pso_results['training_time'],
                'Features Selected': self.pso_results['num_selected_features']
            }
        }
        
        comparison_plot = self.visualizer.plot_comparison(
            comparison_data,
            title='Feature Selection Comparison'
        )
        
        return {
            'ga_results': self.ga_results,
            'pso_results': self.pso_results,
            'convergence_plot': convergence_plot,
            'comparison_plot': comparison_plot,
            'ga_feature_importance_plot': ga_feature_importance_plot,
            'pso_feature_importance_plot': pso_feature_importance_plot,
            'common_features': set(self.ga_results['selected_feature_names']).intersection(
                set(self.pso_results['selected_feature_names'])
            )
        }
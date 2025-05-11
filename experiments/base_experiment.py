"""
Base experiment class for all optimization experiments.
"""

import numpy as np
import time
from sklearn.model_selection import train_test_split
from utils.visualization import OptimizationVisualizer

class BaseExperiment:
    """
    Base class for all optimization experiments.
    Provides common functionality for data splitting, result tracking, and visualization.
    """
    
    def __init__(self, X, y, test_size=0.2, validation_size=0.1, 
                 ga_params=None, pso_params=None):
        """
        Initialize the base experiment.
        
        Args:
            X: Input features
            y: Target variable
            test_size: Proportion of data for testing
            validation_size: Proportion of training data for validation
            ga_params: Parameters for GA
            pso_params: Parameters for PSO
        """
        self.X = X
        self.y = y
        self.test_size = test_size
        self.validation_size = validation_size
        
        # Split data
        self._split_data()
        
        # Default parameters for GA
        self.ga_params = {
            'population_size': 50,
            'num_generations': 100,
            'crossover_rate': 0.8,
            'mutation_rate': 0.2,
            'selection_method': 'tournament',
            'tournament_size': 3,
            'elitism': True,
            'elite_size': 2,
            'chromosome_type': 'binary',
            'value_range': (0, 1)
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
        # First split: training + validation and test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=42
        )
        
        # Second split: training and validation
        val_size_adjusted = self.validation_size / (1 - self.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size_adjusted, random_state=42
        )
        
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
    
    def compare_algorithms(self):
        """
        Compare GA and PSO performance.
        
        Returns:
            Comparison results and visualization
        """
        if self.ga_results is None or self.pso_results is None:
            raise ValueError("Both GA and PSO must be run before comparison")
        
        # Compare convergence
        histories = [self.ga_results['history'], self.pso_results['history']]
        labels = ['Genetic Algorithm', 'Particle Swarm Optimization']
        
        convergence_plot = self.visualizer.plot_convergence(
            histories, labels, title='Optimization Convergence'
        )
        
        # Compare metrics
        ga_metrics = [
            self.ga_results['test_metrics']['accuracy'],
            self.ga_results['test_metrics']['precision'],
            self.ga_results['test_metrics']['recall'],
            self.ga_results['test_metrics']['f1'],
            self.ga_results['training_time']
        ]
        
        pso_metrics = [
            self.pso_results['test_metrics']['accuracy'],
            self.pso_results['test_metrics']['precision'],
            self.pso_results['test_metrics']['recall'],
            self.pso_results['test_metrics']['f1'],
            self.pso_results['training_time']
        ]
        
        metrics = np.array([ga_metrics, pso_metrics])
        
        comparison_plot = self.visualizer.plot_algorithm_comparison(
            metrics, labels, 
            metric_names=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Training Time (s)']
        )
        
        return {
            'ga_results': self.ga_results,
            'pso_results': self.pso_results,
            'convergence_plot': convergence_plot,
            'comparison_plot': comparison_plot
        }

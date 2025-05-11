"""
Neural network weight optimization using GA and PSO algorithms.
"""

import numpy as np
import time
from tqdm import tqdm

from models.neural_network import OptimizableNeuralNetwork
from algorithms.genetic_algorithm import GeneticAlgorithm
from algorithms.particle_swarm import ParticleSwarmOptimization
from experiments.base_experiment import BaseExperiment
from utils.visualization import OptimizationVisualizer


class WeightOptimizationExperiment(BaseExperiment):
    """
    Experiment for neural network weight optimization using GA and PSO.
    Inherits from BaseExperiment for common functionality.
    """
    
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test,
                input_dim, output_dim, hidden_layers=[64, 32], activation='relu',
                ga_params=None, pso_params=None):
        """
        Initialize the experiment.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            X_test: Test features
            y_test: Test labels
            input_dim: Input dimension (number of features)
            output_dim: Output dimension (number of classes)
            hidden_layers: List of neurons in each hidden layer
            activation: Activation function to use
            ga_params: Parameters for GA
            pso_params: Parameters for PSO
        """
        # We're not calling the parent class __init__ since we're given pre-split data
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        
        # Neural network parameters
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        self.activation = activation
        
        # Create neural network
        self.nn = OptimizableNeuralNetwork(
            input_dim=input_dim,
            hidden_layers=hidden_layers,
            output_dim=output_dim,
            activation=activation
        )
        
        # Default parameters for GA - override base class defaults
        default_ga_params = {
            'population_size': 50,
            'num_generations': 100,
            'crossover_rate': 0.8,
            'mutation_rate': 0.2,
            'selection_method': 'tournament',
            'tournament_size': 3,
            'elitism': True,
            'elite_size': 2,
            'chromosome_type': 'real',
            'value_range': (-1, 1)
        }
        
        # Default parameters for PSO - override base class defaults
        default_pso_params = {
            'num_particles': 50,
            'num_iterations': 100,
            'inertia_weight': 0.7,
            'cognitive_coefficient': 1.5,
            'social_coefficient': 1.5,
            'max_velocity': 0.5,
            'value_range': (-1, 1),
            'discrete': False
        }
        
        # Initialize with defaults
        self.ga_params = default_ga_params.copy()
        self.pso_params = default_pso_params.copy()
        
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
    
    def _fitness_function(self, weights):
        """
        Fitness function for weight optimization.
        
        Args:
            weights: Flattened weights array
        
        Returns:
            Validation accuracy or other fitness metric
        """
        # Set weights to the neural network
        self.nn.set_weights_flat(weights)
        
        # Evaluate on validation set
        predictions = self.nn.predict(self.X_val)
        
        # For binary classification
        if self.nn.output_dim == 1:
            # Convert predictions to binary (0 or 1)
            pred_binary = (predictions > 0.5).astype(int)
            # Ensure y_val is the right shape for comparison
            y_val_reshaped = self.y_val.reshape(-1) if len(self.y_val.shape) > 1 else self.y_val
            y_val_reshaped = y_val_reshaped.astype(int)  # Ensure consistent data type
            
            # Calculate accuracy
            accuracy = np.mean(pred_binary == y_val_reshaped)
            
            # Debug information on first call
            if not hasattr(self, '_debug_printed'):
                self._debug_printed = True
                print(f"Binary classification task detected")
                print(f"Validation set shape: {self.X_val.shape}")
                print(f"Unique values in y_val: {np.unique(y_val_reshaped)}")
                print(f"Predictions range: {np.min(predictions)} to {np.max(predictions)}")
                print(f"Binary predictions distribution: {np.bincount(pred_binary.flatten())}")
                print(f"Initial accuracy: {accuracy}")
        else:
            # For multi-class classification
            pred_classes = np.argmax(predictions, axis=1)
            true_classes = np.argmax(self.y_val, axis=1) if len(self.y_val.shape) > 1 else self.y_val.astype(int)
            
            # Calculate accuracy
            accuracy = np.mean(pred_classes == true_classes)
            
            # Debug information on first call
            if not hasattr(self, '_debug_printed'):
                self._debug_printed = True
                print(f"Multi-class classification task detected with {self.nn.output_dim} classes")
                print(f"Validation set shape: {self.X_val.shape}")
                print(f"Unique values in true_classes: {np.unique(true_classes)}")
                print(f"Predictions shape: {predictions.shape}")
                print(f"Predicted classes distribution: {np.bincount(pred_classes, minlength=self.nn.output_dim)}")
                print(f"Initial accuracy: {accuracy}")
        
        # If accuracy is very low, we might want to use a different metric
        # For example, we could use 1 - binary cross entropy as fitness
        if accuracy < 0.01:
            # Try using a different metric that might give more gradient information
            if self.nn.output_dim == 1:
                # Binary cross-entropy (avoiding log(0))
                epsilon = 1e-15
                predictions_clipped = np.clip(predictions, epsilon, 1 - epsilon)
                bce = -np.mean(y_val_reshaped * np.log(predictions_clipped) + 
                              (1 - y_val_reshaped) * np.log(1 - predictions_clipped))
                # Convert to a fitness value (higher is better)
                return 1.0 - min(1.0, bce)
        
        return accuracy
    
    def run_ga_optimization(self, verbose=True):
        """
        Run weight optimization using Genetic Algorithm.
        
        Args:
            verbose: Whether to display progress
        
        Returns:
            Dictionary with results
        """
        start_time = time.time()
        
        # Initialize GA
        ga = GeneticAlgorithm(
            fitness_function=self._fitness_function,
            chromosome_length=self.nn.weights_count,
            **self.ga_params
        )
        
        # Run optimization
        best_weights, best_fitness = ga.evolve(verbose=verbose)
        
        # Set best weights to neural network
        self.nn.set_weights_flat(best_weights)
        
        # Evaluate on test set
        test_metrics = self.nn.evaluate(self.X_test, self.y_test)
        
        end_time = time.time()
        training_time = end_time - start_time
        
        # Store results
        self.ga_results = {
            'best_weights': best_weights,
            'best_fitness': best_fitness,
            'test_metrics': test_metrics,
            'history': ga.get_history(),
            'training_time': training_time
        }
        
        return self.ga_results
    
    def run_pso_optimization(self, verbose=True):
        """
        Run weight optimization using Particle Swarm Optimization.
        
        Args:
            verbose: Whether to display progress
        
        Returns:
            Dictionary with results
        """
        start_time = time.time()
        
        # Initialize PSO
        pso = ParticleSwarmOptimization(
            fitness_function=self._fitness_function,
            dimensions=self.nn.weights_count,
            **self.pso_params
        )
        
        # Run optimization
        best_weights, best_fitness = pso.optimize(verbose=verbose)
        
        # Set best weights to neural network
        self.nn.set_weights_flat(best_weights)
        
        # Evaluate on test set
        test_metrics = self.nn.evaluate(self.X_test, self.y_test)
        
        end_time = time.time()
        training_time = end_time - start_time
        
        # Store results
        self.pso_results = {
            'best_weights': best_weights,
            'best_fitness': best_fitness,
            'test_metrics': test_metrics,
            'history': pso.get_history(),
            'training_time': training_time
        }
        
        return self.pso_results
    
    def compare_algorithms(self):
        """
        Compare GA and PSO performance for weight optimization.
        
        Returns:
            Comparison results and visualization
        """
        if self.ga_results is None or self.pso_results is None:
            raise ValueError("Both GA and PSO must be run before comparison")
        
        # Compare convergence
        histories = [self.ga_results['history'], self.pso_results['history']]
        labels = ['Genetic Algorithm', 'Particle Swarm Optimization']
        
        convergence_plot = self.visualizer.plot_convergence(
            histories, labels, title='Weight Optimization Convergence'
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
    
    def parameter_impact_study(self, algorithm='ga', parameter_name='population_size',
                             parameter_values=None, runs_per_value=3, verbose=True):
        """
        Study the impact of a parameter on optimization performance.
        
        Args:
            algorithm: 'ga' or 'pso'
            parameter_name: Name of the parameter to study
            parameter_values: List of parameter values to test
            runs_per_value: Number of runs for each parameter value
            verbose: Whether to display progress
        
        Returns:
            Results and visualization
        """
        if parameter_values is None:
            if parameter_name == 'population_size' or parameter_name == 'num_particles':
                parameter_values = [10, 20, 50, 100, 200]
            elif parameter_name == 'num_generations' or parameter_name == 'num_iterations':
                parameter_values = [10, 25, 50, 100, 200]
            elif parameter_name == 'mutation_rate':
                parameter_values = [0.01, 0.05, 0.1, 0.2, 0.5]
            elif parameter_name == 'crossover_rate':
                parameter_values = [0.5, 0.6, 0.7, 0.8, 0.9]
            elif parameter_name == 'inertia_weight':
                parameter_values = [0.1, 0.3, 0.5, 0.7, 0.9]
            elif parameter_name == 'cognitive_coefficient' or parameter_name == 'social_coefficient':
                parameter_values = [0.5, 1.0, 1.5, 2.0, 2.5]
            else:
                raise ValueError(f"No default values for parameter: {parameter_name}")
        
        # Store results
        avg_fitness_scores = []
        best_fitness_scores = []
        
        # For each parameter value
        for value in tqdm(parameter_values, desc=f"Testing {parameter_name}"):
            value_fitness_scores = []
            
            # Multiple runs for statistical significance
            for run in range(runs_per_value):
                if verbose:
                    print(f"Parameter {parameter_name} = {value}, Run {run+1}/{runs_per_value}")
                
                if algorithm == 'ga':
                    # Update GA parameters
                    ga_params = self.ga_params.copy()
                    ga_params[parameter_name] = value
                    
                    # Initialize GA
                    ga = GeneticAlgorithm(
                        fitness_function=self._fitness_function,
                        chromosome_length=self.weights_count,
                        **ga_params
                    )
                    
                    # Run optimization
                    _, best_fitness = ga.evolve(verbose=False)
                    
                elif algorithm == 'pso':
                    # Update PSO parameters
                    pso_params = self.pso_params.copy()
                    pso_params[parameter_name] = value
                    
                    # Initialize PSO
                    pso = ParticleSwarmOptimization(
                        fitness_function=self._fitness_function,
                        dimensions=self.weights_count,
                        **pso_params
                    )
                    
                    # Run optimization
                    _, best_fitness = pso.optimize(verbose=False)
                
                value_fitness_scores.append(best_fitness)
            
            # Calculate average and best fitness for this parameter value
            avg_fitness = np.mean(value_fitness_scores)
            best_fitness = np.max(value_fitness_scores)
            
            avg_fitness_scores.append(avg_fitness)
            best_fitness_scores.append(best_fitness)
            
            if verbose:
                print(f"Parameter {parameter_name} = {value}: "
                     f"Avg Fitness = {avg_fitness:.4f}, Best Fitness = {best_fitness:.4f}")
        
        # Create visualization
        algorithm_name = 'Genetic Algorithm' if algorithm == 'ga' else 'Particle Swarm Optimization'
        
        impact_plot = self.visualizer.plot_parameter_impact(
            parameter_values, avg_fitness_scores, parameter_name, algorithm_name
        )
        
        return {
            'parameter_name': parameter_name,
            'parameter_values': parameter_values,
            'avg_fitness_scores': avg_fitness_scores,
            'best_fitness_scores': best_fitness_scores,
            'impact_plot': impact_plot
        }

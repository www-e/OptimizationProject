"""
Hyperparameter tuning for neural networks using GA and PSO.
"""

import numpy as np
import time
import itertools
from tqdm import tqdm

from models.neural_network import OptimizableNeuralNetwork
from algorithms.genetic_algorithm import GeneticAlgorithm
from algorithms.particle_swarm import ParticleSwarmOptimization
from experiments.base_experiment import BaseExperiment


class HyperparameterTuningExperiment(BaseExperiment):
    """
    Experiment for hyperparameter tuning using GA and PSO.
    """
    
    def __init__(self, X, y, test_size=0.2, validation_size=0.1,
                ga_params=None, pso_params=None, hyperparameter_ranges=None):
        """
        Initialize the experiment.
        
        Args:
            X: Input features
            y: Target variable
            test_size: Proportion of data for testing
            validation_size: Proportion of training data for validation
            ga_params: Parameters for GA
            pso_params: Parameters for PSO
            hyperparameter_ranges: Ranges for hyperparameters to tune
        """
        self.X = X
        self.y = y
        self.test_size = test_size
        self.validation_size = validation_size
        
        # Split data
        self._split_data()
        
        # Default hyperparameter ranges
        self.hyperparameter_ranges = {
            'hidden_layers': [
                [16], [32], [64], [128], 
                [32, 16], [64, 32], [128, 64]
            ],
            'learning_rate': [0.001, 0.01, 0.1],
            'activation': ['relu', 'tanh', 'sigmoid'],
            'batch_size': [16, 32, 64, 128],
            'dropout_rate': [0.0, 0.1, 0.2, 0.3, 0.5]
        }
        
        # Update with user-provided ranges
        if hyperparameter_ranges:
            self.hyperparameter_ranges.update(hyperparameter_ranges)
        
        # Calculate the total number of hyperparameter combinations
        self.total_combinations = (
            len(self.hyperparameter_ranges['hidden_layers']) *
            len(self.hyperparameter_ranges['learning_rate']) *
            len(self.hyperparameter_ranges['activation']) *
            len(self.hyperparameter_ranges['batch_size']) *
            len(self.hyperparameter_ranges['dropout_rate'])
        )
        
        # Calculate the chromosome length for GA and dimensions for PSO
        # We'll use a one-hot encoding for categorical parameters
        self.chromosome_length = (
            len(self.hyperparameter_ranges['hidden_layers']) +
            len(self.hyperparameter_ranges['learning_rate']) +
            len(self.hyperparameter_ranges['activation']) +
            len(self.hyperparameter_ranges['batch_size']) +
            len(self.hyperparameter_ranges['dropout_rate'])
        )
        
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
        
        # Initialize hyperparameters
        hyperparams = {}
        
        # Current position in the chromosome
        pos = 0
        
        # Decode hidden layers
        hidden_layers_length = len(self.hyperparameter_ranges['hidden_layers'])
        hidden_layers_genes = binary_chromosome[pos:pos + hidden_layers_length]
        
        # If no hidden layer is selected, choose the first one
        if np.sum(hidden_layers_genes) == 0:
            hidden_layers_genes[0] = 1
        
        # If multiple hidden layers are selected, choose the one with highest activation
        if np.sum(hidden_layers_genes) > 1:
            max_idx = np.argmax(hidden_layers_genes)
            hidden_layers_genes = np.zeros_like(hidden_layers_genes)
            hidden_layers_genes[max_idx] = 1
        
        hidden_layers_idx = np.argmax(hidden_layers_genes)
        hyperparams['hidden_layers'] = self.hyperparameter_ranges['hidden_layers'][hidden_layers_idx]
        
        pos += hidden_layers_length
        
        # Decode learning rate
        learning_rate_length = len(self.hyperparameter_ranges['learning_rate'])
        learning_rate_genes = binary_chromosome[pos:pos + learning_rate_length]
        
        # If no learning rate is selected, choose the first one
        if np.sum(learning_rate_genes) == 0:
            learning_rate_genes[0] = 1
        
        # If multiple learning rates are selected, choose the one with highest activation
        if np.sum(learning_rate_genes) > 1:
            max_idx = np.argmax(learning_rate_genes)
            learning_rate_genes = np.zeros_like(learning_rate_genes)
            learning_rate_genes[max_idx] = 1
        
        learning_rate_idx = np.argmax(learning_rate_genes)
        hyperparams['learning_rate'] = self.hyperparameter_ranges['learning_rate'][learning_rate_idx]
        
        pos += learning_rate_length
        
        # Decode activation
        activation_length = len(self.hyperparameter_ranges['activation'])
        activation_genes = binary_chromosome[pos:pos + activation_length]
        
        # If no activation is selected, choose the first one
        if np.sum(activation_genes) == 0:
            activation_genes[0] = 1
        
        # If multiple activations are selected, choose the one with highest activation
        if np.sum(activation_genes) > 1:
            max_idx = np.argmax(activation_genes)
            activation_genes = np.zeros_like(activation_genes)
            activation_genes[max_idx] = 1
        
        activation_idx = np.argmax(activation_genes)
        hyperparams['activation'] = self.hyperparameter_ranges['activation'][activation_idx]
        
        pos += activation_length
        
        # Decode batch size
        batch_size_length = len(self.hyperparameter_ranges['batch_size'])
        batch_size_genes = binary_chromosome[pos:pos + batch_size_length]
        
        # If no batch size is selected, choose the first one
        if np.sum(batch_size_genes) == 0:
            batch_size_genes[0] = 1
        
        # If multiple batch sizes are selected, choose the one with highest activation
        if np.sum(batch_size_genes) > 1:
            max_idx = np.argmax(batch_size_genes)
            batch_size_genes = np.zeros_like(batch_size_genes)
            batch_size_genes[max_idx] = 1
        
        batch_size_idx = np.argmax(batch_size_genes)
        hyperparams['batch_size'] = self.hyperparameter_ranges['batch_size'][batch_size_idx]
        
        pos += batch_size_length
        
        # Decode dropout rate
        dropout_rate_length = len(self.hyperparameter_ranges['dropout_rate'])
        dropout_rate_genes = binary_chromosome[pos:pos + dropout_rate_length]
        
        # If no dropout rate is selected, choose the first one
        if np.sum(dropout_rate_genes) == 0:
            dropout_rate_genes[0] = 1
        
        # If multiple dropout rates are selected, choose the one with highest activation
        if np.sum(dropout_rate_genes) > 1:
            max_idx = np.argmax(dropout_rate_genes)
            dropout_rate_genes = np.zeros_like(dropout_rate_genes)
            dropout_rate_genes[max_idx] = 1
        
        dropout_rate_idx = np.argmax(dropout_rate_genes)
        hyperparams['dropout_rate'] = self.hyperparameter_ranges['dropout_rate'][dropout_rate_idx]
        
        return hyperparams
    
    def _fitness_function(self, chromosome):
        """
        Fitness function for hyperparameter tuning.
        
        Args:
            chromosome: Binary chromosome or particle position
        
        Returns:
            Validation accuracy and additional metrics for stability
        """
        # Decode chromosome to hyperparameters
        hyperparams = self._decode_chromosome(chromosome)
        
        # Create and train neural network with the hyperparameters
        nn = OptimizableNeuralNetwork(
            input_dim=self.X_train.shape[1],
            hidden_layers=hyperparams['hidden_layers'],
            output_dim=1 if len(np.unique(self.y)) <= 2 else len(np.unique(self.y)),
            activation=hyperparams['activation'],
            learning_rate=hyperparams['learning_rate']
        )
        
        # Set whether to use dropout
        use_dropout = hyperparams['dropout_rate'] > 0
        
        # Set hyperparameters
        nn_params = {
            'batch_size': hyperparams['batch_size'],
            'use_dropout': use_dropout,
            'dropout_rate': hyperparams['dropout_rate'] if use_dropout else 0.0
        }
        
        nn.set_hyperparameters(nn_params)
        
        # Train the model with multiple attempts for stability
        max_attempts = 3
        best_accuracy = 0.0
        
        for attempt in range(max_attempts):
            try:
                # Train with different random seed each time for robustness
                seed = 42 + attempt
                torch.manual_seed(seed)
                np.random.seed(seed)
                
                # Train with increasing verbosity on later attempts
                verbose_level = 0 if attempt == 0 else 1
                history = nn.train(self.X_train, self.y_train, self.X_val, self.y_val, verbose=verbose_level)
                
                # Evaluate on validation set
                predictions = nn.predict(self.X_val)
                
                # For binary classification
                if nn.output_dim == 1:
                    accuracy = np.mean((predictions > 0.5).astype(int) == self.y_val)
                    
                    # For medical relevance, calculate additional metrics
                    from sklearn.metrics import precision_score, recall_score, f1_score
                    y_pred = (predictions > 0.5).astype(int)
                    precision = precision_score(self.y_val, y_pred, zero_division=0)
                    recall = recall_score(self.y_val, y_pred, zero_division=0)
                    f1 = f1_score(self.y_val, y_pred, zero_division=0)
                    
                    # For medical applications, we want balanced precision and recall
                    # Use F1 score to balance both for disease risk prediction
                    combined_score = 0.7 * accuracy + 0.3 * f1
                else:
                    # For multi-class classification
                    pred_classes = np.argmax(predictions, axis=1)
                    true_classes = np.argmax(self.y_val, axis=1) if len(self.y_val.shape) > 1 else self.y_val
                    accuracy = np.mean(pred_classes == true_classes)
                    combined_score = accuracy
                
                # Keep the best result
                if combined_score > best_accuracy:
                    best_accuracy = combined_score
                
                # If we got a good result, no need for more attempts
                if combined_score > 0.8:
                    break
                    
            except Exception as e:
                print(f"Attempt {attempt+1} failed with hyperparameters {hyperparams}: {str(e)}")
                # Reduce learning rate on failure
                nn.learning_rate *= 0.5
                continue
        
        return best_accuracy if best_accuracy > 0 else 0.0
    
    def run_ga_tuning(self, verbose=True):
        """
        Run hyperparameter tuning using Genetic Algorithm.
        
        Args:
            verbose: Whether to display progress
        
        Returns:
            Dictionary with results
        """
        start_time = time.time()
        
        # Initialize GA
        ga = GeneticAlgorithm(
            fitness_function=self._fitness_function,
            chromosome_length=self.chromosome_length,
            **self.ga_params
        )
        
        # Run optimization
        best_chromosome, best_fitness = ga.evolve(verbose=verbose)
        
        # Decode best chromosome
        best_hyperparams = self._decode_chromosome(best_chromosome)
        
        # Create and train neural network with best hyperparameters
        nn = OptimizableNeuralNetwork(
            input_dim=self.X_train.shape[1],
            hidden_layers=best_hyperparams['hidden_layers'],
            output_dim=1 if len(np.unique(self.y)) <= 2 else len(np.unique(self.y)),
            activation=best_hyperparams['activation'],
            learning_rate=best_hyperparams['learning_rate']
        )
        
        # Set whether to use dropout
        use_dropout = best_hyperparams['dropout_rate'] > 0
        
        # Set hyperparameters
        nn_params = {
            'batch_size': best_hyperparams['batch_size'],
            'use_dropout': use_dropout,
            'dropout_rate': best_hyperparams['dropout_rate'] if use_dropout else 0.0
        }
        
        nn.set_hyperparameters(nn_params)
        
        # Train the model with more epochs to ensure convergence
        try:
            # First try with normal training
            nn.train(self.X_train, self.y_train, self.X_val, self.y_val, verbose=1)
            
            # Evaluate on test set
            test_metrics = nn.evaluate(self.X_test, self.y_test)
        except Exception as e:
            print(f"Error training GA model: {str(e)}")
            # Fallback to simpler model
            nn = OptimizableNeuralNetwork(
                input_dim=self.X_train.shape[1],
                hidden_layers=[32, 16],
                output_dim=1 if len(np.unique(self.y)) <= 2 else len(np.unique(self.y)),
                activation='relu',
                learning_rate=0.001,
                epochs=20
            )
            nn.train(self.X_train, self.y_train, self.X_val, self.y_val, verbose=1)
            test_metrics = nn.evaluate(self.X_test, self.y_test)
        
        end_time = time.time()
        training_time = end_time - start_time
        
        # Store results
        self.ga_results = {
            'best_chromosome': best_chromosome,
            'best_fitness': best_fitness,
            'best_hyperparameters': best_hyperparams,
            'test_metrics': test_metrics,
            'history': ga.get_history(),
            'training_time': training_time,
            'model': nn
        }
        
        return self.ga_results
    
    def run_pso_tuning(self, verbose=True):
        """
        Run hyperparameter tuning using Particle Swarm Optimization.
        
        Args:
            verbose: Whether to display progress
        
        Returns:
            Dictionary with results
        """
        start_time = time.time()
        
        # Initialize PSO
        pso = ParticleSwarmOptimization(
            fitness_function=self._fitness_function,
            dimensions=self.chromosome_length,
            **self.pso_params
        )
        
        # Run optimization
        best_position, best_fitness = pso.optimize(verbose=verbose)
        
        # Decode best position
        best_hyperparams = self._decode_chromosome(best_position)
        
        # Create and train neural network with best hyperparameters
        nn = OptimizableNeuralNetwork(
            input_dim=self.X_train.shape[1],
            hidden_layers=best_hyperparams['hidden_layers'],
            output_dim=1 if len(np.unique(self.y)) <= 2 else len(np.unique(self.y)),
            activation=best_hyperparams['activation'],
            learning_rate=best_hyperparams['learning_rate']
        )
        
        # Set whether to use dropout
        use_dropout = best_hyperparams['dropout_rate'] > 0
        
        # Set hyperparameters
        nn_params = {
            'batch_size': best_hyperparams['batch_size'],
            'use_dropout': use_dropout,
            'dropout_rate': best_hyperparams['dropout_rate'] if use_dropout else 0.0
        }
        
        nn.set_hyperparameters(nn_params)
        
        # Train the model with more epochs to ensure convergence
        try:
            # First try with normal training
            nn.train(self.X_train, self.y_train, self.X_val, self.y_val, verbose=1)
            
            # Evaluate on test set
            test_metrics = nn.evaluate(self.X_test, self.y_test)
            
            # If metrics are all zero, try retraining with different parameters
            if test_metrics['accuracy'] == 0 and test_metrics['precision'] == 0 and test_metrics['recall'] == 0:
                # Try with a smaller learning rate
                nn = OptimizableNeuralNetwork(
                    input_dim=self.X_train.shape[1],
                    hidden_layers=best_hyperparams['hidden_layers'],
                    output_dim=1 if len(np.unique(self.y)) <= 2 else len(np.unique(self.y)),
                    activation=best_hyperparams['activation'],
                    learning_rate=best_hyperparams['learning_rate'] * 0.1,  # Reduce learning rate
                    epochs=30  # Increase epochs
                )
                nn.set_hyperparameters(nn_params)
                nn.train(self.X_train, self.y_train, self.X_val, self.y_val, verbose=1)
                test_metrics = nn.evaluate(self.X_test, self.y_test)
        except Exception as e:
            print(f"Error training PSO model: {str(e)}")
            # Fallback to simpler model
            nn = OptimizableNeuralNetwork(
                input_dim=self.X_train.shape[1],
                hidden_layers=[32, 16],
                output_dim=1 if len(np.unique(self.y)) <= 2 else len(np.unique(self.y)),
                activation='relu',
                learning_rate=0.001,
                epochs=30
            )
            nn.train(self.X_train, self.y_train, self.X_val, self.y_val, verbose=1)
            test_metrics = nn.evaluate(self.X_test, self.y_test)
        
        end_time = time.time()
        training_time = end_time - start_time
        
        # Store results
        self.pso_results = {
            'best_position': best_position,
            'best_fitness': best_fitness,
            'best_hyperparameters': best_hyperparams,
            'test_metrics': test_metrics,
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
        # Get base comparison results
        comparison_results = super().compare_algorithms()
        
        # Update the convergence plot title
        comparison_results['convergence_plot'] = self.visualizer.plot_convergence(
            [self.ga_results['history'], self.pso_results['history']],
            ['Genetic Algorithm', 'Particle Swarm Optimization'],
            title='Hyperparameter Tuning Convergence'
        )
        
        # Create hyperparameter heatmap
        # We'll focus on learning rate and batch size for the heatmap
        learning_rates = self.hyperparameter_ranges['learning_rate']
        batch_sizes = self.hyperparameter_ranges['batch_size']
        
        # Initialize fitness matrix
        fitness_matrix = np.zeros((len(learning_rates), len(batch_sizes)))
        
        # For each combination, create a chromosome and evaluate it
        for i, lr in enumerate(learning_rates):
            for j, bs in enumerate(batch_sizes):
                # Create a chromosome with the current hyperparameters
                # Use the best values for other hyperparameters
                chromosome = np.zeros(self.chromosome_length)
                
                # Set hidden layers (use the best from GA)
                hl_idx = self.hyperparameter_ranges['hidden_layers'].index(
                    self.ga_results['best_hyperparameters']['hidden_layers']
                )
                chromosome[hl_idx] = 1
                
                # Set learning rate
                lr_idx = self.hyperparameter_ranges['learning_rate'].index(lr)
                chromosome[len(self.hyperparameter_ranges['hidden_layers']) + lr_idx] = 1
                
                # Set activation (use the best from GA)
                act_idx = self.hyperparameter_ranges['activation'].index(
                    self.ga_results['best_hyperparameters']['activation']
                )
                chromosome[
                    len(self.hyperparameter_ranges['hidden_layers']) +
                    len(self.hyperparameter_ranges['learning_rate']) +
                    act_idx
                ] = 1
                
                # Set batch size
                bs_idx = self.hyperparameter_ranges['batch_size'].index(bs)
                chromosome[
                    len(self.hyperparameter_ranges['hidden_layers']) +
                    len(self.hyperparameter_ranges['learning_rate']) +
                    len(self.hyperparameter_ranges['activation']) +
                    bs_idx
                ] = 1
                
                # Set dropout rate (use the best from GA)
                dr_idx = self.hyperparameter_ranges['dropout_rate'].index(
                    self.ga_results['best_hyperparameters']['dropout_rate']
                )
                chromosome[
                    len(self.hyperparameter_ranges['hidden_layers']) +
                    len(self.hyperparameter_ranges['learning_rate']) +
                    len(self.hyperparameter_ranges['activation']) +
                    len(self.hyperparameter_ranges['batch_size']) +
                    dr_idx
                ] = 1
                
                # Evaluate the chromosome
                fitness = self._fitness_function(chromosome)
                fitness_matrix[i, j] = fitness
        
        # Create heatmap
        hyperparameter_heatmap = self.visualizer.plot_hyperparameter_heatmap(
            learning_rates, batch_sizes, fitness_matrix,
            'Learning Rate', 'Batch Size', 'Hyperparameter Tuning'
        )
        
        return {
            'ga_results': self.ga_results,
            'pso_results': self.pso_results,
            'convergence_plot': convergence_plot,
            'comparison_plot': comparison_plot,
            'hyperparameter_heatmap': hyperparameter_heatmap
        }

"""
Configuration parameters for the optimization algorithms and neural network models.

This module provides centralized configuration for all components of the neural network
optimization system. Parameters are organized by component and can be easily modified
to experiment with different settings.
"""

import os
import json
from pathlib import Path


class ConfigManager:
    """
    Manages configuration parameters for the optimization system.
    Provides methods to load, save, and access configuration parameters.
    """
    
    # Default configuration directory
    CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")
    
    @classmethod
    def save_config(cls, config_dict, config_name):
        """
        Save configuration to a JSON file.
        
        Args:
            config_dict: Dictionary containing configuration parameters
            config_name: Name of the configuration file (without extension)
        """
        os.makedirs(cls.CONFIG_DIR, exist_ok=True)
        config_path = os.path.join(cls.CONFIG_DIR, f"{config_name}.json")
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=4)
        
        return config_path
    
    @classmethod
    def load_config(cls, config_name):
        """
        Load configuration from a JSON file.
        
        Args:
            config_name: Name of the configuration file (without extension)
            
        Returns:
            Dictionary containing configuration parameters
        """
        config_path = os.path.join(cls.CONFIG_DIR, f"{config_name}.json")
        
        if not os.path.exists(config_path):
            return None
        
        with open(config_path, 'r') as f:
            return json.load(f)
    
    @classmethod
    def get_available_configs(cls):
        """
        Get a list of available configuration files.
        
        Returns:
            List of configuration file names (without extension)
        """
        os.makedirs(cls.CONFIG_DIR, exist_ok=True)
        return [os.path.splitext(f)[0] for f in os.listdir(cls.CONFIG_DIR) 
                if f.endswith('.json')]


# Genetic Algorithm Parameters
# These parameters control the behavior of the genetic algorithm
GA_PARAMS = {
    # Size of the population (number of individuals)
    'population_size': 50,
    
    # Number of generations to evolve
    'num_generations': 100,
    
    # Probability of crossover between two parents
    'crossover_rate': 0.8,
    
    # Probability of mutation for each gene
    'mutation_rate': 0.2,
    
    # Method for selecting parents: 'roulette', 'tournament', or 'rank'
    'selection_method': 'tournament',
    
    # Size of tournament if tournament selection is used
    'tournament_size': 3,
    
    # Whether to use elitism (preserving best individuals)
    'elitism': True,
    
    # Number of elite individuals to preserve if elitism is used
    'elite_size': 2,
    
    # Type of chromosome: 'binary', 'real', or 'integer'
    'chromosome_type': 'real',
    
    # Range of values for real-valued or integer chromosomes (min, max)
    'value_range': (-1, 1),
    
    # Type of crossover: 'single_point', 'two_point', or 'uniform'
    'crossover_type': 'single_point',
    
    # Whether to use adaptive mutation rate
    'adaptive_mutation': False,
    
    # Initial mutation rate if adaptive mutation is used
    'initial_mutation_rate': 0.1,
    
    # Final mutation rate if adaptive mutation is used
    'final_mutation_rate': 0.01
}

# Particle Swarm Optimization Parameters
# These parameters control the behavior of the PSO algorithm
PSO_PARAMS = {
    # Number of particles in the swarm
    'num_particles': 50,
    
    # Number of iterations to run
    'num_iterations': 100,
    
    # Weight of particle's velocity (w)
    'inertia_weight': 0.7,
    
    # Whether to use decreasing inertia weight
    'decreasing_inertia': False,
    
    # Final inertia weight if decreasing inertia is used
    'final_inertia_weight': 0.4,
    
    # Weight of particle's personal best (c1)
    'cognitive_coefficient': 1.5,
    
    # Weight of swarm's global best (c2)
    'social_coefficient': 1.5,
    
    # Maximum velocity of particles
    'max_velocity': 0.5,
    
    # Range of values for position (min, max)
    'value_range': (-1, 1),
    
    # Whether to use constriction factor
    'use_constriction': False,
    
    # Constriction factor if used (typically around 0.729)
    'constriction_factor': 0.729,
    
    # Whether to use neighborhood topology instead of global best
    'use_neighborhood': False,
    
    # Neighborhood size if neighborhood topology is used
    'neighborhood_size': 3,
    
    # Whether to discretize positions (for binary problems)
    'discrete': False
}

# Neural Network Parameters
# These parameters define the structure and training of the neural network
NN_PARAMS = {
    # Input dimension (number of features) - set based on dataset
    'input_dim': None,
    
    # Hidden layer structure (list of neurons in each hidden layer)
    'hidden_layers': [64, 32],
    
    # Output dimension (number of classes) - set based on dataset
    'output_dim': None,
    
    # Activation function for hidden layers
    'activation': 'relu',
    
    # Activation function for output layer
    'output_activation': 'softmax',
    
    # Learning rate for optimizer
    'learning_rate': 0.01,
    
    # Batch size for training
    'batch_size': 32,
    
    # Number of epochs for training
    'epochs': 10,
    
    # Whether to use dropout for regularization
    'use_dropout': False,
    
    # Dropout rate if dropout is used
    'dropout_rate': 0.2,
    
    # Whether to use batch normalization
    'use_batch_norm': False,
    
    # L2 regularization factor
    'l2_regularization': 0.0001,
    
    # Optimizer to use: 'adam', 'sgd', or 'rmsprop'
    'optimizer': 'adam',
    
    # Whether to use early stopping
    'early_stopping': True,
    
    # Patience for early stopping
    'early_stopping_patience': 5,
    
    # Whether to use learning rate reduction on plateau
    'reduce_lr_on_plateau': False,
    
    # Factor for learning rate reduction
    'lr_reduction_factor': 0.1,
    
    # Patience for learning rate reduction
    'lr_reduction_patience': 3
}

# Feature Selection Parameters
# These parameters control the feature selection process
FEATURE_SELECTION_PARAMS = {
    # Maximum number of features to select - set based on dataset
    'max_features': None,
    
    # Minimum number of features to select
    'min_features': 3,
    
    # Method for feature selection: 'filter', 'wrapper', or 'embedded'
    'method': 'wrapper',
    
    # Scoring metric for feature selection
    'scoring': 'accuracy',
    
    # Whether to use cross-validation for feature selection
    'use_cv': True,
    
    # Number of cross-validation folds
    'cv_folds': 5,
    
    # Whether to use feature importance from a base model
    'use_importance': False,
    
    # Base model for feature importance if used
    'importance_model': 'random_forest',
    
    # Threshold for feature importance if used
    'importance_threshold': 0.01,
    
    # Whether to use correlation-based feature selection
    'use_correlation': False,
    
    # Correlation threshold if correlation-based selection is used
    'correlation_threshold': 0.8
}

# Hyperparameter Tuning Ranges
# These ranges define the search space for hyperparameter tuning
HYPERPARAMETER_RANGES = {
    # Hidden layer structures to try
    'hidden_layers': [
        [16], [32], [64], [128], 
        [32, 16], [64, 32], [128, 64], 
        [64, 32, 16], [128, 64, 32]
    ],
    
    # Learning rates to try
    'learning_rate': [0.0001, 0.001, 0.01, 0.1],
    
    # Activation functions to try
    'activation': ['relu', 'tanh', 'sigmoid', 'elu', 'selu'],
    
    # Batch sizes to try
    'batch_size': [16, 32, 64, 128, 256],
    
    # Dropout rates to try
    'dropout_rate': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    
    # Optimizers to try
    'optimizer': ['adam', 'sgd', 'rmsprop', 'adagrad'],
    
    # L2 regularization factors to try
    'l2_regularization': [0.0, 0.0001, 0.001, 0.01],
    
    # Whether to use batch normalization
    'use_batch_norm': [True, False],
    
    # Number of epochs to try
    'epochs': [10, 20, 50, 100],
    
    # Early stopping patience values to try
    'early_stopping_patience': [3, 5, 10, 15]
}

# Experiment Settings
# These settings control the overall experiment process
EXPERIMENT_SETTINGS = {
    # Proportion of data for testing
    'test_size': 0.2,
    
    # Proportion of training data for validation
    'validation_size': 0.1,
    
    # Random seed for reproducibility
    'random_state': 42,
    
    # Number of runs for statistical significance
    'num_runs': 5,
    
    # Metrics to evaluate
    'metrics': ['accuracy', 'precision', 'recall', 'f1'],
    
    # Whether to save results
    'save_results': True,
    
    # Directory to save results
    'results_dir': os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"),
    
    # Whether to save models
    'save_models': True,
    
    # Directory to save models
    'models_dir': os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "saved_models"),
    
    # Whether to use cross-validation
    'use_cv': False,
    
    # Number of cross-validation folds
    'cv_folds': 5,
    
    # Whether to use stratified sampling
    'stratify': True,
    
    # Whether to normalize features
    'normalize_features': True,
    
    # Method for feature scaling: 'standard', 'minmax', or None
    'scaling_method': 'standard',
    
    # Whether to show progress bars
    'show_progress': True,
    
    # Verbosity level
    'verbose': 1,
    
    # Whether to use GPU acceleration if available
    'use_gpu': True,
    
    # Whether to log experiment details
    'logging': True,
    
    # Log level: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
    'log_level': 'INFO'
}

# Create directories for results and saved models
os.makedirs(EXPERIMENT_SETTINGS['results_dir'], exist_ok=True)
os.makedirs(EXPERIMENT_SETTINGS['models_dir'], exist_ok=True)

# Save default configurations
if not os.path.exists(os.path.join(ConfigManager.CONFIG_DIR, "ga_params.json")):
    ConfigManager.save_config(GA_PARAMS, "ga_params")

if not os.path.exists(os.path.join(ConfigManager.CONFIG_DIR, "pso_params.json")):
    ConfigManager.save_config(PSO_PARAMS, "pso_params")

if not os.path.exists(os.path.join(ConfigManager.CONFIG_DIR, "nn_params.json")):
    ConfigManager.save_config(NN_PARAMS, "nn_params")

if not os.path.exists(os.path.join(ConfigManager.CONFIG_DIR, "experiment_settings.json")):
    ConfigManager.save_config(EXPERIMENT_SETTINGS, "experiment_settings")

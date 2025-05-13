"""
Neural Network Optimization Web Application

This module provides a web interface for the neural network optimization project,
allowing users to run experiments with GA and PSO algorithms and visualize results.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
from werkzeug.utils import secure_filename
import io
import base64
import traceback
from datetime import datetime

# Import project modules
from utils.data_loader import DataLoader
from models.neural_network import OptimizableNeuralNetwork
from algorithms.genetic_algorithm import GeneticAlgorithm
from algorithms.particle_swarm import ParticleSwarmOptimization
from experiments.weight_optimization import WeightOptimizationExperiment
from experiments.feature_selection import FeatureSelectionExperiment
from experiments.hyperparameter_tuning import HyperparameterTuningExperiment
from experiments.hyperparameter_tuning_enhanced import EnhancedHyperparameterTuning
from utils.visualization import OptimizationVisualizer
from config.parameters import (
    GA_PARAMS, PSO_PARAMS, NN_PARAMS, FEATURE_SELECTION_PARAMS,
    HYPERPARAMETER_RANGES, EXPERIMENT_SETTINGS, ConfigManager
)

# Import error handling utilities
from functools import wraps
import traceback
import time
from typing import Dict, Any, Callable, TypeVar, Union, Optional

# Initialize Flask app
app = Flask(__name__, 
            static_folder='static',
            template_folder='templates')

# Add custom Jinja2 filters
@app.template_filter('strftime')
def strftime_filter(date_format):
    """Convert a date format string to a formatted date string using the current time."""
    return datetime.now().strftime(date_format)

# Configure upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Configure app settings
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload size
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'csv', 'xlsx', 'xls'}

# Setup logging
import logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()])
logger = logging.getLogger(__name__)

# Helper functions for API responses
def api_success(data: Any = None, message: str = "Success") -> Dict[str, Any]:
    """Standard success response format"""
    return {
        "success": True,
        "message": message,
        "data": data
    }

def api_error(message: str, status_code: int = 400, details: Any = None) -> tuple:
    """Standard error response format"""
    response = {
        "success": False,
        "error": message
    }
    if details:
        response["details"] = details
    return jsonify(response), status_code

# Type variable for return type
T = TypeVar('T')

# API route decorator for consistent error handling
def api_route_handler(f: Callable[..., T]) -> Callable[..., Union[T, tuple]]:
    """Decorator for API routes to handle exceptions consistently"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            start_time = time.time()
            result = f(*args, **kwargs)
            end_time = time.time()
            
            # Log request time for performance monitoring
            logger.info(f"API call to {request.path} completed in {end_time - start_time:.4f} seconds")
            
            return result
        except ValueError as e:
            logger.warning(f"ValueError in {request.path}: {str(e)}")
            return api_error(str(e), 400)
        except KeyError as e:
            logger.warning(f"KeyError in {request.path}: {str(e)}")
            return api_error(f"Missing required parameter: {str(e)}", 400)
        except Exception as e:
            logger.error(f"Exception in {request.path}: {str(e)}\n{traceback.format_exc()}")
            return api_error("An unexpected error occurred", 500, str(e) if app.debug else None)
    return decorated_function

# Request validation helper
def validate_request_data(data: Dict[str, Any], required_fields: list) -> Optional[tuple]:
    """Validate request data has all required fields"""
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        return api_error(f"Missing required fields: {', '.join(missing_fields)}", 400)
    return None

@app.route('/api/visualize_hyperparameters', methods=['POST'])
def visualize_hyperparameters():
    """API endpoint to visualize hyperparameter tuning results"""
    try:
        # Get data from request
        data = request.json
        if not data or 'ga_results' not in data or 'pso_results' not in data:
            return jsonify({'error': 'Invalid data format'}), 400
            
        # Extract hyperparameters from results
        ga_hyperparams = data['ga_results'].get('best_hyperparameters', {})
        pso_hyperparams = data['pso_results'].get('best_hyperparameters', {})
        
        # Create comparison metrics
        comparison_data = {
            'GA': {
                'Accuracy': data['ga_results'].get('test_accuracy', 0),
                'Training Time': data['ga_results'].get('training_time', 0),
                'Convergence': data['ga_results'].get('convergence_gen', 0)
            },
            'PSO': {
                'Accuracy': data['pso_results'].get('test_accuracy', 0),
                'Training Time': data['pso_results'].get('training_time', 0),
                'Convergence': data['pso_results'].get('convergence_gen', 0)
            }
        }
        
        # Create visualizer
        visualizer = OptimizationVisualizer()
        
        # Generate comparison chart
        comparison_fig = visualizer.plot_hyperparameter_comparison(
            ga_hyperparams, pso_hyperparams, title='Best Hyperparameters Comparison'
        )
        
        # Generate performance chart
        performance_fig = visualizer.plot_comparison(
            comparison_data, title='Algorithm Performance Comparison'
        )
        
        # Convert figures to base64 encoded PNGs
        comparison_buffer = io.BytesIO()
        comparison_fig.savefig(comparison_buffer, format='png', dpi=100)
        comparison_buffer.seek(0)
        comparison_b64 = base64.b64encode(comparison_buffer.read()).decode('utf-8')
        
        performance_buffer = io.BytesIO()
        performance_fig.savefig(performance_buffer, format='png', dpi=100)
        performance_buffer.seek(0)
        performance_b64 = base64.b64encode(performance_buffer.read()).decode('utf-8')
        
        # Return the encoded images
        return jsonify({
            'comparison_chart': comparison_b64,
            'performance_chart': performance_b64
        })
        
    except Exception as e:
        logger.error(f"Error visualizing hyperparameters: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Configure results folder
RESULTS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Global variables to store session data
current_data = {
    'X': None,
    'y': None,
    'feature_names': None,
    'target_name': None,
    'data_loader': None,
    'splits': None,
    'current_experiment': None,
    'ga_results': None,
    'pso_results': None,
    'comparison_results': None
}

# Configure results folder
RESULTS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(RESULTS_FOLDER, exist_ok=True)


@app.route('/')
def index():
    """Render the home page."""
    return render_template('index.html')


@app.route('/dataset')
def dataset_page():
    """Render the dataset management page."""
    return render_template('dataset.html')


@app.route('/algorithms')
def algorithms_page():
    """Render the algorithm configuration page."""
    # Load current configurations
    ga_params = GA_PARAMS.copy()
    pso_params = PSO_PARAMS.copy()
    
    # Get available saved configurations directly from the config directory
    config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config')
    saved_configs = {
        'ga': [],
        'pso': []
    }
    
    # Ensure the config directory exists
    os.makedirs(config_dir, exist_ok=True)
    
    # List all JSON files in the config directory
    for filename in os.listdir(config_dir):
        if filename.endswith('.json'):
            # Skip default parameter files
            if filename in ['ga_params.json', 'pso_params.json', 'nn_params.json', 'experiment_settings.json']:
                continue
                
            # Determine the type of configuration
            if '_ga.json' in filename:
                saved_configs['ga'].append(filename.replace('_ga.json', ''))
            elif '_pso.json' in filename:
                saved_configs['pso'].append(filename.replace('_pso.json', ''))
            else:
                # Try to determine the type by loading and checking the content
                try:
                    with open(os.path.join(config_dir, filename), 'r') as f:
                        config_data = json.load(f)
                        if 'population_size' in config_data:
                            saved_configs['ga'].append(filename.replace('.json', ''))
                        elif 'num_particles' in config_data:
                            saved_configs['pso'].append(filename.replace('.json', ''))
                except:
                    # Skip files that can't be parsed
                    continue
    
    print(f"Found saved configurations: {saved_configs}")
    
    return render_template('algorithms.html', 
                          ga_params=ga_params,
                          pso_params=pso_params,
                          saved_configs=saved_configs)


@app.route('/experiments')
def experiments_page():
    """Render the experiment setup page."""
    # Check if data is loaded
    if current_data['X'] is None:
        return redirect(url_for('dataset_page'))
    
    return render_template('experiments.html',
                          has_data=current_data['X'] is not None,
                          feature_names=current_data['feature_names'],
                          target_name=current_data['target_name'],
                          num_features=len(current_data['feature_names']) if current_data['feature_names'] else 0,
                          num_samples=len(current_data['X']) if current_data['X'] is not None else 0)


@app.route('/results')
def results_page():
    """Render the results visualization page."""
    # Check if experiments have been run
    if current_data['ga_results'] is None and current_data['pso_results'] is None:
        return redirect(url_for('experiments_page'))
    
    return render_template('results.html',
                          has_ga_results=current_data['ga_results'] is not None,
                          has_pso_results=current_data['pso_results'] is not None,
                          has_comparison=current_data['comparison_results'] is not None)

@app.route('/api/upload_dataset', methods=['POST'])
@api_route_handler
def upload_dataset():
    """Handle dataset upload."""
    # Check if a file was uploaded
    if 'file' not in request.files:
        return api_error('No file part')
    
    file = request.files['file']
    
    # Check if file is empty
    if file.filename == '':
        return api_error('No file selected')
    
    # Check file extension
    if not file.filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']:
        return api_error('Invalid file type. Only CSV and Excel files are allowed')
    
    # Save the file
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    # Load the dataset
    data_loader = DataLoader()
    
    # Determine file type and load accordingly
    file_ext = file.filename.rsplit('.', 1)[1].lower()
    
    try:
        if file_ext == 'csv':
            # Read the CSV file
            df = pd.read_csv(file_path)
        elif file_ext in ['xls', 'xlsx']:
            # For Excel files, read and convert to CSV first
            df = pd.read_excel(file_path)
            temp_csv_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_converted.csv')
            df.to_csv(temp_csv_path, index=False)
            file_path = temp_csv_path  # Use the converted CSV path
        else:
            return api_error(f'Unsupported file type: {file_ext}')
        
        # Log dataset information
        logger.info(f"Dataset loaded with shape: {df.shape}, columns: {df.columns.tolist()}")
        
        # Intelligent target column selection
        # Look for common target column names
        target_column_candidates = ['disease_risk', 'target', 'label', 'class', 'y', 'output', 'result']
        target_column = None
        
        # First try exact matches
        for candidate in target_column_candidates:
            if candidate in df.columns:
                target_column = candidate
                logger.info(f"Found target column: {target_column}")
                break
        
        # If no exact match, try case-insensitive partial matches
        if target_column is None:
            for col in df.columns:
                if any(candidate in col.lower() for candidate in target_column_candidates):
                    target_column = col
                    logger.info(f"Found likely target column: {target_column}")
                    break
        
        # If still no match, use the last column as it's a common convention
        if target_column is None:
            target_column = df.columns[-1]
            logger.info(f"No target column found, using last column: {target_column}")
        
        # Load the data with the selected target column
        X, y = data_loader.load_csv(file_path, target_column=target_column)
        feature_names = data_loader.feature_names
        target_name = data_loader.target_name
        
        # Log more information about the dataset
        logger.info(f"Dataset processed: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Target column: {target_name}, unique values: {np.unique(y)}")
        
    except Exception as e:
        logger.error(f"Error processing dataset: {str(e)}")
        return api_error(f'Error processing dataset: {str(e)}')
    
    # Store in session data
    current_data['X'] = X
    current_data['y'] = y
    current_data['feature_names'] = feature_names
    current_data['target_name'] = target_name
    current_data['data_loader'] = data_loader
    
    # Create train/val/test splits
    current_data['splits'] = data_loader.preprocess_data(X, y)
    
    # Return success response with dataset info
    return jsonify(api_success(
        data={
            'dataset_info': {
                'num_samples': X.shape[0],
                'num_features': X.shape[1],
                'feature_names': feature_names.tolist() if isinstance(feature_names, np.ndarray) else feature_names,
                'target_name': target_name,
                'target_column': target_name  # Adding target_column for consistency with frontend
            }
        },
        message='Dataset uploaded and processed successfully'
    ))

@app.route('/api/generate_dataset', methods=['POST'])
@api_route_handler
def generate_dataset():
    """Generate synthetic dataset."""
    # Get parameters from request
    data = request.get_json()
    if not data:
        return api_error('Invalid JSON data')
    
    # Validate required fields
    validation_error = validate_request_data(data, ['num_samples', 'num_features'])
    if validation_error:
        return validation_error
    
    num_samples = int(data.get('num_samples', 100))
    num_features = int(data.get('num_features', 10))
    noise = float(data.get('noise', 0.1))
    
    # Validate parameter values
    if num_samples <= 0 or num_samples > 10000:
        return api_error('Number of samples must be between 1 and 10000')
    
    if num_features <= 0 or num_features > 100:
        return api_error('Number of features must be between 1 and 100')
    
    if noise < 0 or noise > 1:
        return api_error('Noise must be between 0 and 1')
    
    # Generate synthetic data
    data_loader = DataLoader()
    X, y, feature_names, target_name = data_loader.generate_synthetic_data(
        num_samples=num_samples,
        num_features=num_features,
        noise=noise
    )
    
    # Store in session data
    current_data['X'] = X
    current_data['y'] = y
    current_data['feature_names'] = feature_names
    current_data['target_name'] = target_name
    current_data['data_loader'] = data_loader
    
    # Create train/val/test splits
    current_data['splits'] = data_loader.preprocess_data(X, y)
    
    # Log dataset generation
    logger.info(f"Generated synthetic dataset with {num_samples} samples and {num_features} features")
    
    # Return success response with dataset info
    return jsonify(api_success(
        data={
            'dataset_info': {
                'num_samples': X.shape[0],
                'num_features': X.shape[1],
                'feature_names': feature_names.tolist() if isinstance(feature_names, np.ndarray) else feature_names,
                'target_name': target_name
            }
        },
        message='Synthetic dataset generated successfully'
    ))

@app.route('/api/save_config', methods=['POST'])
@api_route_handler
def save_config():
    """Save algorithm configuration."""
    try:
        # Get form data
        config_type = request.form.get('config_type')
        config_name = request.form.get('config_name')
        config_data_str = request.form.get('config_data')
        
        if not config_name:
            return api_error('Configuration name is required')
        
        if not config_data_str:
            return api_error('Configuration data is required')
        
        # Parse the JSON data
        try:
            config_data = json.loads(config_data_str)
        except json.JSONDecodeError as e:
            return api_error(f'Invalid JSON data: {str(e)}')
        
        # Ensure config directory exists
        config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config')
        os.makedirs(config_dir, exist_ok=True)
        
        # Save configuration directly to a file
        if config_type == 'ga':
            filename = f"{config_name}_ga.json"
        elif config_type == 'pso':
            filename = f"{config_name}_pso.json"
        else:
            filename = f"{config_name}.json"
            
        config_path = os.path.join(config_dir, filename)
        
        # Write the configuration to the file
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=4)
        
        print(f"Configuration saved to: {config_path}")
        
        return api_success(
            data={
                'config_path': config_path
            },
            message=f'Configuration saved successfully as {config_name}'
        )
        
    except Exception as e:
        print(f"Error saving configuration: {str(e)}")
        return api_error(str(e))


@app.route('/api/load_config', methods=['POST'])
@api_route_handler
def load_config():
    """Load algorithm configuration."""
    try:
        # Get form data
        config_name = request.form.get('config_name')
        
        if not config_name:
            return api_error('Configuration name is required')
        
        # Determine the file path based on the config name
        config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config')
        
        # Try different possible filenames
        possible_paths = [
            os.path.join(config_dir, f"{config_name}.json"),
            os.path.join(config_dir, f"{config_name}_ga.json"),
            os.path.join(config_dir, f"{config_name}_pso.json")
        ]
        
        config_data = None
        config_path = None
        
        for path in possible_paths:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    config_data = json.load(f)
                config_path = path
                break
        
        if config_data is None:
            return api_error(f'Configuration {config_name} not found')
        
        print(f"Configuration loaded from: {config_path}")
        
        return api_success(
            data={
                'config_data': config_data
            },
            message=f'Configuration {config_name} loaded successfully'
        )
        
    except Exception as e:
        print(f"Error loading configuration: {str(e)}")
        return api_error(str(e))


@app.route('/api/run_experiment', methods=['POST'])
@api_route_handler
def run_experiment():
    """Run an optimization experiment."""
    # Check if data is loaded
    if current_data['X'] is None or current_data['splits'] is None:
        return api_error('No dataset loaded. Please upload or generate a dataset first.')
    
    # Get experiment parameters from request
    data = request.form.to_dict()
    
    # Validate required fields
    required_fields = ['experiment_type', 'algorithm']
    missing_fields = [field for field in required_fields if field not in data or not data[field]]
    if missing_fields:
        return api_error(f"Missing required fields: {', '.join(missing_fields)}")
    
    experiment_type = data['experiment_type']
    algorithm = data['algorithm']
    
    # Validate algorithm type
    if algorithm not in ['ga', 'pso', 'both']:
        return api_error(f"Invalid algorithm type: {algorithm}. Must be 'ga', 'pso', or 'both'.")
    
    # Get split data
    splits = current_data['splits']
    
    # Log experiment request
    logger.info(f"Running {experiment_type} experiment with {algorithm} algorithm")
    
    # Initialize experiment parameters
    experiment_params = {}
    
    # Get GA parameters if needed
    if algorithm in ['ga', 'both']:
        ga_params = {
            'population_size': int(data.get('ga_population_size', 50)),
            'num_generations': int(data.get('ga_num_generations', 100)),
            'mutation_rate': float(data.get('ga_mutation_rate', 0.2)),
            'mutation_type': data.get('ga_mutation_type', 'bit-flip'),
            'crossover_type': data.get('ga_crossover_type', 'single_point'),
            'selection_method': data.get('ga_selection_method', 'tournament'),
            'elitism': data.get('ga_elitism', 'true').lower() == 'true',
            'chromosome_type': 'real'  # Always use real-valued chromosomes for neural networks
        }
        experiment_params['ga_params'] = ga_params
    
    # Get PSO parameters if needed
    if algorithm in ['pso', 'both']:
        pso_params = {
            'num_particles': int(data.get('pso_num_particles', 50)),
            'num_iterations': int(data.get('pso_num_iterations', 100)),
            'inertia_weight': float(data.get('pso_inertia_weight', 0.7)),
            'cognitive_coefficient': float(data.get('pso_cognitive_coefficient', 1.5)),
            'social_coefficient': float(data.get('pso_social_coefficient', 1.5))
        }
        experiment_params['pso_params'] = pso_params
    
    # Run appropriate experiment type
    if experiment_type == 'weight_optimization':
        # Parse neural network parameters
        try:
            hidden_layers_str = data.get('hidden_layers', '[64, 32]')
            hidden_layers = json.loads(hidden_layers_str)
            if not isinstance(hidden_layers, list):
                return api_error('Hidden layers must be a list of integers')
        except json.JSONDecodeError:
            return api_error('Invalid hidden layers format. Must be a JSON array.')
        
        activation = data.get('activation', 'relu')
        if activation not in ['relu', 'sigmoid', 'tanh']:
            return api_error(f"Invalid activation function: {activation}. Must be 'relu', 'sigmoid', or 'tanh'.")
        
        # Create experiment with appropriate parameters
        experiment = WeightOptimizationExperiment(
            X_train=splits['X_train'], 
            y_train=splits['y_train'],
            X_val=splits['X_val'], 
            y_val=splits['y_val'],
            X_test=splits['X_test'], 
            y_test=splits['y_test'],
            input_dim=current_data['X'].shape[1],
            output_dim=1 if current_data['data_loader'].num_classes <= 2 else current_data['data_loader'].num_classes,
            hidden_layers=hidden_layers,
            activation=activation,
            **experiment_params
        )
            
        # Store the experiment
        current_data['current_experiment'] = experiment
        
        # Run the selected algorithm(s) with progress tracking
        start_time = time.time()
        
        if algorithm in ['ga', 'both']:
            logger.info(f"Starting GA optimization with {experiment_params.get('ga_params', {}).get('population_size', 50)} population size and {experiment_params.get('ga_params', {}).get('num_generations', 100)} generations")
            current_data['ga_results'] = experiment.run_ga_optimization(verbose=True)
            logger.info(f"GA optimization completed with best fitness: {current_data['ga_results']['best_fitness']}")
        
        if algorithm in ['pso', 'both']:
            logger.info(f"Starting PSO optimization with {experiment_params.get('pso_params', {}).get('num_particles', 50)} particles and {experiment_params.get('pso_params', {}).get('num_iterations', 100)} iterations")
            current_data['pso_results'] = experiment.run_pso_optimization(verbose=True)
            logger.info(f"PSO optimization completed with best fitness: {current_data['pso_results']['best_fitness']}")
        
        # Compare algorithms if both were run
        if algorithm == 'both':
            logger.info("Comparing GA and PSO performance")
            current_data['comparison_results'] = experiment.compare_algorithms()
            
            # Save plots with timestamp to avoid overwriting
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create plot filenames
            convergence_plot_path = os.path.join(RESULTS_FOLDER, f'weight_optimization_convergence_{timestamp}.png')
            comparison_plot_path = os.path.join(RESULTS_FOLDER, f'weight_optimization_comparison_{timestamp}.png')
            
            # Save plots to disk
            experiment.visualizer.save_plot(
                current_data['comparison_results']['convergence_plot'],
                convergence_plot_path
            )
            
            experiment.visualizer.save_plot(
                current_data['comparison_results']['comparison_plot'],
                comparison_plot_path
            )
            
            logger.info(f"Comparison plots saved to {RESULTS_FOLDER}")
        
        # Calculate total execution time
        execution_time = time.time() - start_time
        
        return jsonify(api_success(
            data={
                'ran_ga': algorithm in ['ga', 'both'],
                'ran_pso': algorithm in ['pso', 'both'],
                'ran_comparison': algorithm == 'both',
                'execution_time': f"{execution_time:.2f} seconds",
                'redirect_to': url_for('results_page')
            },
            message=f'Weight optimization experiment completed successfully in {execution_time:.2f} seconds'
        ))
            
    elif experiment_type == 'feature_selection':
        # Parse feature selection parameters
        try:
            num_features = int(data.get('num_features', 0))
            if num_features <= 0:
                num_features = max(1, current_data['X'].shape[1] // 2)  # Default to half of features if not specified
                
            # Create feature selection experiment
            experiment = FeatureSelectionExperiment(
                X=current_data['X'],
                y=current_data['y'],
                feature_names=current_data['feature_names'],
                test_size=0.2,
                validation_size=0.1,
                feature_selection_params={'max_features': num_features},
                ga_params=experiment_params.get('ga_params'),
                pso_params=experiment_params.get('pso_params')
            )
            
            # Store the experiment
            current_data['current_experiment'] = experiment
            
            # Run the selected algorithm(s)
            start_time = time.time()
            
            if algorithm in ['ga', 'both']:
                logger.info(f"Starting GA feature selection for {num_features} features")
                current_data['ga_results'] = experiment.run_ga_feature_selection(verbose=True)
                logger.info(f"GA feature selection completed with best fitness: {current_data['ga_results']['best_fitness']}")
            
            if algorithm in ['pso', 'both']:
                logger.info(f"Starting PSO feature selection for {num_features} features")
                current_data['pso_results'] = experiment.run_pso_feature_selection(verbose=True)
                logger.info(f"PSO feature selection completed with best fitness: {current_data['pso_results']['best_fitness']}")
            
            # Compare algorithms if both were run
            if algorithm == 'both':
                logger.info("Comparing GA and PSO feature selection performance")
                current_data['comparison_results'] = experiment.compare_algorithms()
                
                # Save plots with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                convergence_plot_path = os.path.join(RESULTS_FOLDER, f'feature_selection_convergence_{timestamp}.png')
                selected_features_plot_path = os.path.join(RESULTS_FOLDER, f'feature_selection_features_{timestamp}.png')
                
                experiment.visualizer.save_plot(
                    current_data['comparison_results']['convergence_plot'],
                    convergence_plot_path
                )
                
                # Save GA feature importance plot instead of selected_features_plot
                experiment.visualizer.save_plot(
                    current_data['comparison_results']['ga_feature_importance_plot'],
                    selected_features_plot_path
                )
            
            # Calculate total execution time
            execution_time = time.time() - start_time
            
            return jsonify(api_success(
                data={
                    'ran_ga': algorithm in ['ga', 'both'],
                    'ran_pso': algorithm in ['pso', 'both'],
                    'ran_comparison': algorithm == 'both',
                    'execution_time': f"{execution_time:.2f} seconds",
                    'redirect_to': url_for('results_page')
                },
                message=f'Feature selection experiment completed successfully in {execution_time:.2f} seconds'
            ))
            
        except ValueError as e:
            return api_error(f"Invalid parameter for feature selection: {str(e)}")
            
    elif experiment_type == 'hyperparameter_tuning':
        # Parse hyperparameter tuning parameters
        try:
            # Create hyperparameter configuration based on form data
            hyperparameter_config = {
                # Which hyperparameters to tune
                'tune_hidden_layers': data.get('tune_hidden_layers', 'on') == 'on',
                'tune_learning_rate': data.get('tune_learning_rate', 'on') == 'on',
                'tune_activation': data.get('tune_activation', 'on') == 'on',
                'tune_batch_size': data.get('tune_batch_size', 'on') == 'on',
                'tune_dropout': data.get('tune_dropout', 'on') == 'on',
                'tune_optimizer': data.get('tune_optimizer', 'off') == 'on',
                
                # Custom ranges if provided
                'hidden_layers': [
                    [16], [32], [64], [128],
                    [32, 16], [64, 32], [128, 64],
                    [64, 32, 16], [128, 64, 32]
                ],
                'learning_rate': [0.0001, 0.001, 0.01, 0.1],
                'activation': ['relu', 'tanh', 'sigmoid', 'elu'],
                'batch_size': [16, 32, 64, 128, 256],
                'dropout_rate': [0.0, 0.1, 0.2, 0.3, 0.5],
                'optimizer': ['adam', 'sgd', 'rmsprop']
            }
            
            # Log hyperparameter tuning configuration
            logger.info(f"Hyperparameter tuning configuration: {hyperparameter_config}")
            
            # Create enhanced hyperparameter tuning experiment
            experiment = EnhancedHyperparameterTuning(
                X=current_data['X'],
                y=current_data['y'],
                test_size=0.2,
                validation_size=0.1,
                ga_params=experiment_params.get('ga_params'),
                pso_params=experiment_params.get('pso_params'),
                hyperparameter_config=hyperparameter_config
            )
            
            # Store the experiment
            current_data['current_experiment'] = experiment
            
            # Run the selected algorithm(s)
            start_time = time.time()
            
            if algorithm in ['ga', 'both']:
                logger.info("Starting GA hyperparameter tuning")
                current_data['ga_results'] = experiment.run_ga_optimization(verbose=True)
                logger.info(f"GA hyperparameter tuning completed with best fitness: {current_data['ga_results']['best_fitness']}")
            
            if algorithm in ['pso', 'both']:
                logger.info("Starting PSO hyperparameter tuning")
                current_data['pso_results'] = experiment.run_pso_optimization(verbose=True)
                logger.info(f"PSO hyperparameter tuning completed with best fitness: {current_data['pso_results']['best_fitness']}")
            
            # Compare algorithms if both were run
            if algorithm == 'both':
                logger.info("Comparing GA and PSO hyperparameter tuning performance")
                current_data['comparison_results'] = experiment.compare_algorithms()
                
                # Save plots with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                convergence_plot_path = os.path.join(RESULTS_FOLDER, f'hyperparameter_convergence_{timestamp}.png')
                param_importance_plot_path = os.path.join(RESULTS_FOLDER, f'hyperparameter_importance_{timestamp}.png')
                
                experiment.visualizer.save_plot(
                    current_data['comparison_results']['convergence_plot'],
                    convergence_plot_path
                )
                
                experiment.visualizer.save_plot(
                    current_data['comparison_results']['param_importance_plot'],
                    param_importance_plot_path
                )
            
            # Calculate total execution time
            execution_time = time.time() - start_time
            
            return jsonify(api_success(
                data={
                    'ran_ga': algorithm in ['ga', 'both'],
                    'ran_pso': algorithm in ['pso', 'both'],
                    'ran_comparison': algorithm == 'both',
                    'execution_time': f"{execution_time:.2f} seconds",
                    'redirect_to': url_for('results_page')
                },
                message=f'Hyperparameter tuning experiment completed successfully in {execution_time:.2f} seconds'
            ))
            
        except ValueError as e:
            return api_error(f"Invalid parameter for hyperparameter tuning: {str(e)}")
    
    else:
        return api_error(f"Experiment type '{experiment_type}' is not supported. Available types: weight_optimization, feature_selection, hyperparameter_tuning", 400)


@app.route('/api/get_results')
@api_route_handler
def get_results():
    """Get experiment results."""
    # Check if experiments have been run
    if current_data['ga_results'] is None and current_data['pso_results'] is None:
        return api_error('No experiments have been run. Please run an experiment first.')
    
    # Prepare results data structure
    results = {
        'has_ga_results': current_data['ga_results'] is not None,
        'has_pso_results': current_data['pso_results'] is not None,
        'has_comparison': current_data['comparison_results'] is not None,
        'experiment_type': current_data['current_experiment'].__class__.__name__ if current_data['current_experiment'] else 'Unknown'
    }
    
    # Generate patient assessments directly in the get_results endpoint
    # Use the best performing model for predictions
    best_model = None
    if current_data['ga_results'] is not None and current_data['pso_results'] is not None:
        # Compare GA and PSO accuracy to determine best model
        ga_accuracy = current_data['ga_results']['test_metrics']['accuracy']
        pso_accuracy = current_data['pso_results']['test_metrics']['accuracy']
        # Check if 'model' key exists before accessing it
        if 'model' in current_data['ga_results'] and 'model' in current_data['pso_results']:
            best_model = current_data['ga_results']['model'] if ga_accuracy >= pso_accuracy else current_data['pso_results']['model']
    elif current_data['ga_results'] is not None and 'model' in current_data['ga_results']:
        best_model = current_data['ga_results']['model']
    elif current_data['pso_results'] is not None and 'model' in current_data['pso_results']:
        best_model = current_data['pso_results']['model']
    
    # Generate patient assessments if model is available
    if best_model is not None and current_data['X'] is not None:
        try:
            # Get feature names for better context
            feature_names = current_data['feature_names']
            if isinstance(feature_names, np.ndarray):
                feature_names = feature_names.tolist()
            
            # Generate medical risk assessments
            logger.info(f"Generating patient assessments in get_results endpoint")
            patient_assessments = best_model.medical_risk_assessment(current_data['X'], feature_names)
            logger.info(f"Generated {len(patient_assessments)} patient assessments")
            
            # Add to results
            results['patient_assessments'] = patient_assessments
        except Exception as e:
            logger.error(f"Error generating patient assessments in get_results: {str(e)}")
            logger.error(traceback.format_exc())
    
    # Add GA results if available
    if current_data['ga_results'] is not None:
        results['ga'] = {
            'best_fitness': float(current_data['ga_results']['best_fitness']),
            'test_metrics': {
                k: (float(v) if v != '' else 0.0) if isinstance(v, (int, float, str)) and k not in ['medical_interpretation'] else v 
                for k, v in current_data['ga_results']['test_metrics'].items()
            },
            'training_time': float(current_data['ga_results']['training_time'])
        }
        
        # Add selected features if this was a feature selection experiment
        if 'selected_features' in current_data['ga_results']:
            results['ga']['selected_features'] = current_data['ga_results']['selected_features']
        
        # Add best hyperparameters if this was a hyperparameter tuning experiment
        if 'best_hyperparameters' in current_data['ga_results']:
            results['ga']['best_hyperparameters'] = current_data['ga_results']['best_hyperparameters']
    
    # Add PSO results if available
    if current_data['pso_results'] is not None:
        results['pso'] = {
            'best_fitness': float(current_data['pso_results']['best_fitness']),
            'test_metrics': {
                k: (float(v) if v != '' else 0.0) if isinstance(v, (int, float, str)) and k not in ['medical_interpretation'] else v 
                for k, v in current_data['pso_results']['test_metrics'].items()
            },
            'training_time': float(current_data['pso_results']['training_time'])
        }
        
        # Add selected features if this was a feature selection experiment
        if 'selected_features' in current_data['pso_results']:
            results['pso']['selected_features'] = current_data['pso_results']['selected_features']
        
        # Add best hyperparameters if this was a hyperparameter tuning experiment
        if 'best_hyperparameters' in current_data['pso_results']:
            results['pso']['best_hyperparameters'] = current_data['pso_results']['best_hyperparameters']
    
    # Add comparison data if available
    if current_data['comparison_results'] is not None:
        # Helper function to convert matplotlib figure to base64 image
        def fig_to_base64(fig):
            img_buffer = io.BytesIO()
            fig.savefig(img_buffer, format='png', bbox_inches='tight')
            img_buffer.seek(0)
            img_str = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
            return img_str
            
        # Get plot figures and convert to base64 images
        results['comparison'] = {}
        
        if 'convergence_plot' in current_data['comparison_results']:
            convergence_plot = current_data['comparison_results']['convergence_plot']
            results['comparison']['convergence_plot'] = fig_to_base64(convergence_plot)
        
        if 'comparison_plot' in current_data['comparison_results']:
            comparison_plot = current_data['comparison_results']['comparison_plot']
            results['comparison']['comparison_plot'] = fig_to_base64(comparison_plot)
        
        # Add additional plots based on experiment type
        if 'selected_features_plot' in current_data['comparison_results']:
            features_plot = current_data['comparison_results']['selected_features_plot']
            results['comparison']['selected_features_plot'] = fig_to_base64(features_plot)
        
        if 'param_importance_plot' in current_data['comparison_results']:
            param_plot = current_data['comparison_results']['param_importance_plot']
            results['comparison']['param_importance_plot'] = fig_to_base64(param_plot)
        
    # Log results retrieval
    logger.info(f"Retrieved results for {results['experiment_type']} experiment")
    
    # Return standardized success response
    return jsonify(api_success(
        data=results,
        message='Experiment results retrieved successfully'
    ))


@app.route('/api/parameter_impact_study', methods=['POST'])
@api_route_handler
def parameter_impact_study():
    """Run a parameter impact study."""
    # Check if an experiment has been run
    if current_data['current_experiment'] is None:
        return api_error('No experiment has been run. Please run an experiment first.')
    
    # Get request data
    data = request.get_json()
    if not data:
        return api_error('Invalid JSON data')
    
    # Validate required fields
    validation_error = validate_request_data(data, ['parameter', 'min_val', 'max_val'])
    if validation_error:
        return validation_error
    
    # Get parameter to study
    parameter = data['parameter']
    min_val = float(data['min_val'])
    max_val = float(data['max_val'])
    num_points = int(data.get('num_points', 10))
    
    # Validate parameter values
    if min_val >= max_val:
        return api_error('min_val must be less than max_val')
    
    if num_points < 2 or num_points > 50:
        return api_error('num_points must be between 2 and 50')
    
    # Log parameter study
    logger.info(f"Running parameter impact study for {parameter} from {min_val} to {max_val} with {num_points} points")
    
    # Run the parameter impact study
    results = current_data['current_experiment'].parameter_impact_study(
        parameter=parameter,
        min_val=min_val,
        max_val=max_val,
        num_points=num_points
    )
    
    # Convert plot to base64 image
    img_buffer = io.BytesIO()
    results['impact_plot'].savefig(img_buffer, format='png', bbox_inches='tight')
    img_buffer.seek(0)
    impact_plot_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    
    # Return results
    return jsonify(api_success(
        data={
            'parameter': parameter,
            'values': results['values'].tolist(),
            'metrics': results['metrics'].tolist(),
            'plot': impact_plot_base64
        },
        message=f'Parameter impact study for {parameter} completed successfully'
    ))


@app.route('/api/export_results', methods=['POST'])
@api_route_handler
def export_results():
    """Export experiment results to a file."""
    # Check if experiments have been run
    if current_data['ga_results'] is None and current_data['pso_results'] is None:
        return api_error('No experiments have been run. Please run an experiment first.')
    
    # Get export format from request
    data = request.form.to_dict()
    export_format = data.get('format', 'json').lower()
    
    # Validate export format
    if export_format not in ['json', 'csv']:
        return api_error(f"Unsupported export format: {export_format}. Supported formats: json, csv")
    
    # Create results dictionary with metadata
    results = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'experiment_type': current_data['current_experiment'].__class__.__name__,
        'dataset_info': {
            'num_samples': current_data['X'].shape[0],
            'num_features': current_data['X'].shape[1],
            'feature_names': current_data['feature_names'].tolist() if isinstance(current_data['feature_names'], np.ndarray) else current_data['feature_names'],
            'target_name': current_data['target_name']
        }
    }
    
    # Add per-patient predictions if models are available
    patient_predictions = []
    
    # Use the best performing model for predictions
    best_model = None
    if current_data['ga_results'] is not None and current_data['pso_results'] is not None:
        # Compare GA and PSO accuracy to determine best model
        ga_accuracy = current_data['ga_results']['test_metrics']['accuracy']
        pso_accuracy = current_data['pso_results']['test_metrics']['accuracy']
        best_model = current_data['ga_results']['model'] if ga_accuracy >= pso_accuracy else current_data['pso_results']['model']
    elif current_data['ga_results'] is not None:
        best_model = current_data['ga_results']['model']
    elif current_data['pso_results'] is not None:
        best_model = current_data['pso_results']['model']
    
    # Generate per-patient predictions if model is available
    if best_model is not None and current_data['X'] is not None:
        try:
            # Get feature names for better context
            feature_names = current_data['feature_names']
            if isinstance(feature_names, np.ndarray):
                feature_names = feature_names.tolist()
            
            # Get comprehensive medical risk assessments with detailed logging
            logger.info(f"Generating medical risk assessments for {len(current_data['X'])} patients")
            patient_assessments = best_model.medical_risk_assessment(current_data['X'], feature_names)
            logger.info(f"Successfully generated {len(patient_assessments)} patient assessments")
            
            # Add assessments to results
            results['patient_assessments'] = patient_assessments
            logger.info(f"Added patient assessments to results")
            
            # Log a sample assessment for debugging
            if patient_assessments and len(patient_assessments) > 0:
                logger.info(f"Sample assessment keys: {list(patient_assessments[0].keys())}")
            
        except Exception as e:
            logger.error(f"Error in medical risk assessment: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Fallback to basic prediction if medical_risk_assessment fails
            # Get predictions for all patients
            predictions = best_model.predict(current_data['X'])
            
            # Convert to risk probabilities/classes
            if best_model.output_dim == 1:
                risk_probs = predictions.flatten()
                risk_classes = (risk_probs > 0.5).astype(int)
            else:
                risk_probs = np.max(predictions, axis=1)
                risk_classes = np.argmax(predictions, axis=1)
        
        for i in range(len(current_data['X'])):
            # Get original biomarker values
            biomarkers = {}
            for j, feature in enumerate(current_data['feature_names']):
                biomarkers[feature] = float(current_data['X'][i, j])
            
            # Add medical interpretation based on biomarker values
            medical_notes = []
            if biomarkers.get('glucose_level', 0) > 0.7:
                medical_notes.append("High glucose level - diabetes risk factor")
            if biomarkers.get('blood_pressure', 0) > 0.75:
                medical_notes.append("High blood pressure - hypertension risk factor")
            if biomarkers.get('bmi', 0) > 0.67:
                medical_notes.append("High BMI - obesity risk factor")
            if biomarkers.get('cholesterol', 0) > 0.65:
                medical_notes.append("High cholesterol - cardiovascular risk factor")
            if biomarkers.get('oxygen_saturation', 0) < 0.9:
                medical_notes.append("Low oxygen saturation - respiratory concern")
            
            # Calculate risk category based on probability
            risk_category = "Low"
            if risk_probs[i] > 0.7:
                risk_category = "High"
            elif risk_probs[i] > 0.3:
                risk_category = "Moderate"
            
            # Create patient record
            patient_record = {
                'patient_id': i + 1,
                'biomarkers': biomarkers,
                'actual_risk': int(current_data['y'][i]) if i < len(current_data['y']) else None,
                'predicted_risk_probability': float(risk_probs[i]),
                'predicted_risk_class': int(risk_classes[i]),
                'risk_category': risk_category,
                'medical_notes': medical_notes
            }
            
            patient_predictions.append(patient_record)
        
        # Add patient predictions to results
        results['patient_predictions'] = patient_predictions
    
    # Add GA results if available
    if current_data['ga_results'] is not None:
        results['ga_results'] = {
            'best_fitness': float(current_data['ga_results']['best_fitness']),
            'test_metrics': {
                k: (float(v) if v != '' else 0.0) if isinstance(v, (int, float, str)) and k not in ['medical_interpretation'] else v 
                for k, v in current_data['ga_results']['test_metrics'].items()
            },
            'training_time': float(current_data['ga_results']['training_time'])
        }
        
    # Add PSO results if available
    if current_data['pso_results'] is not None:
        results['pso_results'] = {
            'best_fitness': float(current_data['pso_results']['best_fitness']),
            'test_metrics': {
                k: (float(v) if v != '' else 0.0) if isinstance(v, (int, float, str)) and k not in ['medical_interpretation'] else v 
                for k, v in current_data['pso_results']['test_metrics'].items()
            },
            'training_time': float(current_data['pso_results']['training_time'])
        }
        
        # Add experiment-specific results
        if 'selected_features' in current_data['pso_results']:
            results['pso_results']['selected_features'] = current_data['pso_results']['selected_features']
        
        if 'best_hyperparameters' in current_data['pso_results']:
            results['pso_results']['best_hyperparameters'] = current_data['pso_results']['best_hyperparameters']
    
    # Create timestamp and filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{results['experiment_type']}_{timestamp}.{export_format}"
    filepath = os.path.join(RESULTS_FOLDER, filename)
    
    # Export based on format
    if export_format == 'json':
        # Write to JSON file
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=4, cls=NumpyEncoder)
    
    elif export_format == 'csv':
        # Flatten results for CSV format
        flat_results = {}
        flat_results['timestamp'] = results['timestamp']
        flat_results['experiment_type'] = results['experiment_type']
        flat_results['num_samples'] = results['dataset_info']['num_samples']
        flat_results['num_features'] = results['dataset_info']['num_features']
        
        # Flatten GA results
        if current_data['ga_results'] is not None:
            for k, v in results['ga_results']['test_metrics'].items():
                flat_results[f'ga_{k}'] = v
            flat_results['ga_best_fitness'] = results['ga_results']['best_fitness']
            flat_results['ga_training_time'] = results['ga_results']['training_time']
        
        # Flatten PSO results
        if current_data['pso_results'] is not None:
            for k, v in results['pso_results']['test_metrics'].items():
                flat_results[f'pso_{k}'] = v
            flat_results['pso_best_fitness'] = results['pso_results']['best_fitness']
            flat_results['pso_training_time'] = results['pso_results']['training_time']
        
        # Write to CSV file
        df = pd.DataFrame([flat_results])
        df.to_csv(filepath, index=False)
    
    # Log export
    logger.info(f"Exported results to {filepath}")
    
    # Return success response
    return jsonify(api_success(
        data={
            'filename': filename,
            'filepath': filepath,
            'format': export_format
        },
        message=f'Results successfully exported to {filename}'
    ))


# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


# Helper function to convert matplotlib figure to base64 image for web display
def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string for embedding in HTML."""
    img_buffer = io.BytesIO()
    fig.savefig(img_buffer, format='png', bbox_inches='tight')
    img_buffer.seek(0)
    img_str = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    return img_str


if __name__ == '__main__':
    # Configure app before running
    app.config['JSON_SORT_KEYS'] = False  # Preserve key order in JSON responses
    app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False  # Disable pretty printing for production
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
    
    # Log application startup
    logger.info("Starting Neural Network Optimization Web Application")
    
    # Run the application
    app.run(debug=True, host='0.0.0.0', port=5000)

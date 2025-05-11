"""
Neural Network Optimization Project - Main Entry Point

This script provides a command-line interface for running experiments
with Genetic Algorithms and Particle Swarm Optimization for neural network optimization.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import shutil
import glob

from utils.data_loader import DataLoader
from experiments.weight_optimization import WeightOptimizationExperiment
from experiments.feature_selection import FeatureSelectionExperiment
from experiments.hyperparameter_tuning import HyperparameterTuningExperiment
from config.parameters import (
    GA_PARAMS, PSO_PARAMS, NN_PARAMS, FEATURE_SELECTION_PARAMS,
    HYPERPARAMETER_RANGES, EXPERIMENT_SETTINGS, ConfigManager
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Neural Network Optimization using GA and PSO')
    
    # Dataset options
    parser.add_argument('--dataset', type=str, help='Path to dataset file (CSV)')
    parser.add_argument('--target', type=str, help='Name of target column in dataset')
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic dataset')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples for synthetic dataset')
    parser.add_argument('--num_features', type=int, default=10, help='Number of features for synthetic dataset')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes for synthetic dataset')
    
    # Experiment options
    parser.add_argument('--experiment', type=str, choices=['weight', 'feature', 'hyperparameter'], 
                      default='weight', help='Type of experiment to run')
    parser.add_argument('--algorithm', type=str, choices=['ga', 'pso', 'both'], 
                      default='both', help='Algorithm to use')
    
    # Output options
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--save_model', action='store_true', help='Save the best model')
    parser.add_argument('--verbose', action='store_true', help='Show verbose output')
    
    # Configuration options
    parser.add_argument('--ga_config', type=str, help='Name of GA configuration to load')
    parser.add_argument('--pso_config', type=str, help='Name of PSO configuration to load')
    parser.add_argument('--nn_config', type=str, help='Name of NN configuration to load')
    
    return parser.parse_args()


def load_data(args):
    """Load dataset based on command-line arguments."""
    data_loader = DataLoader(random_state=EXPERIMENT_SETTINGS['random_state'])
    
    if args.synthetic:
        print(f"Generating synthetic dataset with {args.num_samples} samples and {args.num_features} features...")
        X, y = data_loader.generate_synthetic_data(
            num_samples=args.num_samples,
            num_features=args.num_features,
            num_classes=args.num_classes
        )
    elif args.dataset:
        if not os.path.exists(args.dataset):
            raise FileNotFoundError(f"Dataset file not found: {args.dataset}")
        
        if not args.target:
            raise ValueError("Target column name must be specified with --target")
        
        print(f"Loading dataset from {args.dataset}...")
        X, y = data_loader.load_csv(args.dataset, target_column=args.target)
    else:
        raise ValueError("Either --dataset or --synthetic must be specified")
    
    # Preprocess the data
    data = data_loader.preprocess_data(
        X, y, 
        test_size=EXPERIMENT_SETTINGS['test_size'],
        validation_size=EXPERIMENT_SETTINGS['validation_size'],
        scale_method=EXPERIMENT_SETTINGS['scaling_method']
    )
    
    print(f"Dataset loaded with {X.shape[0]} samples and {X.shape[1]} features")
    
    return data_loader, data


def run_weight_optimization(data_loader, data, args):
    """Run weight optimization experiment."""
    print("\n=== Running Weight Optimization Experiment ===")
    
    # Load configurations
    ga_params = GA_PARAMS.copy()
    pso_params = PSO_PARAMS.copy()
    
    if args.ga_config:
        ga_config = ConfigManager.load_config(args.ga_config)
        if ga_config:
            ga_params.update(ga_config)
    
    if args.pso_config:
        pso_config = ConfigManager.load_config(args.pso_config)
        if pso_config:
            pso_params.update(pso_config)
    
    # Create experiment
    experiment = WeightOptimizationExperiment(
        X_train=data['X_train'],
        y_train=data['y_train'],
        X_val=data['X_val'],
        y_val=data['y_val'],
        X_test=data['X_test'],
        y_test=data['y_test'],
        input_dim=data_loader.num_features,
        output_dim=data_loader.num_classes if data_loader.num_classes > 2 else 1,
        hidden_layers=NN_PARAMS['hidden_layers'],
        ga_params=ga_params,
        pso_params=pso_params
    )
    
    # Run algorithms
    ga_results = None
    pso_results = None
    
    if args.algorithm in ['ga', 'both']:
        print("\nRunning Genetic Algorithm...")
        ga_results = experiment.run_ga_optimization(verbose=args.verbose)
        print(f"GA Best Fitness: {ga_results['best_fitness']:.4f}")
        print(f"GA Test Accuracy: {ga_results['test_metrics']['accuracy']:.4f}")
    
    if args.algorithm in ['pso', 'both']:
        print("\nRunning Particle Swarm Optimization...")
        pso_results = experiment.run_pso_optimization(verbose=args.verbose)
        print(f"PSO Best Fitness: {pso_results['best_fitness']:.4f}")
        print(f"PSO Test Accuracy: {pso_results['test_metrics']['accuracy']:.4f}")
    
    # Compare algorithms if both were run
    if args.algorithm == 'both':
        print("\nComparing Algorithms...")
        comparison = experiment.compare_algorithms()
        
        # Save plots
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(args.output_dir, exist_ok=True)
        
        convergence_plot_path = os.path.join(args.output_dir, f'weight_optimization_convergence_{timestamp}.png')
        comparison_plot_path = os.path.join(args.output_dir, f'weight_optimization_comparison_{timestamp}.png')
        
        experiment.visualizer.save_plot(
            comparison['convergence_plot'],
            convergence_plot_path
        )
        
        experiment.visualizer.save_plot(
            comparison['comparison_plot'],
            comparison_plot_path
        )
        
        print(f"Plots saved to {args.output_dir}")
    
    # Save models if requested
    if args.save_model:
        os.makedirs(os.path.join(args.output_dir, 'models'), exist_ok=True)
        
        if ga_results:
            model_path = os.path.join(args.output_dir, 'models', f'ga_model_{timestamp}.h5')
            experiment.nn.save_model(model_path)
            print(f"GA model saved to {model_path}")
        
        if pso_results:
            model_path = os.path.join(args.output_dir, 'models', f'pso_model_{timestamp}.h5')
            experiment.nn.save_model(model_path)
            print(f"PSO model saved to {model_path}")
    
    return experiment


def run_feature_selection(data_loader, data, args):
    """Run feature selection experiment."""
    print("\n=== Running Feature Selection Experiment ===")
    
    # Load configurations
    ga_params = GA_PARAMS.copy()
    pso_params = PSO_PARAMS.copy()
    feature_selection_params = FEATURE_SELECTION_PARAMS.copy()
    
    if args.ga_config:
        ga_config = ConfigManager.load_config(args.ga_config)
        if ga_config:
            ga_params.update(ga_config)
    
    if args.pso_config:
        pso_config = ConfigManager.load_config(args.pso_config)
        if pso_config:
            pso_params.update(pso_config)
    
    # Create experiment
    experiment = FeatureSelectionExperiment(
        X=data_loader.X,
        y=data_loader.y,
        feature_names=data_loader.feature_names,
        test_size=EXPERIMENT_SETTINGS['test_size'],
        validation_size=EXPERIMENT_SETTINGS['validation_size'],
        ga_params=ga_params,
        pso_params=pso_params,
        feature_selection_params=feature_selection_params
    )
    
    # Run algorithms
    ga_results = None
    pso_results = None
    
    if args.algorithm in ['ga', 'both']:
        print("\nRunning Genetic Algorithm...")
        ga_results = experiment.run_ga_feature_selection(verbose=args.verbose)
        print(f"GA Best Fitness: {ga_results['best_fitness']:.4f}")
        print(f"GA Test Accuracy: {ga_results['test_metrics']['accuracy']:.4f}")
        print(f"GA Selected Features ({ga_results['num_selected_features']}): {ga_results['selected_feature_names']}")
    
    if args.algorithm in ['pso', 'both']:
        print("\nRunning Particle Swarm Optimization...")
        pso_results = experiment.run_pso_feature_selection(verbose=args.verbose)
        print(f"PSO Best Fitness: {pso_results['best_fitness']:.4f}")
        print(f"PSO Test Accuracy: {pso_results['test_metrics']['accuracy']:.4f}")
        print(f"PSO Selected Features ({pso_results['num_selected_features']}): {pso_results['selected_feature_names']}")
    
    # Compare algorithms if both were run
    if args.algorithm == 'both':
        print("\nComparing Algorithms...")
        comparison = experiment.compare_algorithms()
        
        # Save plots
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(args.output_dir, exist_ok=True)
        
        convergence_plot_path = os.path.join(args.output_dir, f'feature_selection_convergence_{timestamp}.png')
        comparison_plot_path = os.path.join(args.output_dir, f'feature_selection_comparison_{timestamp}.png')
        ga_feature_importance_path = os.path.join(args.output_dir, f'ga_feature_importance_{timestamp}.png')
        pso_feature_importance_path = os.path.join(args.output_dir, f'pso_feature_importance_{timestamp}.png')
        
        experiment.visualizer.save_plot(
            comparison['convergence_plot'],
            convergence_plot_path
        )
        
        experiment.visualizer.save_plot(
            comparison['comparison_plot'],
            comparison_plot_path
        )
        
        experiment.visualizer.save_plot(
            comparison['ga_feature_importance_plot'],
            ga_feature_importance_path
        )
        
        experiment.visualizer.save_plot(
            comparison['pso_feature_importance_plot'],
            pso_feature_importance_path
        )
        
        print(f"Plots saved to {args.output_dir}")
        
        # Print common features
        common_features = comparison['common_features']
        print(f"\nCommon Features Selected by Both Algorithms ({len(common_features)}): {common_features}")
    
    return experiment


def run_hyperparameter_tuning(data_loader, data, args):
    """Run hyperparameter tuning experiment."""
    print("\n=== Running Hyperparameter Tuning Experiment ===")
    
    # Load configurations
    ga_params = GA_PARAMS.copy()
    pso_params = PSO_PARAMS.copy()
    hyperparameter_ranges = HYPERPARAMETER_RANGES.copy()
    
    if args.ga_config:
        ga_config = ConfigManager.load_config(args.ga_config)
        if ga_config:
            ga_params.update(ga_config)
    
    if args.pso_config:
        pso_config = ConfigManager.load_config(args.pso_config)
        if pso_config:
            pso_params.update(pso_config)
    
    # Create experiment
    experiment = HyperparameterTuningExperiment(
        X=data_loader.X,
        y=data_loader.y,
        test_size=EXPERIMENT_SETTINGS['test_size'],
        validation_size=EXPERIMENT_SETTINGS['validation_size'],
        ga_params=ga_params,
        pso_params=pso_params,
        hyperparameter_ranges=hyperparameter_ranges
    )
    
    # Run algorithms
    ga_results = None
    pso_results = None
    
    if args.algorithm in ['ga', 'both']:
        print("\nRunning Genetic Algorithm...")
        ga_results = experiment.run_ga_tuning(verbose=args.verbose)
        print(f"GA Best Fitness: {ga_results['best_fitness']:.4f}")
        print(f"GA Test Accuracy: {ga_results['test_metrics']['accuracy']:.4f}")
        print(f"GA Best Hyperparameters: {ga_results['best_hyperparameters']}")
    
    if args.algorithm in ['pso', 'both']:
        print("\nRunning Particle Swarm Optimization...")
        pso_results = experiment.run_pso_tuning(verbose=args.verbose)
        print(f"PSO Best Fitness: {pso_results['best_fitness']:.4f}")
        print(f"PSO Test Accuracy: {pso_results['test_metrics']['accuracy']:.4f}")
        print(f"PSO Best Hyperparameters: {pso_results['best_hyperparameters']}")
    
    # Compare algorithms if both were run
    if args.algorithm == 'both':
        print("\nComparing Algorithms...")
        comparison = experiment.compare_algorithms()
        
        # Save plots
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(args.output_dir, exist_ok=True)
        
        convergence_plot_path = os.path.join(args.output_dir, f'hyperparameter_tuning_convergence_{timestamp}.png')
        comparison_plot_path = os.path.join(args.output_dir, f'hyperparameter_tuning_comparison_{timestamp}.png')
        hyperparameter_heatmap_path = os.path.join(args.output_dir, f'hyperparameter_heatmap_{timestamp}.png')
        
        experiment.visualizer.save_plot(
            comparison['convergence_plot'],
            convergence_plot_path
        )
        
        experiment.visualizer.save_plot(
            comparison['comparison_plot'],
            comparison_plot_path
        )
        
        experiment.visualizer.save_plot(
            comparison['hyperparameter_heatmap'],
            hyperparameter_heatmap_path
        )
        
        print(f"Plots saved to {args.output_dir}")
    
    # Save models if requested
    if args.save_model:
        os.makedirs(os.path.join(args.output_dir, 'models'), exist_ok=True)
        
        if ga_results:
            model_path = os.path.join(args.output_dir, 'models', f'ga_hyperparameter_model_{timestamp}.h5')
            ga_results['model'].save_model(model_path)
            print(f"GA model saved to {model_path}")
        
        if pso_results:
            model_path = os.path.join(args.output_dir, 'models', f'pso_hyperparameter_model_{timestamp}.h5')
            pso_results['model'].save_model(model_path)
            print(f"PSO model saved to {model_path}")
    
    return experiment


def main():
    """Main entry point."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Load data
        data_loader, data = load_data(args)
        
        # Run the selected experiment
        if args.experiment == 'weight':
            experiment = run_weight_optimization(data_loader, data, args)
        elif args.experiment == 'feature':
            experiment = run_feature_selection(data_loader, data, args)
        elif args.experiment == 'hyperparameter':
            experiment = run_hyperparameter_tuning(data_loader, data, args)
        
        print("\nExperiment completed successfully!")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


def clean_pycache_directories():
    """Remove all __pycache__ directories and .pyc files from the project."""
    print("Cleaning __pycache__ directories and .pyc files...")
    
    # Get the project root directory
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Find and remove all __pycache__ directories
    pycache_dirs = glob.glob(os.path.join(root_dir, '**/__pycache__'), recursive=True)
    for pycache_dir in pycache_dirs:
        if os.path.exists(pycache_dir) and os.path.isdir(pycache_dir):
            print(f"Removing {pycache_dir}")
            shutil.rmtree(pycache_dir)
    
    # Find and remove all .pyc files
    pyc_files = glob.glob(os.path.join(root_dir, '**/*.pyc'), recursive=True)
    for pyc_file in pyc_files:
        if os.path.exists(pyc_file) and os.path.isfile(pyc_file):
            print(f"Removing {pyc_file}")
            os.remove(pyc_file)
    
    print("Cleaning completed.")


def run_web_app():
    """Run the web application."""
    import subprocess
    import webbrowser
    import threading
    import time
    import os
    from app import app
    
    # Define the host and port
    host = '127.0.0.1'
    port = 5000
    url = f'http://{host}:{port}'
    
    # Function to open browser after a short delay
    def open_browser():
        time.sleep(1.5)  # Wait for the server to start
        webbrowser.open(url)
        print(f"\nWeb application opened in browser at {url}")
    
    # Start browser thread
    threading.Thread(target=open_browser).start()
    
    # Run the Flask app
    print("\nStarting Neural Network Optimization web application...")
    print("Press Ctrl+C to stop the server")
    app.run(host=host, port=port, debug=False)

if __name__ == '__main__':
    import sys
    
    # Clean __pycache__ directories before starting
    clean_pycache_directories()
    
    # Check if any command-line arguments are provided
    if len(sys.argv) > 1:
        # If arguments are provided, run the CLI version
        exit(main())
    else:
        # If no arguments are provided, run the web application
        run_web_app()

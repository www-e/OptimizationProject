"""
Visualization utilities for analyzing optimization results.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns


class OptimizationVisualizer:
    """
    Visualization tools for optimization algorithms.
    """
    
    def __init__(self, figsize=(12, 8), style='whitegrid'):
        """
        Initialize the visualizer.
        
        Args:
            figsize: Figure size for plots
            style: Seaborn style for plots
        """
        self.figsize = figsize
        sns.set_style(style)
        
        # Set color palette for consistency
        self.colors = sns.color_palette("deep", 10)
    
    def plot_convergence(self, histories, labels=None, title='Convergence Curve'):
        """
        Plot convergence curves for optimization algorithms.
        
        Args:
            histories: List of history dictionaries from optimization algorithms
            labels: List of labels for each history
            title: Plot title
        """
        plt.figure(figsize=self.figsize)
        
        if labels is None:
            labels = [f'Algorithm {i+1}' for i in range(len(histories))]
        
        for i, history in enumerate(histories):
            plt.plot(history['best_fitness'], label=f'{labels[i]} - Best')
            plt.plot(history['avg_fitness'], label=f'{labels[i]} - Average', linestyle='--', alpha=0.7)
        
        plt.title(title, fontsize=16)
        plt.xlabel('Iteration / Generation', fontsize=14)
        plt.ylabel('Fitness Score', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_parameter_impact(self, parameter_values, fitness_scores, 
                             parameter_name, algorithm_name):
        """
        Plot the impact of a parameter on fitness.
        
        Args:
            parameter_values: List of parameter values
            fitness_scores: List of corresponding fitness scores
            parameter_name: Name of the parameter
            algorithm_name: Name of the algorithm
        """
        plt.figure(figsize=self.figsize)
        
        plt.plot(parameter_values, fitness_scores, 'o-', linewidth=2, markersize=8)
        
        plt.title(f'Impact of {parameter_name} on {algorithm_name} Performance', fontsize=16)
        plt.xlabel(parameter_name, fontsize=14)
        plt.ylabel('Best Fitness Score', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # Set x-axis to only show the parameter values we tested
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_algorithm_comparison(self, metrics, algorithm_names, metric_names=None):
        """
        Compare performance metrics between algorithms.
        
        Args:
            metrics: 2D array of metrics (algorithms x metrics)
            algorithm_names: List of algorithm names
            metric_names: List of metric names
        """
        if metric_names is None:
            metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Training Time']
        
        # Ensure metrics is a 2D array
        metrics = np.array(metrics)
        
        # Create a figure
        plt.figure(figsize=self.figsize)
        
        # Set width of bars
        bar_width = 0.15
        index = np.arange(len(metric_names))
        
        # Plot bars
        for i, alg_name in enumerate(algorithm_names):
            plt.bar(index + i * bar_width, metrics[i], bar_width, label=alg_name)
        
        # Add labels and title
        plt.xlabel('Metrics', fontsize=14)
        plt.ylabel('Score', fontsize=14)
        plt.title('Performance Comparison Between Algorithms', fontsize=16)
        plt.xticks(index + bar_width * (len(algorithm_names) - 1) / 2, metric_names)
        plt.legend()
        
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_feature_importance(self, feature_mask, feature_names=None, title='Feature Importance'):
        """
        Visualize feature importance based on selection mask.
        
        Args:
            feature_mask: Binary or continuous mask of feature importance
            feature_names: List of feature names
            title: Plot title
        """
        # If feature names not provided, generate them
        if feature_names is None:
            feature_names = [f'Feature {i+1}' for i in range(len(feature_mask))]
        
        # Sort features by importance
        sorted_indices = np.argsort(feature_mask)[::-1]
        sorted_mask = feature_mask[sorted_indices]
        sorted_names = [feature_names[i] for i in sorted_indices]
        
        # Create plot
        plt.figure(figsize=self.figsize)
        
        # Bar plot
        plt.bar(range(len(sorted_mask)), sorted_mask)
        
        # Add labels
        plt.xticks(range(len(sorted_mask)), sorted_names, rotation=45, ha='right')
        plt.xlabel('Features', fontsize=14)
        plt.ylabel('Importance Score', fontsize=14)
        plt.title(title, fontsize=16)
        
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_hyperparameter_heatmap(self, param1_values, param2_values, fitness_matrix,
                                  param1_name, param2_name, algorithm_name):
        """
        Plot heatmap of fitness for two hyperparameters.
        
        Args:
            param1_values: List of values for first parameter
            param2_values: List of values for second parameter
            fitness_matrix: 2D matrix of fitness values
            param1_name: Name of first parameter
            param2_name: Name of second parameter
            algorithm_name: Name of algorithm
        """
        plt.figure(figsize=self.figsize)
        
        # Create heatmap
        sns.heatmap(fitness_matrix, annot=True, fmt='.3f', cmap='viridis',
                   xticklabels=param2_values, yticklabels=param1_values)
        
        # Add labels
        plt.xlabel(param2_name, fontsize=14)
        plt.ylabel(param1_name, fontsize=14)
        plt.title(f'{algorithm_name} Performance: {param1_name} vs {param2_name}', fontsize=16)
        
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_hyperparameter_comparison(self, ga_hyperparams, pso_hyperparams, title='Best Hyperparameters Comparison'):
        """
        Compare hyperparameters found by GA and PSO.
        
        Args:
            ga_hyperparams: Dictionary of hyperparameters found by GA
            pso_hyperparams: Dictionary of hyperparameters found by PSO
            title: Plot title
        
        Returns:
            Figure object
        """
        # Handle potential None values or empty dictionaries
        if ga_hyperparams is None:
            ga_hyperparams = {}
        if pso_hyperparams is None:
            pso_hyperparams = {}
        
        # Check for 'best_hyperparameters' vs 'best_hyperparams' key inconsistency
        if 'best_hyperparameters' in ga_hyperparams and isinstance(ga_hyperparams['best_hyperparameters'], dict):
            ga_hyperparams = ga_hyperparams['best_hyperparameters']
        elif 'best_hyperparams' in ga_hyperparams and isinstance(ga_hyperparams['best_hyperparams'], dict):
            ga_hyperparams = ga_hyperparams['best_hyperparams']
        
        if 'best_hyperparameters' in pso_hyperparams and isinstance(pso_hyperparams['best_hyperparameters'], dict):
            pso_hyperparams = pso_hyperparams['best_hyperparameters']
        elif 'best_hyperparams' in pso_hyperparams and isinstance(pso_hyperparams['best_hyperparams'], dict):
            pso_hyperparams = pso_hyperparams['best_hyperparams']
    
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Combine all hyperparameters
        all_params = set(list(ga_hyperparams.keys()) + list(pso_hyperparams.keys()))
        
        # Filter out non-hyperparameter keys
        exclude_keys = ['model', 'history', 'fitness']
        param_names = sorted([p for p in all_params if p not in exclude_keys])
        
        if not param_names:  # If no valid parameters found
            ax.text(0.5, 0.5, 'No hyperparameter data available', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=14)
            return fig
        
        # Set positions for bars
        x = np.arange(len(param_names))
        width = 0.35
        
        # Prepare data for plotting
        ga_values = []
        pso_values = []
        ga_display_values = []
        pso_display_values = []
        
        for param in param_names:
            # For GA
            if param in ga_hyperparams:
                value = ga_hyperparams[param]
                # Store original value for display
                ga_display_values.append(str(value))
                # Convert to numeric for plotting
                if isinstance(value, (int, float)):
                    ga_values.append(float(value))
                elif isinstance(value, list):
                    # For list values (like hidden layers), use the length
                    ga_values.append(len(value))
                else:
                    # For non-numeric values, use a hash function to get a consistent numeric value
                    ga_values.append(abs(hash(str(value)) % 10))
            else:
                ga_values.append(0)
                ga_display_values.append('N/A')
        
            # For PSO
            if param in pso_hyperparams:
                value = pso_hyperparams[param]
                # Store original value for display
                pso_display_values.append(str(value))
                # Convert to numeric for plotting
                if isinstance(value, (int, float)):
                    pso_values.append(float(value))
                elif isinstance(value, list):
                    # For list values (like hidden layers), use the length
                    pso_values.append(len(value))
                else:
                    # For non-numeric values, use a hash function to get a consistent numeric value
                    pso_values.append(abs(hash(str(value)) % 10))
            else:
                pso_values.append(0)
                pso_display_values.append('N/A')
    
        # Create bars
        ax.bar(x - width/2, ga_values, width, label='GA', color=self.colors[0])
        ax.bar(x + width/2, pso_values, width, label='PSO', color=self.colors[1])
    
        # Add labels and title
        ax.set_xlabel('Hyperparameters', fontsize=14)
        ax.set_ylabel('Value', fontsize=14)
        ax.set_title(title, fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(param_names, rotation=45, ha='right')
        ax.legend()
    
        # Add a table below the chart with actual values
        table_data = []
        for i, param in enumerate(param_names):
            table_data.append([param, ga_display_values[i], pso_display_values[i]])
    
        # Create the table
        table = plt.table(cellText=table_data,
                          colLabels=['Parameter', 'GA Value', 'PSO Value'],
                          loc='bottom',
                          bbox=[0, -0.5, 1, 0.3])
    
        # Adjust layout to make room for the table
        plt.subplots_adjust(bottom=0.3)
    
        plt.tight_layout(rect=[0, 0.3, 1, 1])
    
        return fig
    
    def plot_comparison(self, comparison_data, title='Algorithm Comparison'):
        """
        Create a comparison plot for algorithm performance metrics.
        
        Args:
            comparison_data: Dictionary with algorithm names as keys and dictionaries of metrics as values
            title: Plot title
        
        Returns:
            Figure object
        """
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Get algorithm names and metrics
        alg_names = list(comparison_data.keys())
        metrics = list(comparison_data[alg_names[0]].keys())
        
        # Set positions for bars
        x = np.arange(len(metrics))
        width = 0.8 / len(alg_names)
        
        # Create bars for each algorithm
        for i, alg in enumerate(alg_names):
            values = [comparison_data[alg][metric] for metric in metrics]
            ax.bar(x + (i - len(alg_names)/2 + 0.5) * width, values, width, label=alg, color=self.colors[i])
        
        # Add labels and title
        ax.set_xlabel('Metrics', fontsize=14)
        ax.set_ylabel('Value', fontsize=14)
        ax.set_title(title, fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.legend()
        
        plt.tight_layout()
        
        return fig
    
    def save_plot(self, fig, filename):
        """
        Save a plot to a file.
        
        Args:
            fig: Figure object
            filename: Output filename
        """
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)

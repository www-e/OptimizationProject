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
    
    def save_plot(self, fig, filename):
        """
        Save a plot to a file.
        
        Args:
            fig: Figure object
            filename: Output filename
        """
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)

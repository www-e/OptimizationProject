# Neural Network Optimization Project - Testing Guide

This guide provides detailed instructions on how to test the Neural Network Optimization project, with a focus on the frontend functionality and specific test scenarios for the Genetic Algorithm (GA) and Particle Swarm Optimization (PSO) implementations.

## Table of Contents

1. [Installation](#installation)
2. [Project Structure](#project-structure)
3. [Frontend Testing](#frontend-testing)
   - [Home Page](#home-page)
   - [Dataset Management](#dataset-management)
   - [Algorithm Configuration](#algorithm-configuration)
   - [Experiment Setup](#experiment-setup)
   - [Results Visualization](#results-visualization)
4. [Test Scenarios](#test-scenarios)
   - [Weight Optimization Tests](#weight-optimization-tests)
   - [Feature Selection Tests](#feature-selection-tests)
   - [Hyperparameter Tuning Tests](#hyperparameter-tuning-tests)
5. [Running Experiments](#running-experiments)
6. [Comparing GA and PSO](#comparing-ga-and-pso)
7. [Customizing Parameters](#customizing-parameters)
8. [Troubleshooting](#troubleshooting)

## Installation

1. Clone the repository or extract the project files to your desired location.

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:
   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Verify the installation by running a simple test:
   ```bash
   python -c "import tensorflow as tf; print(tf.__version__)"
   ```

## Project Structure

The project is organized into the following directories:

- `algorithms/`: Contains the implementation of GA and PSO algorithms
- `models/`: Contains the neural network model implementation
- `utils/`: Contains utility functions for data loading and visualization
- `experiments/`: Contains experiment implementations for different optimization tasks
- `config/`: Contains configuration parameters for the algorithms and experiments
- `static/`: Contains static files for the web interface (CSS, JavaScript, images)
- `templates/`: Contains HTML templates for the web interface
- `results/`: Directory where experiment results are saved
- `saved_models/`: Directory where optimized models are saved

## Frontend Testing

This section provides detailed instructions for testing the web interface of the Neural Network Optimization project. The application features a modern, futuristic UI design with responsive components.

### Starting the Application

To start the web application for testing:

```bash
python main.py
```

The application will automatically clean all __pycache__ directories and then start the web server. A browser window should automatically open at `http://localhost:5000`.

### Home Page

**Test Steps:**

1. Verify that the home page loads correctly with the following elements:
   - Navigation bar with links to all sections
   - Hero section with application title and description
   - Feature cards highlighting key capabilities
   - Footer with navigation links

2. Test responsive design:
   - Resize browser window to different sizes (desktop, tablet, mobile)
   - Verify that all elements adjust appropriately
   - Check that the navigation collapses to a hamburger menu on small screens

3. Test navigation:
   - Click each navigation link and verify it leads to the correct page
   - Test the dropdown menus for Algorithms and Experiments
   - Verify that the active page is highlighted in the navigation

**Expected Results:**
- All elements should display with proper styling and spacing
- Navigation should work smoothly with correct highlighting
- Page should be responsive and adapt to different screen sizes

### Dataset Management

**Test Steps:**

1. Navigate to the Dataset page

2. Test synthetic data generation:
   - Enter values for number of samples (e.g., 1000)
   - Enter values for number of features (e.g., 10)
   - Select number of classes (e.g., 2 for binary classification)
   - Click "Generate Synthetic Data"
   - Verify that success message appears
   - Check that dataset statistics are displayed

3. Test dataset upload (if available):
   - Prepare a CSV file with labeled data
   - Click "Upload Dataset"
   - Select the CSV file
   - Verify that the upload completes successfully
   - Check that dataset statistics are displayed

4. Test dataset visualization:
   - Click on visualization options (if available)
   - Verify that charts and graphs display correctly

**Expected Results:**
- Synthetic data should be generated without errors
- Dataset statistics should display correctly
- Upload functionality should work with proper validation
- Visualizations should render accurately

### Algorithm Configuration

**Test Steps:**

1. Navigate to the Algorithms page

2. Test GA configuration:
   - Click on the GA tab
   - Modify parameters (population size, generations, etc.)
   - Click "Apply Configuration"
   - Verify that success toast notification appears
   - Click "Save Configuration"
   - Enter a name for the configuration
   - Verify that the configuration is saved successfully

3. Test PSO configuration:
   - Click on the PSO tab
   - Modify parameters (particles, iterations, etc.)
   - Click "Apply Configuration"
   - Verify that success toast notification appears
   - Click "Save Configuration"
   - Enter a name for the configuration
   - Verify that the configuration is saved successfully

4. Test loading configurations:
   - Locate a saved configuration in the list
   - Click "Load"
   - Verify that the form fields are populated with the saved values
   - Click "Apply Configuration"
   - Verify that success toast notification appears

**Expected Results:**
- Parameter forms should display with proper validation
- Configuration saving and loading should work correctly
- Toast notifications should appear with appropriate messages
- All buttons and controls should be styled consistently

### Experiment Setup

**Test Steps:**

1. Navigate to the Experiments page

2. Test experiment type selection:
   - Select each experiment type (weight optimization, feature selection, hyperparameter tuning)
   - Verify that the appropriate parameter sections appear/disappear
   - Check that the URL updates with the correct parameters

3. Test algorithm selection:
   - Select different algorithms (GA, PSO, both)
   - Verify that the appropriate options are enabled/disabled

4. Test experiment execution:
   - Configure an experiment (preferably with small parameter values for quick testing)
   - Click "Run Experiment"
   - Verify that the progress bar appears and updates
   - Check that status messages change appropriately
   - Wait for the experiment to complete
   - Verify that success message appears
   - Check that you are redirected to the results page

5. Test experiment cancellation:
   - Start an experiment
   - Click "Cancel Experiment"
   - Confirm the cancellation
   - Verify that the experiment form reappears

**Expected Results:**
- Experiment type selection should update the form correctly
- Progress tracking should work with visual feedback
- Cancellation should work without errors
- Redirection to results should happen after completion

### Results Visualization

**Test Steps:**

1. Navigate to the Results page

2. Test results loading:
   - If results exist, verify they load correctly
   - Check that charts and graphs render properly
   - Verify that metrics are displayed accurately

3. Test comparison features:
   - If multiple algorithm results exist, test the comparison view
   - Verify that comparison charts display correctly
   - Check that metrics are compared accurately

4. Test export functionality (if available):
   - Click export buttons
   - Verify that data or images are downloaded correctly

**Expected Results:**
- Results should display with proper formatting and styling
- Charts should render with correct data and labels
- Comparison views should highlight differences clearly
- Export functionality should work without errors

## Test Scenarios

This section provides specific test scenarios for each optimization type to verify that the algorithms are functioning correctly. These scenarios can be used to validate both the backend implementation and the frontend functionality.

### Weight Optimization Tests

#### Test Scenario 1: Simple Binary Classification

**Setup:**
- Generate synthetic data with 500 samples, 5 features, 2 classes
- Configure a small neural network (input: 5, hidden: [8, 4], output: 1)
- Set GA parameters: population_size=20, num_generations=20
- Set PSO parameters: num_particles=20, num_iterations=20

**Test Steps:**
1. Navigate to the Experiments page
2. Select "Weight Optimization" experiment type
3. Select "Both" for algorithm
4. Configure the parameters as specified above
5. Run the experiment

**Expected Results:**
- Both algorithms should complete without errors
- Convergence plots should show fitness improving over generations/iterations
- Final accuracy should be above random chance (>50%)
- GA and PSO should achieve comparable performance

#### Test Scenario 2: Comparing GA and PSO Convergence

**Setup:**
- Generate synthetic data with 1000 samples, 10 features, 2 classes
- Configure a medium neural network (input: 10, hidden: [16, 8], output: 1)
- Set GA parameters: population_size=50, num_generations=50
- Set PSO parameters: num_particles=50, num_iterations=50

**Test Steps:**
1. Navigate to the Experiments page
2. Select "Weight Optimization" experiment type
3. Select "Both" for algorithm
4. Configure the parameters as specified above
5. Run the experiment

**Expected Results:**
- Both algorithms should complete without errors
- Convergence plots should show different convergence patterns for GA and PSO
- PSO typically converges faster initially, while GA may achieve better final results
- The comparison view should highlight the differences between the algorithms

### Feature Selection Tests

#### Test Scenario 1: Identifying Important Features

**Setup:**
- Generate synthetic data with 1000 samples, 20 features, 2 classes
- Set GA parameters: population_size=30, num_generations=30
- Set PSO parameters: num_particles=30, num_iterations=30
- Set feature selection parameters: min_features=5, max_features=15

**Test Steps:**
1. Navigate to the Experiments page
2. Select "Feature Selection" experiment type
3. Select "Both" for algorithm
4. Configure the parameters as specified above
5. Run the experiment

**Expected Results:**
- Both algorithms should complete without errors
- Feature importance visualization should show which features were selected
- Selected features should be fewer than the original set (between 5-15)
- Model with selected features should perform similarly or better than with all features

#### Test Scenario 2: Minimal Feature Set

**Setup:**
- Generate synthetic data with 1000 samples, 30 features, 2 classes
- Set GA parameters: population_size=40, num_generations=40
- Set PSO parameters: num_particles=40, num_iterations=40
- Set feature selection parameters: min_features=3, max_features=10

**Test Steps:**
1. Navigate to the Experiments page
2. Select "Feature Selection" experiment type
3. Select "Both" for algorithm
4. Configure the parameters as specified above
5. Run the experiment

**Expected Results:**
- Both algorithms should identify a minimal set of features
- GA and PSO might select different feature subsets
- Performance metrics should show how accuracy changes with feature reduction
- The visualization should clearly show which features contribute most to the model

### Hyperparameter Tuning Tests

#### Test Scenario 1: Basic Hyperparameter Optimization

**Setup:**
- Generate synthetic data with 800 samples, 10 features, 2 classes
- Set GA parameters: population_size=25, num_generations=25
- Set PSO parameters: num_particles=25, num_iterations=25
- Enable tuning for learning_rate, batch_size, and activation function

**Test Steps:**
1. Navigate to the Experiments page
2. Select "Hyperparameter Tuning" experiment type
3. Select "Both" for algorithm
4. Configure the parameters as specified above
5. Run the experiment

**Expected Results:**
- Both algorithms should complete without errors
- Optimal hyperparameters should be identified and displayed
- Performance with optimized hyperparameters should be better than with default values
- Hyperparameter impact visualization should show the effect of different parameter values

#### Test Scenario 2: Complex Network Architecture Tuning

**Setup:**
- Generate synthetic data with 1000 samples, 15 features, 3 classes
- Set GA parameters: population_size=30, num_generations=30
- Set PSO parameters: num_particles=30, num_iterations=30
- Enable tuning for hidden_layers, activation function, and learning_rate

**Test Steps:**
1. Navigate to the Experiments page
2. Select "Hyperparameter Tuning" experiment type
3. Select "Both" for algorithm
4. Configure the parameters as specified above
5. Run the experiment

**Expected Results:**
- Both algorithms should complete without errors
- Optimal network architecture should be identified
- Performance with optimized architecture should be better than with default architecture
- Visualization should show the relationship between network complexity and performance

## Running Experiments

### Weight Optimization

Weight optimization involves using GA or PSO to directly optimize the weights of a neural network instead of using gradient-based methods like backpropagation.

To run a weight optimization experiment:

1. Prepare your dataset or use the built-in synthetic data generator:

```python
from utils.data_loader import DataLoader

# Create a data loader
data_loader = DataLoader(random_state=42)

# Generate synthetic data
X, y = data_loader.generate_synthetic_data(
    num_samples=1000,
    num_features=10,
    num_classes=2
)

# Preprocess the data
data = data_loader.preprocess_data(X, y, test_size=0.2, validation_size=0.1)
```

2. Run the weight optimization experiment:

```python
from experiments.weight_optimization import WeightOptimizationExperiment
from config.parameters_enhanced import GA_PARAMS, PSO_PARAMS

# Create the experiment
experiment = WeightOptimizationExperiment(
    X_train=data['X_train'],
    y_train=data['y_train'],
    X_val=data['X_val'],
    y_val=data['y_val'],
    X_test=data['X_test'],
    y_test=data['y_test'],
    input_dim=data_loader.num_features,
    output_dim=data_loader.num_classes if data_loader.num_classes > 2 else 1,
    hidden_layers=[32, 16],
    ga_params=GA_PARAMS,
    pso_params=PSO_PARAMS
)

# Run GA optimization
ga_results = experiment.run_ga_optimization(verbose=True)

# Run PSO optimization
pso_results = experiment.run_pso_optimization(verbose=True)

# Compare the results
comparison = experiment.compare_algorithms()

# Save the plots
comparison['convergence_plot'].savefig('results/weight_optimization_convergence.png')
comparison['comparison_plot'].savefig('results/weight_optimization_comparison.png')
```

### Feature Selection

Feature selection involves using GA or PSO to select the most important features for the model, which can improve performance and reduce complexity.

To run a feature selection experiment:

```python
from experiments.feature_selection import FeatureSelectionExperiment
from config.parameters_enhanced import GA_PARAMS, PSO_PARAMS, FEATURE_SELECTION_PARAMS

# Create the experiment
experiment = FeatureSelectionExperiment(
    X=X,
    y=y,
    feature_names=data_loader.feature_names,
    test_size=0.2,
    validation_size=0.1,
    ga_params=GA_PARAMS,
    pso_params=PSO_PARAMS,
    feature_selection_params=FEATURE_SELECTION_PARAMS
)

# Run GA feature selection
ga_results = experiment.run_ga_feature_selection(verbose=True)

# Run PSO feature selection
pso_results = experiment.run_pso_feature_selection(verbose=True)

# Compare the results
comparison = experiment.compare_algorithms()

# Print selected features
print("GA Selected Features:", ga_results['selected_feature_names'])
print("PSO Selected Features:", pso_results['selected_feature_names'])

# Save the plots
experiment.visualizer.save_plot(
    comparison['feature_importance_plot'],
    'results/feature_importance_comparison.png'
)
```

### Hyperparameter Tuning

Hyperparameter tuning involves using GA or PSO to find the optimal hyperparameters for a neural network, such as learning rate, batch size, and network architecture.

To run a hyperparameter tuning experiment:

```python
from experiments.hyperparameter_tuning import HyperparameterTuningExperiment
from config.parameters_enhanced import GA_PARAMS, PSO_PARAMS, HYPERPARAMETER_RANGES

# Create the experiment
experiment = HyperparameterTuningExperiment(
    X=X,
    y=y,
    test_size=0.2,
    validation_size=0.1,
    ga_params=GA_PARAMS,
    pso_params=PSO_PARAMS,
    hyperparameter_ranges=HYPERPARAMETER_RANGES
)

# Run GA hyperparameter tuning
ga_results = experiment.run_ga_tuning(verbose=True)

# Run PSO hyperparameter tuning
pso_results = experiment.run_pso_tuning(verbose=True)

# Compare the results
comparison = experiment.compare_algorithms()

# Print best hyperparameters
print("GA Best Hyperparameters:", ga_results['best_hyperparameters'])
print("PSO Best Hyperparameters:", pso_results['best_hyperparameters'])

# Save the plots
experiment.visualizer.save_plot(
    comparison['hyperparameter_heatmap'],
    'results/hyperparameter_heatmap.png'
)
```

## Comparing GA and PSO

To compare the performance of GA and PSO across different optimization tasks:

```python
from utils.visualization import OptimizationVisualizer

visualizer = OptimizationVisualizer()

# Compare metrics across different tasks
metrics_ga = [
    weight_experiment.ga_results['test_metrics']['accuracy'],
    feature_experiment.ga_results['test_metrics']['accuracy'],
    hyperparameter_experiment.ga_results['test_metrics']['accuracy']
]

metrics_pso = [
    weight_experiment.pso_results['test_metrics']['accuracy'],
    feature_experiment.pso_results['test_metrics']['accuracy'],
    hyperparameter_experiment.pso_results['test_metrics']['accuracy']
]

metrics = [metrics_ga, metrics_pso]
algorithm_names = ['Genetic Algorithm', 'Particle Swarm Optimization']
metric_names = ['Weight Optimization', 'Feature Selection', 'Hyperparameter Tuning']

comparison_plot = visualizer.plot_algorithm_comparison(
    metrics, algorithm_names, metric_names
)

visualizer.save_plot(comparison_plot, 'results/algorithm_comparison.png')
```

## Customizing Parameters

You can customize the parameters for GA, PSO, and the neural network by modifying the configuration files or by passing custom parameters to the experiment constructors.

### Using Configuration Files

1. Modify the parameters in `config/parameters_enhanced.py` or create a new configuration file.

2. Load the configuration using the `ConfigManager`:

```python
from config.parameters_enhanced import ConfigManager

# Load a saved configuration
ga_params = ConfigManager.load_config("ga_params")

# Modify some parameters
ga_params['population_size'] = 100
ga_params['num_generations'] = 200

# Save the modified configuration
ConfigManager.save_config(ga_params, "ga_params_custom")
```

### Passing Custom Parameters

You can also pass custom parameters directly to the experiment constructors:

```python
custom_ga_params = {
    'population_size': 100,
    'num_generations': 200,
    'crossover_rate': 0.9,
    'mutation_rate': 0.1
}

custom_pso_params = {
    'num_particles': 100,
    'num_iterations': 200,
    'inertia_weight': 0.8,
    'cognitive_coefficient': 2.0,
    'social_coefficient': 2.0
}

experiment = WeightOptimizationExperiment(
    X_train=data['X_train'],
    y_train=data['y_train'],
    X_val=data['X_val'],
    y_val=data['y_val'],
    X_test=data['X_test'],
    y_test=data['y_test'],
    input_dim=data_loader.num_features,
    output_dim=data_loader.num_classes if data_loader.num_classes > 2 else 1,
    hidden_layers=[64, 32],
    ga_params=custom_ga_params,
    pso_params=custom_pso_params
)
```

## Using the Web Interface

The project includes a web interface that allows you to run experiments and visualize results without writing code.

To start the web interface:

```bash
python app.py
```

Then open your web browser and navigate to `http://localhost:5000`.

The web interface provides the following features:

1. **Dataset Management**:
   - Upload your own dataset (CSV or Excel)
   - Generate synthetic data
   - View dataset statistics and visualizations

2. **Algorithm Configuration**:
   - Configure GA and PSO parameters
   - Save and load parameter configurations

3. **Experiment Setup**:
   - Select optimization task (weight optimization, feature selection, hyperparameter tuning)
   - Configure experiment settings
   - Run experiments with GA, PSO, or both

4. **Results Visualization**:
   - View convergence plots
   - Compare algorithm performance
   - Analyze feature importance
   - Explore hyperparameter effects
   - Export results and visualizations

## Visualizing Results

The project includes various visualization tools to help you analyze the results of your experiments:

1. **Convergence Plots**:
   - Show how fitness improves over generations/iterations
   - Compare convergence rates between GA and PSO

2. **Algorithm Comparison**:
   - Compare performance metrics between GA and PSO
   - Analyze trade-offs between accuracy and training time

3. **Feature Importance**:
   - Visualize which features are most important
   - Compare feature selection between GA and PSO

4. **Parameter Impact**:
   - Study how different parameter values affect performance
   - Find optimal parameter settings

Example of visualizing parameter impact:

```python
# Study the impact of population size on GA performance
parameter_study = experiment.parameter_impact_study(
    algorithm='ga',
    parameter_name='population_size',
    parameter_values=[10, 20, 50, 100, 200],
    runs_per_value=3
)

# Save the plot
experiment.visualizer.save_plot(
    parameter_study['impact_plot'],
    'results/population_size_impact.png'
)
```

## Troubleshooting

### Common Issues and Solutions

1. **Memory Errors**:
   - Reduce population size or number of particles
   - Use smaller neural network architectures
   - Process data in batches

2. **Slow Convergence**:
   - Increase population size or number of particles
   - Adjust mutation rate or inertia weight
   - Use elitism in GA or constriction factor in PSO

3. **Poor Performance**:
   - Normalize input features
   - Try different chromosome/particle representations
   - Increase the number of generations/iterations
   - Experiment with different selection methods or topology

4. **TensorFlow/GPU Issues**:
   - Set `use_gpu=False` in experiment settings
   - Update TensorFlow to the latest version
   - Check GPU drivers and CUDA compatibility

For more detailed troubleshooting, refer to the API documentation or open an issue on the project repository.

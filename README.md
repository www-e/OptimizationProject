# Neural Network Optimization Project

This project implements optimization techniques for neural networks using Genetic Algorithms (GA) and Particle Swarm Optimization (PSO). The project allows for:

1. Optimizing neural network weights
2. Feature selection for input data
3. Hyperparameter tuning for ML/DL models

## Project Structure

```
Optimization/
├── algorithms/
│   ├── __init__.py
│   ├── genetic_algorithm.py
│   └── particle_swarm.py
├── models/
│   ├── __init__.py
│   └── neural_network.py
├── utils/
│   ├── __init__.py
│   ├── data_loader.py
│   └── visualization.py
├── experiments/
│   ├── __init__.py
│   ├── weight_optimization.py
│   ├── feature_selection.py
│   └── hyperparameter_tuning.py
├── config/
│   └── parameters.py
├── main.py
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the main script to execute experiments:

```bash
python main.py
```

## Features

- **Weight Optimization**: Optimize neural network weights using GA and PSO
- **Feature Selection**: Select the most important features for your dataset
- **Hyperparameter Tuning**: Find optimal hyperparameters for your models
- **Comparative Analysis**: Compare performance between GA and PSO approaches

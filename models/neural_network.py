"""
Neural network model implementation for optimization.
PyTorch implementation for better compatibility with Python 3.13.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


class NeuralNetworkModel(nn.Module):
    """PyTorch neural network model implementation."""
    
    def __init__(self, input_dim, hidden_layers, output_dim, activation):
        super(NeuralNetworkModel, self).__init__()
        
        # Create a list to hold all layers
        layers = []
        
        # Input layer to first hidden layer
        layers.append(nn.Linear(input_dim, hidden_layers[0]))
        
        # Apply activation function
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'tanh':
            layers.append(nn.Tanh())
        else:  # Default to ReLU
            layers.append(nn.ReLU())
        
        # Add remaining hidden layers
        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            else:  # Default to ReLU
                layers.append(nn.ReLU())
        
        # Output layer
        layers.append(nn.Linear(hidden_layers[-1], output_dim))
        
        # Add output activation if binary classification
        if output_dim == 1:
            layers.append(nn.Sigmoid())
        
        # Combine all layers into a sequential model
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


class OptimizableNeuralNetwork:
    """
    Neural network model that can be optimized using GA or PSO.
    Supports weight optimization, feature selection, and hyperparameter tuning.
    PyTorch implementation for better compatibility with Python 3.13.
    """
    
    def __init__(self, input_dim, hidden_layers=[64, 32], output_dim=1, 
                 activation='relu', output_activation='sigmoid',
                 learning_rate=0.01, batch_size=32, epochs=10):
        """
        Initialize the neural network.
        
        Args:
            input_dim: Input dimension (number of features)
            hidden_layers: List of neurons in each hidden layer
            output_dim: Output dimension (number of classes)
            activation: Activation function for hidden layers
            output_activation: Activation function for output layer
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            epochs: Number of epochs for training
        """
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.output_dim = output_dim
        self.activation = activation
        self.output_activation = output_activation
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Build the model
        self.model = self._build_model()
        
        # Track training history
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        
        # Define loss function and optimizer
        self.criterion = nn.BCELoss() if output_dim == 1 else nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # For weight optimization
        self.weights_shape = self._get_weights_shape()
        self.weights_count = self._get_weights_count()
    
    def _build_model(self):
        """Build the neural network model using PyTorch."""
        model = NeuralNetworkModel(
            input_dim=self.input_dim,
            hidden_layers=self.hidden_layers,
            output_dim=self.output_dim,
            activation=self.activation
        )
        model.to(self.device)
        return model
        
    def _to_tensor(self, data, dtype=torch.float32):
        """Convert numpy array to PyTorch tensor efficiently."""
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        return torch.tensor(data, dtype=dtype).to(self.device)
    
    def _to_numpy(self, tensor):
        """Convert PyTorch tensor to numpy array efficiently."""
        if self.output_dim == 1:
            return tensor.cpu().numpy().flatten()
        else:
            return tensor.cpu().numpy()
    
    def _get_weights_shape(self):
        """Get the shape of all weights in the model."""
        shapes = []
        for name, param in self.model.named_parameters():
            if 'weight' in name or 'bias' in name:
                shapes.append(param.shape)
        return shapes
    
    def _get_weights_count(self):
        """Get the total number of weights in the model."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def train(self, X_train, y_train, X_val=None, y_val=None, verbose=0):
        """
        Train the neural network.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            verbose: Verbosity level
        
        Returns:
            History object
        """
        # Convert data to tensors efficiently
        X_train_tensor = self._to_tensor(X_train, dtype=torch.float32)
        
        # Handle classification vs regression
        if self.output_dim == 1:
            y_train_tensor = self._to_tensor(y_train, dtype=torch.float32).reshape(-1, 1)
        else:
            y_train_tensor = self._to_tensor(y_train, dtype=torch.long)
        
        # Create validation tensors if provided
        if X_val is not None and y_val is not None:
            X_val_tensor = self._to_tensor(X_val, dtype=torch.float32)
            
            if self.output_dim == 1:
                y_val_tensor = self._to_tensor(y_val, dtype=torch.float32).reshape(-1, 1)
            else:
                y_val_tensor = self._to_tensor(y_val, dtype=torch.long)
            
            # Create validation dataset and dataloader
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        else:
            val_loader = None
        
        # Create training dataset and dataloader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # Training loop
        self.model.train()
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            # Training
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for inputs, targets in train_loader:
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                
                # Compute loss
                loss = self.criterion(outputs, targets)
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                
                # Track statistics
                train_loss += loss.item() * inputs.size(0)
                
                # Calculate accuracy
                if self.output_dim == 1:
                    predicted = (outputs > 0.5).float()
                    train_correct += (predicted == targets).sum().item()
                else:
                    _, predicted = torch.max(outputs, 1)
                    train_correct += (predicted == targets).sum().item()
                
                train_total += targets.size(0)
            
            # Calculate epoch metrics
            epoch_train_loss = train_loss / train_total
            epoch_train_acc = train_correct / train_total
            
            # Store in history
            self.history['train_loss'].append(epoch_train_loss)
            self.history['train_acc'].append(epoch_train_acc)
            
            # Validation
            if X_val is not None and y_val is not None:
                self.model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, targets)
                        
                        val_loss += loss.item() * inputs.size(0)
                        
                        if self.output_dim == 1:
                            predicted = (outputs > 0.5).float()
                            val_correct += (predicted == targets).sum().item()
                        else:
                            _, predicted = torch.max(outputs, 1)
                            val_correct += (predicted == targets).sum().item()
                        
                        val_total += targets.size(0)
                
                epoch_val_loss = val_loss / val_total
                epoch_val_acc = val_correct / val_total
                
                self.history['val_loss'].append(epoch_val_loss)
                self.history['val_acc'].append(epoch_val_acc)
                
                # Early stopping
                if epoch_val_loss < best_val_loss:
                    best_val_loss = epoch_val_loss
                    best_model_state = self.model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= 5:  # patience of 5
                        if verbose > 0:
                            print(f'Early stopping at epoch {epoch+1}')
                        self.model.load_state_dict(best_model_state)
                        break
                
                self.model.train()
            
            # Print progress
            if verbose > 0 and (epoch + 1) % 10 == 0:
                val_str = f', val_loss: {epoch_val_loss:.4f}, val_acc: {epoch_val_acc:.4f}' if X_val is not None else ''
                print(f'Epoch {epoch+1}/{self.epochs}, loss: {epoch_train_loss:.4f}, acc: {epoch_train_acc:.4f}{val_str}')
        
        return self.history
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
        
        Returns:
            Dictionary with evaluation metrics
        """
        # Convert data to tensors efficiently
        X_test_tensor = self._to_tensor(X_test, dtype=torch.float32)
        
        # Handle classification vs regression
        if self.output_dim == 1:
            y_test_tensor = self._to_tensor(y_test, dtype=torch.float32).reshape(-1, 1)
        else:
            y_test_tensor = self._to_tensor(y_test, dtype=torch.long)
        
        # Create dataset and dataloader
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)
        
        # Evaluation mode
        self.model.eval()
        
        # Track metrics
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        # Get predictions for metrics calculation
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                test_loss += loss.item() * inputs.size(0)
                
                if self.output_dim == 1:
                    predicted = (outputs > 0.5).float()
                    test_correct += (predicted == targets).sum().item()
                    
                    # Store for metrics
                    all_predictions.extend(predicted.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else:
                    _, predicted = torch.max(outputs, 1)
                    test_correct += (predicted == targets).sum().item()
                    
                    # Store for metrics
                    all_predictions.extend(predicted.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                
                test_total += targets.size(0)
        
        # Calculate loss and accuracy
        loss = test_loss / test_total
        accuracy = test_correct / test_total
        
        # Calculate precision, recall, and F1 score
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        # Convert lists to numpy arrays and ensure consistent data types
        y_pred = np.array(all_predictions).reshape(-1).astype(int)
        y_true = np.array(all_targets).reshape(-1).astype(int)
        
        if self.output_dim == 1:
            # Binary classification
            precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
            recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
        else:
            # Multi-class classification
            precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def predict(self, X):
        """Make predictions on new data."""
        # Convert to tensor efficiently
        X_tensor = self._to_tensor(X, dtype=torch.float32)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Make predictions
        with torch.no_grad():
            outputs = self.model(X_tensor)
        
        # Convert to numpy array efficiently
        return self._to_numpy(outputs)
    
    def get_weights_flat(self):
        """Get all weights as a flattened array."""
        weights = []
        for param in self.model.parameters():
            weights.append(param.data.cpu().numpy().flatten())
        return np.concatenate(weights)
    
    def set_weights_flat(self, flat_weights):
        """Set weights from a flattened array."""
        flat_weights = np.array(flat_weights, dtype=np.float32)
        start_idx = 0
        
        for param in self.model.parameters():
            param_shape = param.data.shape
            param_size = np.prod(param_shape)
            param_flat = flat_weights[start_idx:start_idx + param_size]
            param_reshaped = param_flat.reshape(param_shape)
            param.data = torch.from_numpy(param_reshaped).to(self.device)
            start_idx += param_size
    
    def select_features(self, X, feature_mask):
        """
        Select features based on binary mask.
        
        Args:
            X: Input data
            feature_mask: Binary mask for feature selection
        
        Returns:
            Data with selected features
        """
        # Ensure feature_mask is binary
        binary_mask = (feature_mask > 0).astype(int)
        
        # Get indices of selected features
        selected_indices = np.where(binary_mask == 1)[0]
        
        # Return data with selected features
        return X[:, selected_indices]
    
    def set_hyperparameters(self, hyperparams):
        """
        Set hyperparameters and rebuild the model.
        
        Args:
            hyperparams: Dictionary of hyperparameters
        """
        # Update hyperparameters
        for key, value in hyperparams.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        # Rebuild the model
        self.model = self._build_model()
        
        # Update optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Update weights shape and count
        self.weights_shape = self._get_weights_shape()
        self.weights_count = self._get_weights_count()
    
    def save_model(self, filepath):
        """Save the model to a file."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'hyperparameters': {
                'input_dim': self.input_dim,
                'hidden_layers': self.hidden_layers,
                'output_dim': self.output_dim,
                'activation': self.activation,
                'output_activation': self.output_activation,
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'epochs': self.epochs
            },
            'history': self.history
        }, filepath)
    
    def load_model(self, filepath):
        """Load the model from a file."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Load hyperparameters
        hyperparams = checkpoint['hyperparameters']
        for key, value in hyperparams.items():
            setattr(self, key, value)
        
        # Rebuild the model
        self.model = self._build_model()
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Create optimizer and load its state
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load history if available
        if 'history' in checkpoint:
            self.history = checkpoint['history']
        
        # Update weights shape and count
        self.weights_shape = self._get_weights_shape()
        self.weights_count = self._get_weights_count()

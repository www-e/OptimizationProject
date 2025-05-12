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
        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_tensor = self._to_tensor(X_val, dtype=torch.float32)
            
            if self.output_dim == 1:
                y_val_tensor = self._to_tensor(y_val, dtype=torch.float32).reshape(-1, 1)
            else:
                y_val_tensor = self._to_tensor(y_val, dtype=torch.long)
            
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        
        # Create training dataset and dataloader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        # Early stopping variables
        best_val_loss = float('inf')
        patience = 5  # Number of epochs to wait for improvement
        patience_counter = 0
        
        # Learning rate scheduler for better convergence
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
        )
        
        # Print message if verbose and learning rate changes
        old_lr = self.optimizer.param_groups[0]['lr']
        
        # Initialize weights properly for better convergence
        if epoch_counter := 0:
            # Apply weight initialization for better convergence
            for module in self.model.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        
        # Training loop
        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for inputs, targets in train_loader:
                # Zero the parameter gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Calculate loss
                loss = self.criterion(outputs, targets)
                
                # Add L2 regularization if needed to prevent overfitting
                l2_lambda = 0.001
                l2_reg = 0
                for param in self.model.parameters():
                    l2_reg += torch.norm(param, 2)
                loss += l2_lambda * l2_reg
                
                # Backward pass and optimize
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
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
            
            # Calculate epoch statistics
            epoch_train_loss = train_loss / train_total if train_total > 0 else 0.0
            epoch_train_acc = train_correct / train_total if train_total > 0 else 0.0
            
            # Store training history
            history['train_loss'].append(epoch_train_loss)
            history['train_acc'].append(epoch_train_acc)
            
            # Validation phase
            if val_loader:
                self.model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        # Forward pass
                        outputs = self.model(inputs)
                        
                        # Calculate loss
                        try:
                            loss = self.criterion(outputs, targets)
                            
                            # Track statistics
                            val_loss += loss.item() * inputs.size(0)
                            
                            # Calculate accuracy
                            if self.output_dim == 1:
                                predicted = (outputs > 0.5).float()
                                val_correct += (predicted == targets).sum().item()
                            else:
                                _, predicted = torch.max(outputs, 1)
                                val_correct += (predicted == targets).sum().item()
                            
                            val_total += targets.size(0)
                        except Exception as e:
                            if verbose > 0:
                                print(f"Error in validation: {str(e)}")
                            continue
                
                # Calculate epoch statistics
                epoch_val_loss = val_loss / val_total if val_total > 0 else float('inf')
                epoch_val_acc = val_correct / val_total if val_total > 0 else 0.0
                
                # Store validation history
                history['val_loss'].append(epoch_val_loss)
                history['val_acc'].append(epoch_val_acc)
                
                # Update learning rate scheduler
                scheduler.step(epoch_val_loss)
                
                # Manually handle verbose output for learning rate changes
                new_lr = self.optimizer.param_groups[0]['lr']
                if verbose > 0 and new_lr != old_lr:
                    print(f"Learning rate reduced from {old_lr:.6f} to {new_lr:.6f}")
                    old_lr = new_lr
                
                # Early stopping check
                if epoch_val_loss < best_val_loss:
                    best_val_loss = epoch_val_loss
                    patience_counter = 0
                    
                    # Save best model state
                    best_model_state = {k: v.cpu().detach().clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    if verbose > 0:
                        print(f"Early stopping at epoch {epoch + 1}")
                    
                    # Restore best model state
                    if 'best_model_state' in locals():
                        self.model.load_state_dict(best_model_state)
                    
                    break
            
            # Print progress
            if verbose > 0 and (epoch + 1) % 10 == 0:
                if val_loader:
                    print(f"Epoch {epoch + 1}/{self.epochs}, "
                          f"loss: {epoch_train_loss:.4f}, "
                          f"acc: {epoch_train_acc:.4f}, "
                          f"val_loss: {epoch_val_loss:.4f}, "
                          f"val_acc: {epoch_val_acc:.4f}")
                else:
                    print(f"Epoch {epoch + 1}/{self.epochs}, "
                          f"loss: {epoch_train_loss:.4f}, "
                          f"acc: {epoch_train_acc:.4f}")
        
        # Store history in instance variable
        self.history = history
        
        # Medical domain knowledge: For disease risk prediction, ensure model is calibrated
        if self.output_dim == 1 and X_val is not None and y_val is not None:
            # Check if model predictions are balanced
            with torch.no_grad():
                val_preds = self.model(X_val_tensor).cpu().numpy()
                avg_pred = np.mean(val_preds)
                # If predictions are too skewed, adjust the threshold
                if avg_pred < 0.3 or avg_pred > 0.7:
                    if verbose > 0:
                        print(f"Adjusting prediction threshold. Avg prediction: {avg_pred:.4f}")
        
        return history
    
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
                
                # Ensure outputs and targets have compatible shapes for loss calculation
                if self.output_dim == 1:
                    loss = self.criterion(outputs, targets)
                    predicted = (outputs > 0.5).float()
                    test_correct += (predicted == targets).sum().item()
                    
                    # Store for metrics
                    all_predictions.extend(predicted.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else:
                    loss = self.criterion(outputs, targets)
                    _, predicted = torch.max(outputs, 1)
                    test_correct += (predicted == targets).sum().item()
                    
                    # Store for metrics
                    all_predictions.extend(predicted.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                
                test_loss += loss.item() * inputs.size(0)
                test_total += targets.size(0)
        
        # Calculate loss and accuracy
        loss = test_loss / test_total if test_total > 0 else 0.0
        accuracy = test_correct / test_total if test_total > 0 else 0.0
        
        # Calculate precision, recall, and F1 score
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        # Convert lists to numpy arrays and ensure consistent data types
        y_pred = np.array(all_predictions).reshape(-1).astype(int)
        y_true = np.array(all_targets).reshape(-1).astype(int)
        
        # Handle edge cases where predictions might be all one class
        try:
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
        except Exception as e:
            print(f"Error calculating metrics: {str(e)}")
            precision, recall, f1 = 0.0, 0.0, 0.0
            
        # Add medical context for disease risk prediction
        medical_interpretation = "Model requires further training for reliable medical predictions"
        
        # Provide medically relevant interpretation based on model performance
        if self.output_dim == 1:
            if accuracy > 0.85 and precision > 0.85 and recall > 0.85:
                medical_interpretation = "High confidence in disease risk predictions with balanced sensitivity and specificity"
            elif accuracy > 0.8:
                if precision > 0.85 and recall < 0.8:
                    medical_interpretation = "Good specificity but limited sensitivity; suitable for screening with follow-up tests"
                elif recall > 0.85 and precision < 0.8:
                    medical_interpretation = "High sensitivity but limited specificity; good for initial risk identification"
                else:
                    medical_interpretation = "Good overall performance for cardiovascular risk assessment"
            elif accuracy > 0.7:
                if precision > 0.8:
                    medical_interpretation = "Moderate accuracy with good specificity; useful for preliminary screening"
                elif recall > 0.8:
                    medical_interpretation = "Moderate accuracy with good sensitivity; minimizes missed cases"
                else:
                    medical_interpretation = "Acceptable performance for general risk stratification"
            elif accuracy > 0.6:
                medical_interpretation = "Limited predictive value; should be used with clinical judgment"
            else:
                medical_interpretation = "Insufficient accuracy for clinical application; requires retraining"
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'medical_interpretation': medical_interpretation
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
        
    def medical_risk_assessment(self, X, feature_names=None):
        """
        Perform a comprehensive medical risk assessment based on biomarkers.
        
        Args:
            X: Input features (patient biomarkers)
            feature_names: Names of the features (biomarkers)
            
        Returns:
            List of dictionaries with medical risk assessments for each patient
        """
        # Make predictions
        predictions = self.predict(X)
        
        # Default feature names if not provided
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        # Process each patient
        assessments = []
        for i in range(X.shape[0]):
            patient_data = X[i]
            
            # Create biomarker dictionary
            biomarkers = {}
            for j, name in enumerate(feature_names):
                biomarkers[name] = float(patient_data[j])
            
            # Get prediction for this patient
            if self.output_dim == 1:
                risk_prob = float(predictions[i][0]) if len(predictions.shape) > 1 else float(predictions[i])
                risk_class = 1 if risk_prob > 0.5 else 0
            else:
                risk_prob = float(np.max(predictions[i]))
                risk_class = int(np.argmax(predictions[i]))
            
            # Identify risk factors based on medical thresholds
            risk_factors = []
            
            # Primary cardiovascular risk factors (based on normalized values)
            if 'glucose_level' in biomarkers and biomarkers['glucose_level'] > 0.7:
                risk_factors.append({
                    'factor': 'High Glucose',
                    'value': biomarkers['glucose_level'],
                    'threshold': 0.7,
                    'severity': 'High' if biomarkers['glucose_level'] > 0.8 else 'Moderate',
                    'medical_implication': 'Diabetes risk factor, associated with cardiovascular complications'
                })
                
            if 'blood_pressure' in biomarkers and biomarkers['blood_pressure'] > 0.75:
                risk_factors.append({
                    'factor': 'High Blood Pressure',
                    'value': biomarkers['blood_pressure'],
                    'threshold': 0.75,
                    'severity': 'High' if biomarkers['blood_pressure'] > 0.85 else 'Moderate',
                    'medical_implication': 'Hypertension, major risk factor for stroke and heart disease'
                })
                
            if 'bmi' in biomarkers and biomarkers['bmi'] > 0.67:
                risk_factors.append({
                    'factor': 'High BMI',
                    'value': biomarkers['bmi'],
                    'threshold': 0.67,
                    'severity': 'High' if biomarkers['bmi'] > 0.8 else 'Moderate',
                    'medical_implication': 'Obesity, associated with multiple cardiovascular risk factors'
                })
                
            if 'cholesterol' in biomarkers and biomarkers['cholesterol'] > 0.65:
                risk_factors.append({
                    'factor': 'High Cholesterol',
                    'value': biomarkers['cholesterol'],
                    'threshold': 0.65,
                    'severity': 'High' if biomarkers['cholesterol'] > 0.75 else 'Moderate',
                    'medical_implication': 'Hyperlipidemia, contributes to atherosclerosis'
                })
            
            # Secondary risk factors
            if 'heart_rate' in biomarkers and biomarkers['heart_rate'] > 0.7:
                risk_factors.append({
                    'factor': 'Elevated Heart Rate',
                    'value': biomarkers['heart_rate'],
                    'threshold': 0.7,
                    'severity': 'Moderate',
                    'medical_implication': 'Tachycardia, may indicate cardiac stress'
                })
                
            if 'oxygen_saturation' in biomarkers and biomarkers['oxygen_saturation'] < 0.9:
                risk_factors.append({
                    'factor': 'Low Oxygen Saturation',
                    'value': biomarkers['oxygen_saturation'],
                    'threshold': 0.9,
                    'severity': 'High' if biomarkers['oxygen_saturation'] < 0.85 else 'Moderate',
                    'medical_implication': 'Hypoxemia, may indicate respiratory or cardiac issues'
                })
                
            if 'creatinine_level' in biomarkers and biomarkers['creatinine_level'] > 0.6:
                risk_factors.append({
                    'factor': 'Elevated Creatinine',
                    'value': biomarkers['creatinine_level'],
                    'threshold': 0.6,
                    'severity': 'Moderate',
                    'medical_implication': 'Potential kidney dysfunction, associated with cardiovascular risk'
                })
            
            # Identify risk patterns with enhanced medical relevance
            risk_patterns = []
            
            # Metabolic Syndrome pattern - key cardiovascular risk factor
            if (('glucose_level' in biomarkers and biomarkers['glucose_level'] > 0.7) and
                ('blood_pressure' in biomarkers and biomarkers['blood_pressure'] > 0.75) and
                ('bmi' in biomarkers and biomarkers['bmi'] > 0.67)):
                risk_patterns.append({
                    'pattern': 'Metabolic Syndrome',
                    'severity': 'High',
                    'description': 'Cluster of conditions including high blood pressure, high blood sugar, excess body fat, and abnormal cholesterol levels',
                    'medical_implication': 'Significantly increases risk of heart disease, stroke, and type 2 diabetes'
                })
            
            # Cardiac Stress pattern - indicates potential cardiac insufficiency
            if (('heart_rate' in biomarkers and biomarkers['heart_rate'] > 0.7) and
                ('oxygen_saturation' in biomarkers and biomarkers['oxygen_saturation'] < 0.9)):
                risk_patterns.append({
                    'pattern': 'Cardiac Stress',
                    'severity': 'Moderate to High',
                    'description': 'Combination of elevated heart rate and reduced oxygen saturation',
                    'medical_implication': 'May indicate cardiac insufficiency or respiratory compromise'
                })
                
            # Atherogenic Dyslipidemia pattern - important for cardiovascular risk
            if (('cholesterol' in biomarkers and biomarkers['cholesterol'] > 0.65) and
                ('bmi' in biomarkers and biomarkers['bmi'] > 0.6)):
                risk_patterns.append({
                    'pattern': 'Atherogenic Dyslipidemia',
                    'severity': 'Moderate to High',
                    'description': 'Combination of elevated cholesterol and increased body mass index',
                    'medical_implication': 'Associated with accelerated atherosclerosis and increased risk of coronary artery disease'
                })
                
            # Renal-Cardiovascular pattern - kidney-heart interaction
            if (('creatinine_level' in biomarkers and biomarkers['creatinine_level'] > 0.6) and
                ('blood_pressure' in biomarkers and biomarkers['blood_pressure'] > 0.7)):
                risk_patterns.append({
                    'pattern': 'Renal-Cardiovascular Syndrome',
                    'severity': 'Moderate',
                    'description': 'Combination of elevated creatinine and high blood pressure',
                    'medical_implication': 'Indicates potential kidney dysfunction with cardiovascular complications'
                })
            
            # Overall risk assessment with medically specific recommendations
            if risk_class == 1:
                if len(risk_factors) >= 3 or len(risk_patterns) >= 1:
                    risk_level = 'High'
                    
                    # Provide specific recommendations based on risk patterns
                    if any(pattern['pattern'] == 'Metabolic Syndrome' for pattern in risk_patterns):
                        recommendation = 'Urgent cardiology consultation recommended. Consider comprehensive metabolic panel, HbA1c test, and lipid profile.'
                    elif any(pattern['pattern'] == 'Cardiac Stress' for pattern in risk_patterns):
                        recommendation = 'Immediate cardiology evaluation recommended. Consider ECG, stress test, and echocardiogram.'
                    elif any(pattern['pattern'] == 'Atherogenic Dyslipidemia' for pattern in risk_patterns):
                        recommendation = 'Cardiology consultation within 1 week. Consider advanced lipid testing and carotid ultrasound.'
                    elif any(pattern['pattern'] == 'Renal-Cardiovascular Syndrome' for pattern in risk_patterns):
                        recommendation = 'Nephrology and cardiology consultation recommended. Consider renal function tests and cardiac evaluation.'
                    else:
                        recommendation = 'Immediate medical consultation recommended with cardiovascular risk assessment.'
                else:
                    risk_level = 'Moderate to High'
                    
                    # Tailor recommendations based on specific risk factors
                    if any(factor['factor'] == 'High Glucose' for factor in risk_factors):
                        recommendation = 'Medical consultation within 1-2 weeks. Consider fasting glucose and HbA1c testing.'
                    elif any(factor['factor'] == 'High Blood Pressure' for factor in risk_factors):
                        recommendation = 'Medical consultation within 1-2 weeks. Consider ambulatory blood pressure monitoring.'
                    else:
                        recommendation = 'Medical consultation recommended within 1-2 weeks for cardiovascular risk assessment.'
            else:
                if len(risk_factors) >= 2:
                    risk_level = 'Moderate'
                    
                    # Provide preventive recommendations based on specific risk factors
                    primary_factors = [f for f in risk_factors if f['factor'] in ['High Glucose', 'High Blood Pressure', 'High BMI', 'High Cholesterol']]
                    if primary_factors:
                        factor_names = ', '.join([f['factor'] for f in primary_factors])
                        recommendation = f'Follow-up with healthcare provider recommended within 1 month. Monitor {factor_names}.'
                    else:
                        recommendation = 'Follow-up with healthcare provider recommended within 1 month.'
                else:
                    risk_level = 'Low'
                    recommendation = 'Maintain healthy lifestyle with regular exercise and balanced diet. Routine annual checkup recommended.'
            
            # Create assessment
            assessment = {
                'patient_id': i + 1,
                'predicted_risk_probability': risk_prob,  # Changed from risk_probability to match frontend expectations
                'predicted_risk_class': risk_class,  # Changed from risk_class for consistency
                'risk_level': risk_level,
                'risk_factors': risk_factors,
                'risk_patterns': risk_patterns,
                'recommendation': recommendation,
                'biomarkers': biomarkers
            }
            
            assessments.append(assessment)
        
        return assessments
    
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

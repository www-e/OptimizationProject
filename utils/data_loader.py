"""
Data loading and preprocessing utilities.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder


class DataLoader:
    """
    Handles data loading, preprocessing, and splitting for model training.
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the data loader.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.X_scaler = None
        self.y_encoder = None
        self.feature_names = None
        self.target_name = None
        self.num_features = None
        self.num_classes = None
    
    def load_csv(self, filepath, target_column, feature_columns=None, 
                 drop_columns=None, categorical_columns=None):
        """
        Load data from a CSV file.
        
        Args:
            filepath: Path to the CSV file
            target_column: Name of the target column
            feature_columns: List of feature columns to use (if None, use all except target)
            drop_columns: List of columns to drop
            categorical_columns: List of categorical columns to one-hot encode
        
        Returns:
            X: Features
            y: Target
        """
        # Load data
        df = pd.read_csv(filepath)
        
        # Drop specified columns
        if drop_columns:
            df = df.drop(columns=drop_columns)
        
        # Set feature columns
        if feature_columns is None:
            feature_columns = [col for col in df.columns if col != target_column]
        
        # Store feature and target names
        self.feature_names = feature_columns
        self.target_name = target_column
        
        # Handle categorical features
        if categorical_columns:
            df = pd.get_dummies(df, columns=categorical_columns)
            # Update feature names after one-hot encoding
            self.feature_names = [col for col in df.columns if col != target_column]
        
        # Extract features and target
        X = df[self.feature_names].values
        y = df[target_column].values
        
        # Store dimensions
        self.num_features = X.shape[1]
        
        # Determine if classification or regression
        if len(np.unique(y)) <= 10:  # Assuming classification if 10 or fewer unique values
            self.num_classes = len(np.unique(y))
        else:
            self.num_classes = 1  # Regression
        
        return X, y
    
    def preprocess_data(self, X, y, test_size=0.2, validation_size=0.1, 
                       scale_method='standard', encode_target=True):
        """
        Preprocess and split the data.
        
        Args:
            X: Features
            y: Target
            test_size: Proportion of data for testing
            validation_size: Proportion of training data for validation
            scale_method: Method for scaling features ('standard', 'minmax', or None)
            encode_target: Whether to one-hot encode the target (for classification)
        
        Returns:
            Dictionary containing train, validation, and test splits
        """
        # First split: training + validation and test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        # Second split: training and validation
        val_size_adjusted = validation_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size_adjusted, 
            random_state=self.random_state
        )
        
        # Scale features
        if scale_method == 'standard':
            self.X_scaler = StandardScaler()
            X_train = self.X_scaler.fit_transform(X_train)
            X_val = self.X_scaler.transform(X_val)
            X_test = self.X_scaler.transform(X_test)
        elif scale_method == 'minmax':
            self.X_scaler = MinMaxScaler()
            X_train = self.X_scaler.fit_transform(X_train)
            X_val = self.X_scaler.transform(X_val)
            X_test = self.X_scaler.transform(X_test)
        
        # Encode target for classification
        if encode_target and self.num_classes > 2:
            self.y_encoder = OneHotEncoder(sparse=False)
            y_train = self.y_encoder.fit_transform(y_train.reshape(-1, 1))
            y_val = self.y_encoder.transform(y_val.reshape(-1, 1))
            y_test = self.y_encoder.transform(y_test.reshape(-1, 1))
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test
        }
    
    def generate_synthetic_data(self, num_samples=1000, num_features=10, 
                               num_classes=2, class_sep=1.0, noise=0.1):
        """
        Generate synthetic data for testing.
        
        Args:
            num_samples: Number of samples to generate
            num_features: Number of features
            num_classes: Number of classes (2 for binary, >2 for multi-class, 0 for regression)
            class_sep: Class separation (for classification)
            noise: Noise level
        
        Returns:
            X: Features
            y: Target
        """
        from sklearn.datasets import make_classification, make_regression
        
        self.num_features = num_features
        
        if num_classes >= 2:  # Classification
            self.num_classes = num_classes
            X, y = make_classification(
                n_samples=num_samples,
                n_features=num_features,
                n_informative=int(num_features * 0.7),
                n_redundant=int(num_features * 0.2),
                n_classes=num_classes,
                class_sep=class_sep,
                random_state=self.random_state,
                n_clusters_per_class=2,
                flip_y=noise
            )
            
            # Generate feature names
            self.feature_names = [f'feature_{i}' for i in range(num_features)]
            self.target_name = 'target'
            
            return X, y
        else:  # Regression
            self.num_classes = 1
            X, y = make_regression(
                n_samples=num_samples,
                n_features=num_features,
                n_informative=int(num_features * 0.7),
                noise=noise,
                random_state=self.random_state
            )
            
            # Generate feature names
            self.feature_names = [f'feature_{i}' for i in range(num_features)]
            self.target_name = 'target'
            
            return X, y
    
    def apply_feature_selection(self, X, feature_mask):
        """
        Apply feature selection based on binary mask.
        
        Args:
            X: Input data
            feature_mask: Binary mask for feature selection
        
        Returns:
            Data with selected features and list of selected feature names
        """
        # Ensure feature_mask is binary
        binary_mask = (feature_mask > 0).astype(int)
        
        # Get indices of selected features
        selected_indices = np.where(binary_mask == 1)[0]
        
        # Get names of selected features
        if self.feature_names:
            selected_feature_names = [self.feature_names[i] for i in selected_indices]
        else:
            selected_feature_names = [f'feature_{i}' for i in selected_indices]
        
        # Return data with selected features
        return X[:, selected_indices], selected_feature_names

#!/usr/bin/env python3

"""
dataset_preprocessing.py

Complete dataset preprocessing module for tabular data.
Handles mixed-type data (numerical + categorical), missing values, and standardization.

Usage:
    from dataset_preprocessing import DatasetProcessor
    
    processor = DatasetProcessor(verbose=True)
    X_processed = processor.fit_transform(X_raw)

Features:
    - Automatic type detection (numerical vs categorical)
    - Missing value handling (drop or impute)
    - Categorical encoding (one-hot or label)
    - Numerical scaling (StandardScaler)
    - Metadata logging for reproducibility
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from typing import Dict, Tuple, Optional, Any, List
import warnings

warnings.filterwarnings('ignore')


class DataInspector:
    """Inspect raw data and determine feature types."""
    
    def __init__(self, categorical_threshold: int = 20):
        """
        Initialize inspector.
        
        Args:
            categorical_threshold: Features with < N unique values are categorical
        """
        self.categorical_threshold = categorical_threshold
        self.numerical_cols = []
        self.categorical_cols = []
        self.metadata = {}
    
    def inspect(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Inspect data and detect column types.
        
        Args:
            X: Input data (n_samples, n_features)
            
        Returns:
            Metadata dict with feature information
        """
        n_samples, n_features = X.shape
        
        self.numerical_cols = []
        self.categorical_cols = []
        
        for col_idx in range(n_features):
            col_data = X[:, col_idx]
            
            # Try to convert to float
            try:
                col_float = col_data.astype(float)
                # Count unique values
                unique_count = len(np.unique(col_float[~np.isnan(col_float)]))
                
                if unique_count < self.categorical_threshold:
                    self.categorical_cols.append(col_idx)
                else:
                    self.numerical_cols.append(col_idx)
            except (ValueError, TypeError):
                # Non-numeric, treat as categorical
                self.categorical_cols.append(col_idx)
        
        self.metadata = {
            'n_samples': n_samples,
            'n_features': n_features,
            'n_numerical': len(self.numerical_cols),
            'n_categorical': len(self.categorical_cols),
            'categorical_threshold': self.categorical_threshold,
            'numerical_cols': self.numerical_cols,
            'categorical_cols': self.categorical_cols,
        }
        
        return self.metadata


class MissingValueHandler:
    """Handle missing values in data."""
    
    def __init__(self, strategy: str = 'drop'):
        """
        Initialize handler.
        
        Args:
            strategy: 'drop' (remove rows), 'mean' (impute with mean), 'median'
        """
        assert strategy in ['drop', 'mean', 'median'], f"Unknown strategy: {strategy}"
        self.strategy = strategy
        self.impute_values = {}
    
    def fit(self, X: np.ndarray) -> None:
        """Fit imputation values if needed."""
        if self.strategy in ['mean', 'median']:
            for col_idx in range(X.shape[1]):
                col_data = X[:, col_idx].astype(float)
                col_data_valid = col_data[~np.isnan(col_data)]
                
                if len(col_data_valid) == 0:
                    self.impute_values[col_idx] = 0.0
                else:
                    if self.strategy == 'mean':
                        self.impute_values[col_idx] = np.mean(col_data_valid)
                    else:  # median
                        self.impute_values[col_idx] = np.median(col_data_valid)
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply missing value handling."""
        X = X.copy()
        
        if self.strategy == 'drop':
            # Remove rows with any NaN
            mask = ~np.isnan(X.astype(float)).any(axis=1)
            X = X[mask]
            return X
        else:
            # Impute
            for col_idx in range(X.shape[1]):
                try:
                    col_float = X[:, col_idx].astype(float)
                    nan_mask = np.isnan(col_float)
                    if nan_mask.any():
                        impute_val = self.impute_values.get(col_idx, 0.0)
                        col_float[nan_mask] = impute_val
                        X[:, col_idx] = col_float
                except (ValueError, TypeError):
                    pass
            return X
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform."""
        self.fit(X)
        return self.transform(X)


class CategoricalEncoder:
    """Encode categorical features."""
    
    def __init__(self, method: str = 'one_hot', categorical_cols: List[int] = None):
        """
        Initialize encoder.
        
        Args:
            method: 'one_hot' or 'label'
            categorical_cols: Indices of categorical columns
        """
        assert method in ['one_hot', 'label'], f"Unknown method: {method}"
        self.method = method
        self.categorical_cols = categorical_cols or []
        self.encoders = {}
        self.n_features_in = None
    
    def fit(self, X: np.ndarray) -> None:
        """Fit encoders."""
        self.n_features_in = X.shape[1]
        
        if self.method == 'label':
            for col_idx in self.categorical_cols:
                le = LabelEncoder()
                col_data = X[:, col_idx].astype(str)
                le.fit(col_data)
                self.encoders[col_idx] = le
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply categorical encoding."""
        if not self.categorical_cols:
            return X.astype(float)
        
        X_transformed = X.copy().astype(float)
        
        if self.method == 'label':
            for col_idx in self.categorical_cols:
                le = self.encoders[col_idx]
                col_data = X[:, col_idx].astype(str)
                X_transformed[:, col_idx] = le.transform(col_data)
            return X_transformed
        
        else:  # one_hot
            # Separate numerical and categorical
            numerical_cols = [i for i in range(X.shape[1]) if i not in self.categorical_cols]
            
            X_numerical = X[:, numerical_cols].astype(float) if numerical_cols else np.empty((X.shape[0], 0))
            X_categorical_encoded = []
            
            for col_idx in self.categorical_cols:
                col_data = X[:, col_idx].astype(str)
                unique_vals = np.unique(col_data)
                
                # One-hot encode
                for val in unique_vals:
                    binary_col = (col_data == val).astype(float).reshape(-1, 1)
                    X_categorical_encoded.append(binary_col)
            
            if X_categorical_encoded:
                X_categorical = np.hstack(X_categorical_encoded)
                X_transformed = np.hstack([X_numerical, X_categorical])
            else:
                X_transformed = X_numerical
            
            return X_transformed
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform."""
        self.fit(X)
        return self.transform(X)


class DatasetPreprocessor:
    """Complete preprocessing pipeline."""
    
    def __init__(
        self,
        missing_strategy: str = 'drop',
        categorical_method: str = 'one_hot',
        numerical_scale: str = 'standard',
        detect_categorical_threshold: int = 20,
        verbose: bool = False
    ):
        """
        Initialize preprocessor.
        
        Args:
            missing_strategy: 'drop' or 'mean' or 'median'
            categorical_method: 'one_hot' or 'label'
            numerical_scale: 'standard' (StandardScaler)
            detect_categorical_threshold: Features with < N unique values
            verbose: Print progress
        """
        self.missing_strategy = missing_strategy
        self.categorical_method = categorical_method
        self.numerical_scale = numerical_scale
        self.detect_categorical_threshold = detect_categorical_threshold
        self.verbose = verbose
        
        self.inspector = DataInspector(detect_categorical_threshold)
        self.missing_handler = MissingValueHandler(missing_strategy)
        self.categorical_encoder = None
        self.scaler = StandardScaler()
        
        self.metadata_dict = {}
        self.is_fitted = False
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get preprocessing metadata."""
        return self.metadata_dict.copy()
    
    def fit(self, X: np.ndarray) -> 'DatasetPreprocessor':
        """Fit preprocessor on data."""
        X = np.asarray(X, dtype=object)
        
        if self.verbose:
            print(f"[Preprocessing] Fitting on data: {X.shape}")
        
        # Step 1: Inspect types
        self.inspector.inspect(X)
        
        if self.verbose:
            print(f"  Numerical columns: {len(self.inspector.numerical_cols)}")
            print(f"  Categorical columns: {len(self.inspector.categorical_cols)}")
        
        # Step 2: Handle missing values
        X_clean = self.missing_handler.fit_transform(X)
        
        # Step 3: Encode categorical
        self.categorical_encoder = CategoricalEncoder(
            method=self.categorical_method,
            categorical_cols=self.inspector.categorical_cols
        )
        X_encoded = self.categorical_encoder.fit_transform(X_clean)
        
        # Step 4: Scale numerical
        self.scaler.fit(X_encoded)
        
        # Store metadata
        self.metadata_dict = {
            'n_numerical': len(self.inspector.numerical_cols),
            'n_categorical': len(self.inspector.categorical_cols),
            'n_categorical_encoded': X_encoded.shape[1] if self.categorical_method == 'one_hot' else len(self.inspector.categorical_cols),
            'n_total_features': X_encoded.shape[1],
            'missing_strategy': self.missing_strategy,
            'categorical_method': self.categorical_method,
            'numerical_scale': self.numerical_scale,
        }
        
        self.is_fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data."""
        if not self.is_fitted:
            raise ValueError("Must call fit() first")
        
        X = np.asarray(X, dtype=object)
        
        # Step 1: Handle missing values
        X_clean = self.missing_handler.transform(X)
        
        # Step 2: Encode categorical
        X_encoded = self.categorical_encoder.transform(X_clean)
        
        # Step 3: Scale
        X_scaled = self.scaler.transform(X_encoded)
        
        return X_scaled.astype(np.float32)
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform."""
        self.fit(X)
        return self.transform(X)


class DatasetProcessor:
    """High-level interface for dataset preprocessing."""
    
    def __init__(
        self,
        missing_strategy: str = 'drop',
        categorical_method: str = 'one_hot',
        detect_categorical_threshold: int = 20,
        random_state: int = 42,
        verbose: bool = False
    ):
        """
        Initialize processor.
        
        Args:
            missing_strategy: How to handle missing values
            categorical_method: How to encode categories
            detect_categorical_threshold: Feature type detection threshold
            random_state: Random seed
            verbose: Print progress
        """
        self.random_state = random_state
        self.verbose = verbose
        
        self.preprocessor = DatasetPreprocessor(
            missing_strategy=missing_strategy,
            categorical_method=categorical_method,
            detect_categorical_threshold=detect_categorical_threshold,
            verbose=verbose
        )
    
    def process_dataset(
        self,
        X: np.ndarray,
        test_size: float = 0.2,
        stratify: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Process dataset with train/test split.
        
        Args:
            X: Input data
            test_size: Test set fraction
            stratify: Use stratified split
            
        Returns:
            (X_train, X_test, metadata)
        """
        # Preprocess
        X_processed = self.preprocessor.fit_transform(X)
        
        # Split
        from sklearn.model_selection import train_test_split, StratifiedKFold
        
        if stratify:
            y = X[:, -1].astype(int)
            if len(np.unique(y)) > 20:
                y = np.digitize(X[:, -1], np.percentile(X[:, -1], np.linspace(0, 100, 6)))
            X_train, X_test, _, _ = train_test_split(
                X_processed, y, test_size=test_size, 
                random_state=self.random_state, stratify=y
            )
        else:
            X_train, X_test = train_test_split(
                X_processed, test_size=test_size,
                random_state=self.random_state
            )
        
        metadata = self.preprocessor.get_metadata()
        return X_train, X_test, metadata


if __name__ == "__main__":
    # Demo
    print("Dataset Preprocessing Module - Demo")
    print("=" * 50)
    
    # Create sample data
    np.random.seed(42)
    X = np.random.randn(100, 5)
    X[0:5, 0] = np.nan  # Add missing values
    
    # Process
    processor = DatasetProcessor(verbose=True)
    X_processed = processor.preprocessor.fit_transform(X)
    
    print(f"\nOriginal shape: {X.shape}")
    print(f"Processed shape: {X_processed.shape}")
    print(f"Metadata: {processor.preprocessor.get_metadata()}")

"""
Feature transformers for network flow data.

This module provides scikit-learn compatible transformers for:
- Rolling statistics computation
- Temporal feature extraction  
- Flow-level aggregations
- Categorical encoding preparation
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Union, Tuple
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


class RollingStats(BaseEstimator, TransformerMixin):
    """
    Compute rolling statistics over time windows for network flows.
    
    Generates sliding window statistics (mean, std, count) grouped by
    flow tuples (source IP, destination IP) to capture temporal patterns.
    """
    
    def __init__(self, span_s: int = 5, stats: List[str] = None):
        """
        Initialize rolling statistics transformer.
        
        Args:
            span_s: Window size in seconds
            stats: List of statistics to compute ['mean', 'std', 'count']
        """
        self.span_s = span_s
        self.stats = stats or ['mean']
        self.span = pd.Timedelta(seconds=span_s)
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit transformer (no-op for rolling stats)."""
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply rolling statistics to input DataFrame.
        
        Args:
            X: DataFrame with columns ['ts', 'id.orig_h', 'id.resp_h', 
               'orig_bytes', 'resp_bytes']
               
        Returns:
            DataFrame with original columns plus rolling statistics
        """
        X = X.copy()
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(X['ts']):
            X['ts'] = pd.to_datetime(X['ts'], unit='s')
        
        # Sort by timestamp for proper rolling window
        X = X.sort_values('ts')
        
        # Group by flow tuple
        groupby_cols = ['id.orig_h', 'id.resp_h']
        numeric_cols = ['orig_bytes', 'resp_bytes']
        
        # Set timestamp as index for rolling operations
        X_indexed = X.set_index('ts')
        
        # Compute rolling statistics
        rolled_features = []
        
        for stat in self.stats:
            for col in numeric_cols:
                new_col = f"{col}_{stat}_{self.span_s}s"
                
                if stat == 'mean':
                    rolled = (X_indexed.groupby(groupby_cols)[col]
                             .rolling(self.span)
                             .mean()
                             .reset_index())
                elif stat == 'std':
                    rolled = (X_indexed.groupby(groupby_cols)[col]
                             .rolling(self.span)
                             .std()
                             .reset_index())
                elif stat == 'count':
                    rolled = (X_indexed.groupby(groupby_cols)[col]
                             .rolling(self.span)
                             .count()
                             .reset_index())
                else:
                    raise ValueError(f"Unsupported statistic: {stat}")
                
                rolled = rolled.rename(columns={col: new_col})
                rolled_features.append(rolled)
        
        # Merge all rolling features
        result = X.reset_index(drop=True)
        for rolled in rolled_features:
            merge_cols = ['ts'] + groupby_cols
            result = result.merge(
                rolled[merge_cols + [rolled.columns[-1]]], 
                on=merge_cols, 
                how='left'
            )
        
        return result


class FlowFeatures(BaseEstimator, TransformerMixin):
    """
    Extract derived features from basic flow statistics.
    
    Computes ratios, rates, and other derived metrics that are
    commonly useful for anomaly detection.
    """
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit transformer (no-op for deterministic features)."""
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived flow features.
        
        Args:
            X: DataFrame with flow data
            
        Returns:
            DataFrame with additional feature columns
        """
        X = X.copy()
        
        # Byte ratios and totals
        X['total_bytes'] = X['orig_bytes'] + X['resp_bytes']
        X['flow_ratio'] = X['orig_bytes'] / (X['resp_bytes'] + 1)  # +1 to avoid div by zero
        
        # Packet statistics  
        X['total_pkts'] = X['orig_pkts'] + X['resp_pkts']
        X['pkt_size_avg'] = X['total_bytes'] / (X['total_pkts'] + 1)
        
        # Rate calculations (bytes per second)
        X['orig_rate'] = X['orig_bytes'] / (X['duration'] + 0.001)  # Avoid div by zero
        X['resp_rate'] = X['resp_bytes'] / (X['duration'] + 0.001)
        
        # Asymmetry metrics
        X['byte_asymmetry'] = (X['orig_bytes'] - X['resp_bytes']) / (X['total_bytes'] + 1)
        X['pkt_asymmetry'] = (X['orig_pkts'] - X['resp_pkts']) / (X['total_pkts'] + 1)
        
        return X


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    Encode categorical variables for ML models.
    
    Handles protocol, service, and connection state encoding with
    proper handling of unknown categories.
    """
    
    def __init__(self, categorical_cols: List[str] = None, 
                 handle_unknown: str = 'ignore'):
        """
        Initialize categorical encoder.
        
        Args:
            categorical_cols: List of columns to encode
            handle_unknown: How to handle unknown categories in transform
        """
        self.categorical_cols = categorical_cols or ['proto', 'service', 'conn_state']
        self.handle_unknown = handle_unknown
        self.encoder = None
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit the encoder on categorical columns."""
        # Fill missing values for categorical columns
        X_filled = X[self.categorical_cols].fillna('unknown')
        
        self.encoder = OneHotEncoder(
            handle_unknown=self.handle_unknown,
            sparse_output=False,
            dtype=np.float32
        )
        self.encoder.fit(X_filled)
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform categorical columns to one-hot encoded features."""
        if self.encoder is None:
            raise ValueError("Encoder not fitted. Call fit() first.")
        
        X = X.copy()
        
        # Fill missing values
        X_filled = X[self.categorical_cols].fillna('unknown')
        
        # Transform categorical columns
        encoded = self.encoder.transform(X_filled)
        
        # Create feature names
        feature_names = self.encoder.get_feature_names_out(self.categorical_cols)
        
        # Create DataFrame with encoded features
        encoded_df = pd.DataFrame(encoded, columns=feature_names, index=X.index)
        
        # Drop original categorical columns and add encoded ones
        X_transformed = X.drop(columns=self.categorical_cols)
        X_transformed = pd.concat([X_transformed, encoded_df], axis=1)
        
        return X_transformed


class FlowPreprocessor:
    """
    Complete preprocessing pipeline for network flow data.
    
    Combines all feature engineering steps into a single pipeline
    that can be easily applied to raw Zeek flow data.
    """
    
    def __init__(self, 
                 rolling_window_s: int = 5,
                 rolling_stats: List[str] = None,
                 categorical_cols: List[str] = None,
                 numeric_cols: List[str] = None):
        """
        Initialize the complete preprocessing pipeline.
        
        Args:
            rolling_window_s: Window size for rolling statistics
            rolling_stats: Statistics to compute in rolling window
            categorical_cols: Categorical columns to encode
            numeric_cols: Numeric columns to scale
        """
        self.rolling_window_s = rolling_window_s
        self.rolling_stats = rolling_stats or ['mean']
        self.categorical_cols = categorical_cols or ['proto', 'service', 'conn_state']
        self.numeric_cols = numeric_cols or [
            'orig_bytes', 'resp_bytes', 'duration', 'orig_pkts', 'resp_pkts'
        ]
        
        # Initialize transformers
        self.rolling_transformer = RollingStats(
            span_s=rolling_window_s, 
            stats=rolling_stats
        )
        self.flow_features = FlowFeatures()
        self.categorical_encoder = CategoricalEncoder(categorical_cols)
        self.scaler = StandardScaler()
        
        self.is_fitted = False
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit all transformers in the pipeline."""
        # Apply rolling stats
        X_rolled = self.rolling_transformer.fit_transform(X)
        
        # Add derived features
        X_features = self.flow_features.fit_transform(X_rolled)
        
        # Fit categorical encoder
        self.categorical_encoder.fit(X_features)
        X_encoded = self.categorical_encoder.transform(X_features)
        
        # Fit scaler on numeric columns (including new rolling features)
        numeric_cols_expanded = []
        for col in X_encoded.columns:
            if X_encoded[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                numeric_cols_expanded.append(col)
        
        self.numeric_cols_final = numeric_cols_expanded
        if numeric_cols_expanded:
            self.scaler.fit(X_encoded[numeric_cols_expanded])
        
        self.is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the complete preprocessing pipeline."""
        if not self.is_fitted:
            raise ValueError("Pipeline not fitted. Call fit() first.")
        
        # Apply transformations in sequence
        X_transformed = self.rolling_transformer.transform(X)
        X_transformed = self.flow_features.transform(X_transformed)
        X_transformed = self.categorical_encoder.transform(X_transformed)
        
        # Scale numeric features
        if self.numeric_cols_final:
            X_transformed[self.numeric_cols_final] = self.scaler.transform(
                X_transformed[self.numeric_cols_final]
            )
        
        return X_transformed
    
    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)

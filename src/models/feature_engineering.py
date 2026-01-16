"""Feature engineering pipeline for churn prediction."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils.logging import logger


class FeatureEngineer:
    """Feature engineering and transformation pipeline.
    
    Handles categorical encoding, numerical scaling, and train/test splitting
    with proper handling of class imbalance through stratification.
    """
    
    def __init__(
        self,
        categorical_cols: Optional[List[str]] = None,
        numerical_cols: Optional[List[str]] = None,
        random_state: int = 42,
    ):
        """Initialize the feature engineer.
        
        Args:
            categorical_cols: List of categorical column names
            numerical_cols: List of numerical column names
            random_state: Random seed for reproducibility
        """
        self.categorical_cols = categorical_cols
        self.numerical_cols = numerical_cols
        self.random_state = random_state
        self.preprocessor: Optional[ColumnTransformer] = None
        self.feature_names_out: Optional[List[str]] = None
        self._is_fitted = False
        
    def _identify_columns(self, df: pd.DataFrame) -> None:
        """Automatically identify categorical and numerical columns.
        
        Args:
            df: Input dataframe
        """
        if self.categorical_cols is None:
            self.categorical_cols = df.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()
            
        if self.numerical_cols is None:
            self.numerical_cols = df.select_dtypes(
                include=["int64", "float64"]
            ).columns.tolist()
            
        logger.info(f"Identified {len(self.categorical_cols)} categorical columns")
        logger.info(f"Identified {len(self.numerical_cols)} numerical columns")
        
    def _build_preprocessor(self) -> ColumnTransformer:
        """Build the sklearn preprocessing pipeline.
        
        Returns:
            Configured ColumnTransformer
        """
        transformers = []
        
        # Numerical features: StandardScaler
        if self.numerical_cols:
            numerical_transformer = Pipeline(steps=[
                ("scaler", StandardScaler())
            ])
            transformers.append(
                ("num", numerical_transformer, self.numerical_cols)
            )
            
        # Categorical features: OneHotEncoder
        if self.categorical_cols:
            categorical_transformer = Pipeline(steps=[
                ("onehot", OneHotEncoder(
                    drop="first",  # Avoid multicollinearity
                    sparse_output=False,
                    handle_unknown="ignore"
                ))
            ])
            transformers.append(
                ("cat", categorical_transformer, self.categorical_cols)
            )
            
        return ColumnTransformer(
            transformers=transformers,
            remainder="drop",  # Drop columns not specified
            verbose_feature_names_out=False
        )
        
    def fit(self, X: pd.DataFrame) -> "FeatureEngineer":
        """Fit the feature engineering pipeline.
        
        Args:
            X: Feature dataframe
            
        Returns:
            Self for method chaining
        """
        logger.info("Fitting feature engineering pipeline...")
        
        self._identify_columns(X)
        self.preprocessor = self._build_preprocessor()
        self.preprocessor.fit(X)
        
        # Extract feature names after fitting
        self.feature_names_out = self._get_feature_names()
        self._is_fitted = True
        
        logger.info(f"Fitted preprocessor with {len(self.feature_names_out)} output features")
        return self
        
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform features using the fitted pipeline.
        
        Args:
            X: Feature dataframe
            
        Returns:
            Transformed feature array
        """
        if not self._is_fitted:
            raise RuntimeError("FeatureEngineer must be fitted before transform")
            
        return self.preprocessor.transform(X)
        
    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """Fit and transform in one step.
        
        Args:
            X: Feature dataframe
            
        Returns:
            Transformed feature array
        """
        self.fit(X)
        return self.transform(X)
        
    def _get_feature_names(self) -> List[str]:
        """Extract feature names from the fitted preprocessor.
        
        Returns:
            List of output feature names
        """
        feature_names = []
        
        for name, transformer, columns in self.preprocessor.transformers_:
            if name == "remainder":
                continue
                
            if name == "num":
                feature_names.extend(columns)
            elif name == "cat":
                # Get one-hot encoded feature names
                encoder = transformer.named_steps["onehot"]
                for i, col in enumerate(columns):
                    categories = encoder.categories_[i][1:]  # Skip first due to drop='first'
                    for cat in categories:
                        feature_names.append(f"{col}_{cat}")
                        
        return feature_names
        
    def get_feature_names(self) -> List[str]:
        """Get the output feature names.
        
        Returns:
            List of feature names
        """
        if self.feature_names_out is None:
            raise RuntimeError("FeatureEngineer must be fitted first")
        return self.feature_names_out
        
    def save(self, path: Path) -> None:
        """Save the fitted preprocessor to disk.
        
        Args:
            path: Directory path to save artifacts
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save preprocessor
        joblib.dump(self.preprocessor, path / "feature_preprocessor.joblib")
        
        # Save feature names
        with open(path / "feature_names.json", "w") as f:
            json.dump({
                "feature_names": self.feature_names_out,
                "categorical_cols": self.categorical_cols,
                "numerical_cols": self.numerical_cols
            }, f, indent=2)
            
        logger.info(f"Saved feature engineering artifacts to {path}")
        
    @classmethod
    def load(cls, path: Path) -> "FeatureEngineer":
        """Load a fitted preprocessor from disk.
        
        Args:
            path: Directory path containing saved artifacts
            
        Returns:
            Loaded FeatureEngineer instance
        """
        path = Path(path)
        
        # Load feature names
        with open(path / "feature_names.json", "r") as f:
            feature_info = json.load(f)
            
        instance = cls(
            categorical_cols=feature_info["categorical_cols"],
            numerical_cols=feature_info["numerical_cols"]
        )
        
        instance.preprocessor = joblib.load(path / "feature_preprocessor.joblib")
        instance.feature_names_out = feature_info["feature_names"]
        instance._is_fitted = True
        
        logger.info(f"Loaded feature engineering artifacts from {path}")
        return instance


def create_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Create stratified train/validation/test splits.
    
    Preserves class distribution across all splits, critical for
    imbalanced churn prediction datasets.
    
    Args:
        X: Feature dataframe
        y: Target series
        test_size: Proportion for test set (from total)
        val_size: Proportion for validation set (from remaining after test)
        random_state: Random seed
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    logger.info(f"Creating stratified splits: test={test_size}, val={val_size}")
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )
    
    # Second split: separate validation from training
    # Adjust val_size to account for test already removed
    adjusted_val_size = val_size / (1 - test_size)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=adjusted_val_size,
        stratify=y_temp,
        random_state=random_state
    )
    
    logger.info(f"Split sizes - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    logger.info(f"Class distribution - Train: {y_train.mean():.3f}, Val: {y_val.mean():.3f}, Test: {y_test.mean():.3f}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def prepare_features(
    df: pd.DataFrame,
    target_col: str = "Churn",
    exclude_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare features and target from raw dataframe.
    
    Args:
        df: Input dataframe
        target_col: Name of target column
        exclude_cols: Columns to exclude (e.g., customerID)
        
    Returns:
        Tuple of (features, target)
    """
    exclude_cols = exclude_cols or ["customerID"]
    
    feature_cols = [
        col for col in df.columns 
        if col not in [target_col] + exclude_cols
    ]
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    logger.info(f"Prepared {len(feature_cols)} features, target column: {target_col}")
    return X, y

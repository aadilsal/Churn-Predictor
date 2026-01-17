"""Unit tests for model evaluation functions.

Tests src/models/evaluation.py:
- evaluate_model
- find_optimal_threshold
- analyze_threshold_tradeoffs
"""

import numpy as np
import pandas as pd
import pytest

from src.models.evaluation import (
    analyze_threshold_tradeoffs,
    evaluate_model,
    find_optimal_threshold,
)


@pytest.mark.unit
class TestEvaluateModel:
    """Tests for evaluate_model function."""

    def test_returns_dict(self):
        """Test that function returns a dictionary."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_proba = np.array([0.2, 0.3, 0.7, 0.8])
        
        result = evaluate_model(y_true, y_pred, y_proba)
        
        assert isinstance(result, dict)

    def test_contains_core_metrics(self):
        """Test that result contains core evaluation metrics."""
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0, 1, 0, 1, 1])
        y_proba = np.array([0.2, 0.6, 0.4, 0.8, 0.9])
        
        result = evaluate_model(y_true, y_pred, y_proba)
        
        assert "accuracy" in result
        assert "precision" in result
        assert "recall" in result
        assert "f1" in result or "f1_score" in result
        assert "roc_auc" in result

    def test_perfect_predictions(self):
        """Test evaluation with perfect predictions."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 1, 1, 1])
        y_proba = np.array([0.1, 0.1, 0.2, 0.8, 0.9, 0.95])
        
        result = evaluate_model(y_true, y_pred, y_proba)
        
        assert result["accuracy"] == 1.0
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0

    def test_model_name_parameter(self):
        """Test that model_name parameter is accepted."""
        y_true = np.array([0, 1])
        y_pred = np.array([0, 1])
        y_proba = np.array([0.2, 0.8])
        
        # Should not raise error
        result = evaluate_model(y_true, y_pred, y_proba, model_name="TestModel")
        
        assert result is not None

    def test_handles_imbalanced_data(self):
        """Test with highly imbalanced data."""
        # 90% class 0, 10% class 1
        y_true = np.array([0] * 90 + [1] * 10)
        y_pred = np.array([0] * 90 + [1] * 10)
        y_proba = np.concatenate([
            np.random.uniform(0, 0.3, 90),
            np.random.uniform(0.7, 1.0, 10)
        ])
        
        result = evaluate_model(y_true, y_pred, y_proba)
        
        # Should handle without errors
        assert result["accuracy"] == 1.0


@pytest.mark.unit
class TestFindOptimalThreshold:
    """Tests for find_optimal_threshold function."""

    def test_returns_tuple(self):
        """Test that function returns tuple."""
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.2, 0.4, 0.6, 0.8])
        
        result = find_optimal_threshold(y_true, y_proba)
        
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_threshold_in_valid_range(self):
        """Test that threshold is between 0 and 1."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        
        threshold, _ = find_optimal_threshold(y_true, y_proba)
        
        assert 0.0 <= threshold <= 1.0

    def test_f1_method(self):
        """Test F1 optimization method."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        
        threshold, metrics = find_optimal_threshold(y_true, y_proba, method="f1")
        
        assert threshold is not None
        assert isinstance(metrics, dict)

    def test_youden_method(self):
        """Test Youden's J optimization method."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        
        threshold, metrics = find_optimal_threshold(y_true, y_proba, method="youden")
        
        assert threshold is not None

    def test_cost_method(self):
        """Test cost-based optimization method."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        
        threshold, metrics = find_optimal_threshold(
            y_true, y_proba, method="cost", cost_fp=1.0, cost_fn=5.0
        )
        
        assert threshold is not None

    def test_different_methods_may_differ(self):
        """Test that different methods can produce different thresholds."""
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_proba = np.array([0.1, 0.15, 0.2, 0.35, 0.45, 0.55, 0.6, 0.7, 0.8, 0.9])
        
        threshold_f1, _ = find_optimal_threshold(y_true, y_proba, method="f1")
        threshold_youden, _ = find_optimal_threshold(y_true, y_proba, method="youden")
        
        # Both should be valid thresholds (may or may not be equal)
        assert 0.0 <= threshold_f1 <= 1.0
        assert 0.0 <= threshold_youden <= 1.0


@pytest.mark.unit
class TestAnalyzeThresholdTradeoffs:
    """Tests for analyze_threshold_tradeoffs function."""

    def test_returns_dataframe(self):
        """Test that function returns DataFrame."""
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.2, 0.4, 0.6, 0.8])
        
        result = analyze_threshold_tradeoffs(y_true, y_proba)
        
        assert isinstance(result, pd.DataFrame)

    def test_contains_threshold_column(self):
        """Test that result has threshold column."""
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.2, 0.4, 0.6, 0.8])
        
        result = analyze_threshold_tradeoffs(y_true, y_proba)
        
        assert "threshold" in result.columns

    def test_threshold_range(self):
        """Test that thresholds span valid range."""
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.2, 0.4, 0.6, 0.8])
        
        result = analyze_threshold_tradeoffs(y_true, y_proba)
        
        thresholds = result["threshold"].values
        assert thresholds.min() >= 0.0
        assert thresholds.max() <= 1.0

    def test_contains_metric_columns(self):
        """Test that result contains key metric columns."""
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.2, 0.4, 0.6, 0.8])
        
        result = analyze_threshold_tradeoffs(y_true, y_proba)
        
        # Should contain precision, recall, f1 at minimum
        columns = result.columns.tolist()
        assert any("precision" in col.lower() for col in columns)
        assert any("recall" in col.lower() for col in columns)

    def test_multiple_rows(self):
        """Test that result has multiple threshold rows."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        
        result = analyze_threshold_tradeoffs(y_true, y_proba)
        
        assert len(result) > 1

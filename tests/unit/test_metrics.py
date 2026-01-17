"""Unit tests for metrics utilities.

Tests src/utils/metrics.py:
- calculate_classification_metrics
- precision_at_k
- calculate_calibration_error
- calculate_business_metrics
"""

import numpy as np
import pytest

from src.utils.metrics import (
    calculate_business_metrics,
    calculate_calibration_error,
    calculate_classification_metrics,
    precision_at_k,
)


@pytest.mark.unit
class TestCalculateClassificationMetrics:
    """Tests for calculate_classification_metrics function."""

    def test_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.8, 0.9, 0.95])
        
        metrics = calculate_classification_metrics(y_true, y_pred, y_proba)
        
        assert metrics["accuracy"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1_score"] == 1.0

    def test_all_wrong_predictions(self):
        """Test metrics with all wrong predictions."""
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([1, 1, 0, 0, 0])
        y_proba = np.array([0.9, 0.8, 0.2, 0.1, 0.15])
        
        metrics = calculate_classification_metrics(y_true, y_pred, y_proba)
        
        assert metrics["accuracy"] == 0.0
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0

    def test_balanced_dataset(self):
        """Test with balanced predictions."""
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        y_proba = np.array([0.2, 0.3, 0.6, 0.7, 0.4, 0.45, 0.75, 0.8])
        
        metrics = calculate_classification_metrics(y_true, y_pred, y_proba)
        
        assert metrics["accuracy"] == 0.5
        assert metrics["true_positives"] == 2
        assert metrics["true_negatives"] == 2
        assert metrics["false_positives"] == 2
        assert metrics["false_negatives"] == 2

    def test_returns_expected_keys(self, mock_labels, mock_model_predictions):
        """Test that all expected metric keys are returned."""
        y_pred = (mock_model_predictions >= 0.5).astype(int)
        
        metrics = calculate_classification_metrics(
            mock_labels, y_pred, mock_model_predictions
        )
        
        expected_keys = [
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "roc_auc",
            "true_positives",
            "true_negatives",
            "false_positives",
            "false_negatives",
            "specificity",
        ]
        
        for key in expected_keys:
            assert key in metrics

    def test_roc_auc_range(self, mock_labels, mock_model_predictions):
        """Test that ROC AUC is in valid range."""
        y_pred = (mock_model_predictions >= 0.5).astype(int)
        
        metrics = calculate_classification_metrics(
            mock_labels, y_pred, mock_model_predictions
        )
        
        assert 0.0 <= metrics["roc_auc"] <= 1.0

    def test_specificity_calculation(self):
        """Test specificity calculation."""
        y_true = np.array([0, 0, 0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 1, 1, 1])  # 3 TN, 1 FP
        y_proba = np.array([0.1, 0.2, 0.3, 0.6, 0.85, 0.9])
        
        metrics = calculate_classification_metrics(y_true, y_pred, y_proba)
        
        # Specificity = TN / (TN + FP) = 3 / 4 = 0.75
        assert metrics["specificity"] == 0.75


@pytest.mark.unit
class TestPrecisionAtK:
    """Tests for precision_at_k function."""

    def test_top_10_percent(self):
        """Test precision at top 10%."""
        y_true = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.15, 0.25, 0.3, 0.35, 0.4, 0.9, 0.85, 0.95])
        
        # Top 10% = 1 sample (the highest prob: 0.95, true label: 1)
        precision = precision_at_k(y_true, y_proba, k=0.1)
        
        assert precision == 1.0

    def test_top_50_percent(self):
        """Test precision at top 50%."""
        y_true = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.15, 0.25, 0.3, 0.35, 0.4, 0.9, 0.85, 0.95])
        
        # Top 50% = 5 samples (highest 5 probs)
        precision = precision_at_k(y_true, y_proba, k=0.5)
        
        # All 3 churners are in top 5
        assert precision == 3 / 5

    def test_returns_float(self):
        """Test that result is a float."""
        y_true = np.array([0, 1, 0, 1])
        y_proba = np.array([0.2, 0.8, 0.3, 0.9])
        
        result = precision_at_k(y_true, y_proba, k=0.5)
        
        assert isinstance(result, float)

    def test_perfect_ranking(self):
        """Test with perfectly ranked predictions."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        
        precision = precision_at_k(y_true, y_proba, k=0.5)
        
        assert precision == 1.0

    def test_worst_ranking(self):
        """Test with worst possible ranking."""
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_proba = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        
        precision = precision_at_k(y_true, y_proba, k=0.5)
        
        assert precision == 0.0


@pytest.mark.unit
class TestCalculateCalibrationError:
    """Tests for calculate_calibration_error function."""

    def test_perfectly_calibrated(self):
        """Test ECE for perfectly calibrated predictions."""
        # Predictions that match actual outcomes perfectly
        n = 1000
        np.random.seed(42)
        y_proba = np.random.uniform(0, 1, n)
        y_true = (np.random.random(n) < y_proba).astype(int)
        
        ece, _, _ = calculate_calibration_error(y_true, y_proba)
        
        # ECE should be close to 0 for calibrated predictions
        assert ece < 0.1

    def test_returns_correct_tuple(self):
        """Test that function returns correct tuple structure."""
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.2, 0.3, 0.7, 0.8])
        
        result = calculate_calibration_error(y_true, y_proba)
        
        assert len(result) == 3
        ece, bin_acc, bin_conf = result
        
        assert isinstance(ece, float)
        assert isinstance(bin_acc, np.ndarray)
        assert isinstance(bin_conf, np.ndarray)

    def test_ece_range(self):
        """Test that ECE is in valid range [0, 1]."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        
        ece, _, _ = calculate_calibration_error(y_true, y_proba)
        
        assert 0.0 <= ece <= 1.0

    def test_custom_bins(self):
        """Test with custom number of bins."""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_proba = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
        
        ece_5, acc_5, conf_5 = calculate_calibration_error(y_true, y_proba, n_bins=5)
        ece_10, acc_10, conf_10 = calculate_calibration_error(y_true, y_proba, n_bins=10)
        
        assert len(acc_5) == 5
        assert len(acc_10) == 10


@pytest.mark.unit
class TestCalculateBusinessMetrics:
    """Tests for calculate_business_metrics function."""

    def test_returns_expected_keys(self, mock_labels, mock_model_predictions):
        """Test that all expected keys are returned."""
        metrics = calculate_business_metrics(mock_labels, mock_model_predictions)
        
        expected_keys = [
            "revenue_at_risk",
            "customers_saved",
            "revenue_saved",
            "intervention_costs",
            "net_benefit",
            "roi_percentage",
        ]
        
        for key in expected_keys:
            assert key in metrics

    def test_revenue_at_risk_calculation(self):
        """Test revenue at risk calculation."""
        y_true = np.array([0, 0, 0, 1, 1])  # 2 churners
        y_proba = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
        
        metrics = calculate_business_metrics(
            y_true, y_proba, revenue_per_customer=500.0
        )
        
        # Revenue at risk = 2 churners * $500 = $1000
        assert metrics["revenue_at_risk"] == 1000.0

    def test_zero_churners(self):
        """Test with no churners in data."""
        y_true = np.array([0, 0, 0, 0, 0])
        y_proba = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        
        metrics = calculate_business_metrics(y_true, y_proba)
        
        assert metrics["revenue_at_risk"] == 0.0
        assert metrics["customers_saved"] == 0.0

    def test_custom_parameters(self):
        """Test with custom business parameters."""
        y_true = np.array([1, 1, 1, 1, 1])  # All churners
        y_proba = np.array([0.9, 0.85, 0.8, 0.75, 0.7])
        
        metrics = calculate_business_metrics(
            y_true,
            y_proba,
            revenue_per_customer=2000.0,
            intervention_cost=100.0,
            intervention_success_rate=0.5,
        )
        
        # Should use custom values
        assert metrics["revenue_at_risk"] == 10000.0  # 5 * 2000

    def test_positive_roi(self):
        """Test that good predictions give positive ROI."""
        # Perfect ranking of churners
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_proba = np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.7, 0.75, 0.8, 0.85, 0.9])
        
        metrics = calculate_business_metrics(
            y_true,
            y_proba,
            revenue_per_customer=1000.0,
            intervention_cost=50.0,
            intervention_success_rate=0.3,
        )
        
        assert metrics["net_benefit"] > 0
        assert metrics["roi_percentage"] > 0

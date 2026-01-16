"""Model comparison framework for churn prediction."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.logging import logger


def compare_models(
    model_results: Dict[str, Dict[str, Any]],
    metrics_to_compare: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Compare multiple models across key metrics.
    
    Args:
        model_results: Dict of {model_name: metrics_dict}
        metrics_to_compare: Metrics to include in comparison
        
    Returns:
        DataFrame with model comparison
    """
    metrics_to_compare = metrics_to_compare or [
        "roc_auc", "pr_auc", "f1_score", "precision", "recall",
        "brier_score", "expected_calibration_error"
    ]
    
    rows = []
    for model_name, metrics in model_results.items():
        row = {"model": model_name}
        for metric in metrics_to_compare:
            # Handle both direct and prefixed metrics
            if metric in metrics:
                row[metric] = metrics[metric]
            elif f"val_{metric}" in metrics:
                row[metric] = metrics[f"val_{metric}"]
            elif f"mean_{metric}" in metrics:
                row[metric] = metrics[f"mean_{metric}"]
        rows.append(row)
        
    df = pd.DataFrame(rows)
    return df.set_index("model")


def compare_cv_results(
    cv_results: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:
    """Compare cross-validation results across models.
    
    Args:
        cv_results: Dict of {model_name: cv_results_dict}
        
    Returns:
        DataFrame with mean and std for each metric
    """
    rows = []
    for model_name, results in cv_results.items():
        row = {"model": model_name}
        
        for key, value in results.items():
            if key.startswith("mean_"):
                metric = key.replace("mean_", "")
                row[f"{metric}_mean"] = value
                std_key = f"std_{metric}"
                if std_key in results:
                    row[f"{metric}_std"] = results[std_key]
                    
        rows.append(row)
        
    return pd.DataFrame(rows).set_index("model")


def rank_models(
    comparison_df: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None,
    higher_is_better: Optional[Dict[str, bool]] = None,
) -> pd.DataFrame:
    """Rank models using weighted scoring.
    
    Args:
        comparison_df: Model comparison DataFrame
        weights: Dict of {metric: weight}
        higher_is_better: Dict of {metric: True/False}
        
    Returns:
        DataFrame with rankings
    """
    weights = weights or {
        "roc_auc": 0.25,
        "pr_auc": 0.20,
        "f1_score": 0.15,
        "recall": 0.15,
        "precision": 0.10,
        "brier_score": 0.10,
        "expected_calibration_error": 0.05,
    }
    
    higher_is_better = higher_is_better or {
        "roc_auc": True,
        "pr_auc": True,
        "f1_score": True,
        "precision": True,
        "recall": True,
        "brier_score": False,
        "expected_calibration_error": False,
    }
    
    df = comparison_df.copy()
    
    # Calculate normalized scores
    scores = np.zeros(len(df))
    
    for metric, weight in weights.items():
        if metric not in df.columns:
            continue
            
        values = df[metric].values
        
        # Normalize to 0-1 range
        min_val, max_val = values.min(), values.max()
        if max_val > min_val:
            normalized = (values - min_val) / (max_val - min_val)
        else:
            normalized = np.ones_like(values) * 0.5
            
        # Flip if lower is better
        if not higher_is_better.get(metric, True):
            normalized = 1 - normalized
            
        scores += normalized * weight
        
    df["weighted_score"] = scores
    df["rank"] = df["weighted_score"].rank(ascending=False).astype(int)
    
    return df.sort_values("rank")


def select_best_model(
    comparison_df: pd.DataFrame,
    primary_metric: str = "roc_auc",
    stability_threshold: float = 0.03,
    cv_results: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Select best model with justification.
    
    Args:
        comparison_df: Model comparison DataFrame
        primary_metric: Primary metric for selection
        stability_threshold: Maximum acceptable std deviation
        cv_results: Cross-validation results for stability check
        
    Returns:
        Tuple of (best_model_name, selection_justification)
    """
    ranked = rank_models(comparison_df)
    
    justification = {
        "ranking": ranked[["weighted_score", "rank"]].to_dict(),
        "primary_metric": primary_metric,
        "considerations": [],
    }
    
    # Get top ranked model
    best_model = ranked.index[0]
    
    # Check stability if CV results provided
    if cv_results and best_model in cv_results:
        std_key = f"std_{primary_metric}"
        if std_key in cv_results[best_model]:
            std = cv_results[best_model][std_key]
            if std > stability_threshold:
                justification["considerations"].append(
                    f"Warning: {best_model} has high variance (std={std:.4f})"
                )
                
    # Compare top 2 models
    if len(ranked) > 1:
        runner_up = ranked.index[1]
        score_diff = ranked.loc[best_model, "weighted_score"] - ranked.loc[runner_up, "weighted_score"]
        justification["considerations"].append(
            f"Score difference vs {runner_up}: {score_diff:.4f}"
        )
        
    logger.info(f"Selected best model: {best_model}")
    
    return best_model, justification


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Create visual comparison of models.
    
    Args:
        comparison_df: Model comparison DataFrame
        metrics: Metrics to plot
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    metrics = metrics or ["roc_auc", "pr_auc", "f1_score", "precision", "recall"]
    
    # Filter to available metrics
    metrics = [m for m in metrics if m in comparison_df.columns]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(metrics))
    width = 0.35
    n_models = len(comparison_df)
    
    for i, (model_name, row) in enumerate(comparison_df.iterrows()):
        offset = (i - n_models / 2 + 0.5) * width
        values = [row[m] for m in metrics]
        ax.bar(x + offset, values, width, label=model_name)
        
    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved comparison plot to {save_path}")
        
    return fig


def create_performance_report(
    model_results: Dict[str, Dict[str, Any]],
    cv_results: Dict[str, Dict[str, Any]],
    best_model: str,
    justification: Dict[str, Any],
    output_path: Path,
) -> None:
    """Generate markdown performance report.
    
    Args:
        model_results: Model evaluation results
        cv_results: Cross-validation results
        best_model: Selected model name
        justification: Selection justification
        output_path: Path to save report
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    lines = [
        "# Model Performance Report",
        "",
        "## Summary",
        "",
        f"**Selected Model:** {best_model}",
        "",
        "## Model Comparison",
        "",
    ]
    
    # Create comparison table
    comparison_df = compare_models(model_results)
    lines.append(comparison_df.to_markdown())
    lines.append("")
    
    # CV Results
    lines.append("## Cross-Validation Stability")
    lines.append("")
    
    for model_name, results in cv_results.items():
        mean_auc = results.get("mean_roc_auc", "N/A")
        std_auc = results.get("std_roc_auc", "N/A")
        if isinstance(mean_auc, float):
            lines.append(f"- **{model_name}**: ROC-AUC = {mean_auc:.4f} ± {std_auc:.4f}")
        else:
            lines.append(f"- **{model_name}**: ROC-AUC = {mean_auc}")
            
    lines.append("")
    
    # Selection Justification
    lines.append("## Selection Rationale")
    lines.append("")
    
    for consideration in justification.get("considerations", []):
        lines.append(f"- {consideration}")
        
    lines.append("")
    
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
        
    logger.info(f"Saved performance report to {output_path}")

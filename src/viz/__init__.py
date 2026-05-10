"""Visualization utilities for probabilistic models."""

import logging
from typing import Optional, Tuple, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_predictions(
    X: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: Optional[np.ndarray] = None,
    title: str = "Model Predictions",
    save_path: Optional[str] = None
) -> None:
    """Plot model predictions with uncertainty bands.
    
    Args:
        X: Input features
        y_true: True target values
        y_pred: Predicted values
        y_std: Prediction standard deviations
        title: Plot title
        save_path: Path to save plot
    """
    plt.figure(figsize=(10, 6))
    
    # Sort by X for better visualization
    sort_idx = np.argsort(X.flatten())
    X_sorted = X[sort_idx]
    y_true_sorted = y_true[sort_idx]
    y_pred_sorted = y_pred[sort_idx]
    
    # Plot true values
    plt.scatter(X_sorted, y_true_sorted, alpha=0.6, label="True values", color="blue")
    
    # Plot predictions
    plt.plot(X_sorted, y_pred_sorted, color="red", linewidth=2, label="Predictions")
    
    # Plot uncertainty bands if available
    if y_std is not None:
        y_std_sorted = y_std[sort_idx]
        
        # 95% confidence interval
        upper_bound = y_pred_sorted + 1.96 * y_std_sorted
        lower_bound = y_pred_sorted - 1.96 * y_std_sorted
        
        plt.fill_between(
            X_sorted.flatten(),
            lower_bound,
            upper_bound,
            alpha=0.3,
            color="red",
            label="95% Confidence Interval"
        )
        
        # 68% confidence interval
        upper_bound_68 = y_pred_sorted + y_std_sorted
        lower_bound_68 = y_pred_sorted - y_std_sorted
        
        plt.fill_between(
            X_sorted.flatten(),
            lower_bound_68,
            upper_bound_68,
            alpha=0.5,
            color="red",
            label="68% Confidence Interval"
        )
    
    plt.xlabel("Input Features")
    plt.ylabel("Target Values")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logging.info(f"Plot saved to {save_path}")
    
    plt.show()


def plot_training_history(
    losses: list,
    title: str = "Training History",
    save_path: Optional[str] = None
) -> None:
    """Plot training loss history.
    
    Args:
        losses: List of loss values
        title: Plot title
        save_path: Path to save plot
    """
    plt.figure(figsize=(10, 6))
    
    plt.plot(losses, linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    # Add smoothed line
    if len(losses) > 10:
        window_size = min(50, len(losses) // 10)
        smoothed = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(losses)), smoothed, 
                linewidth=2, alpha=0.7, label="Smoothed")
        plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logging.info(f"Training history plot saved to {save_path}")
    
    plt.show()


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: Optional[np.ndarray] = None,
    title: str = "Residual Analysis",
    save_path: Optional[str] = None
) -> None:
    """Plot residual analysis.
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        y_std: Prediction standard deviations
        title: Plot title
        save_path: Path to save plot
    """
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Residuals vs predictions
    axes[0, 0].scatter(y_pred, residuals, alpha=0.6)
    axes[0, 0].axhline(y=0, color='red', linestyle='--')
    axes[0, 0].set_xlabel("Predictions")
    axes[0, 0].set_ylabel("Residuals")
    axes[0, 0].set_title("Residuals vs Predictions")
    axes[0, 0].grid(True, alpha=0.3)
    
    # Histogram of residuals
    axes[0, 1].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel("Residuals")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].set_title("Distribution of Residuals")
    axes[0, 1].grid(True, alpha=0.3)
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title("Q-Q Plot")
    axes[1, 0].grid(True, alpha=0.3)
    
    # Standardized residuals vs predictions
    if y_std is not None:
        standardized_residuals = residuals / y_std
        axes[1, 1].scatter(y_pred, standardized_residuals, alpha=0.6)
        axes[1, 1].axhline(y=0, color='red', linestyle='--')
        axes[1, 1].axhline(y=2, color='orange', linestyle='--', alpha=0.7)
        axes[1, 1].axhline(y=-2, color='orange', linestyle='--', alpha=0.7)
        axes[1, 1].set_xlabel("Predictions")
        axes[1, 1].set_ylabel("Standardized Residuals")
        axes[1, 1].set_title("Standardized Residuals vs Predictions")
    else:
        axes[1, 1].scatter(y_pred, residuals, alpha=0.6)
        axes[1, 1].axhline(y=0, color='red', linestyle='--')
        axes[1, 1].set_xlabel("Predictions")
        axes[1, 1].set_ylabel("Residuals")
        axes[1, 1].set_title("Residuals vs Predictions")
    
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logging.info(f"Residual analysis plot saved to {save_path}")
    
    plt.show()


def plot_uncertainty_calibration(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    title: str = "Uncertainty Calibration",
    save_path: Optional[str] = None
) -> None:
    """Plot uncertainty calibration analysis.
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        y_std: Prediction standard deviations
        title: Plot title
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Reliability diagram
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    accuracies = []
    confidences = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_std >= bin_lower) & (y_std < bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = (np.abs(y_true - y_pred)[in_bin] <= bin_upper).mean()
            avg_confidence_in_bin = y_std[in_bin].mean()
            
            accuracies.append(accuracy_in_bin)
            confidences.append(avg_confidence_in_bin)
        else:
            accuracies.append(0)
            confidences.append(0)
    
    axes[0, 0].plot(confidences, accuracies, 'o-', linewidth=2, markersize=6)
    axes[0, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[0, 0].set_xlabel("Confidence")
    axes[0, 0].set_ylabel("Accuracy")
    axes[0, 0].set_title("Reliability Diagram")
    axes[0, 0].grid(True, alpha=0.3)
    
    # Coverage vs confidence level
    confidence_levels = np.linspace(0.1, 0.99, 20)
    coverages = []
    
    for conf_level in confidence_levels:
        z_score = {0.1: 0.13, 0.2: 0.25, 0.3: 0.39, 0.4: 0.52, 0.5: 0.67,
                  0.6: 0.84, 0.7: 1.04, 0.8: 1.28, 0.9: 1.64, 0.95: 1.96, 0.99: 2.58}
        
        # Interpolate z-score
        z = np.interp(conf_level, list(z_score.keys()), list(z_score.values()))
        
        lower_bound = y_pred - z * y_std
        upper_bound = y_pred + z * y_std
        coverage = np.mean((y_true >= lower_bound) & (y_true <= upper_bound))
        coverages.append(coverage)
    
    axes[0, 1].plot(confidence_levels, coverages, 'o-', linewidth=2, markersize=4)
    axes[0, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[0, 1].set_xlabel("Confidence Level")
    axes[0, 1].set_ylabel("Coverage")
    axes[0, 1].set_title("Coverage vs Confidence Level")
    axes[0, 1].grid(True, alpha=0.3)
    
    # Uncertainty vs error
    errors = np.abs(y_true - y_pred)
    axes[1, 0].scatter(y_std, errors, alpha=0.6)
    axes[1, 0].plot([0, y_std.max()], [0, y_std.max()], 'r--', alpha=0.7)
    axes[1, 0].set_xlabel("Predicted Uncertainty")
    axes[1, 0].set_ylabel("Actual Error")
    axes[1, 0].set_title("Uncertainty vs Error")
    axes[1, 0].grid(True, alpha=0.3)
    
    # Distribution of uncertainties
    axes[1, 1].hist(y_std, bins=30, alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel("Predicted Uncertainty")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].set_title("Distribution of Uncertainties")
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logging.info(f"Uncertainty calibration plot saved to {save_path}")
    
    plt.show()


def create_interactive_plot(
    X: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: Optional[np.ndarray] = None,
    title: str = "Interactive Predictions"
) -> go.Figure:
    """Create interactive plot using Plotly.
    
    Args:
        X: Input features
        y_true: True target values
        y_pred: Predicted values
        y_std: Prediction standard deviations
        title: Plot title
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    # Add true values
    fig.add_trace(go.Scatter(
        x=X.flatten(),
        y=y_true,
        mode='markers',
        name='True values',
        marker=dict(color='blue', size=6, opacity=0.6)
    ))
    
    # Add predictions
    fig.add_trace(go.Scatter(
        x=X.flatten(),
        y=y_pred,
        mode='lines',
        name='Predictions',
        line=dict(color='red', width=2)
    ))
    
    # Add uncertainty bands if available
    if y_std is not None:
        # 95% confidence interval
        upper_bound = y_pred + 1.96 * y_std
        lower_bound = y_pred - 1.96 * y_std
        
        fig.add_trace(go.Scatter(
            x=X.flatten(),
            y=upper_bound,
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=X.flatten(),
            y=lower_bound,
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(255,0,0,0.2)',
            name='95% Confidence Interval',
            hoverinfo='skip'
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Input Features",
        yaxis_title="Target Values",
        hovermode='closest',
        showlegend=True
    )
    
    return fig


def plot_model_comparison(
    results: Dict[str, Dict[str, float]],
    metrics: list = ["rmse", "mae", "r2", "coverage_95"],
    title: str = "Model Comparison",
    save_path: Optional[str] = None
) -> None:
    """Plot model comparison results.
    
    Args:
        results: Dictionary of model results
        metrics: List of metrics to plot
        title: Plot title
        save_path: Path to save plot
    """
    models = list(results.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics[:4]):
        values = [results[model].get(metric, 0) for model in models]
        
        bars = axes[i].bar(models, values, alpha=0.7)
        axes[i].set_title(f"{metric.upper()}")
        axes[i].set_ylabel(metric.upper())
        axes[i].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logging.info(f"Model comparison plot saved to {save_path}")
    
    plt.show()

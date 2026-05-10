"""Evaluation metrics for probabilistic models."""

import logging
from typing import Dict, Tuple, Union, Optional

import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def regression_metrics(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    y_std: Optional[Union[np.ndarray, torch.Tensor]] = None
) -> Dict[str, float]:
    """Calculate regression evaluation metrics.
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        y_std: Prediction standard deviations (optional)
        
    Returns:
        Dictionary of metrics
    """
    # Convert to numpy if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if y_std is not None and isinstance(y_std, torch.Tensor):
        y_std = y_std.detach().cpu().numpy()
    
    # Basic regression metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # Symmetric Mean Absolute Percentage Error
    smape = np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) * 100
    
    metrics = {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mape": mape,
        "smape": smape
    }
    
    # Uncertainty metrics if available
    if y_std is not None:
        # Prediction interval coverage
        lower_bound = y_pred - 1.96 * y_std
        upper_bound = y_pred + 1.96 * y_std
        coverage = np.mean((y_true >= lower_bound) & (y_true <= upper_bound))
        
        # Calibration error (simplified)
        calibration_error = np.mean(np.abs(y_std - np.abs(y_true - y_pred)))
        
        metrics.update({
            "coverage_95": coverage,
            "calibration_error": calibration_error,
            "mean_uncertainty": np.mean(y_std)
        })
    
    return metrics


def uncertainty_metrics(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    y_std: Union[np.ndarray, torch.Tensor]
) -> Dict[str, float]:
    """Calculate uncertainty quantification metrics.
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        y_std: Prediction standard deviations
        
    Returns:
        Dictionary of uncertainty metrics
    """
    # Convert to numpy if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(y_std, torch.Tensor):
        y_std = y_std.detach().cpu().numpy()
    
    # Prediction interval coverage at different confidence levels
    coverage_metrics = {}
    for alpha in [0.5, 0.8, 0.9, 0.95, 0.99]:
        z_score = {0.5: 0.67, 0.8: 1.28, 0.9: 1.64, 0.95: 1.96, 0.99: 2.58}[alpha]
        
        lower_bound = y_pred - z_score * y_std
        upper_bound = y_pred + z_score * y_std
        coverage = np.mean((y_true >= lower_bound) & (y_true <= upper_bound))
        
        coverage_metrics[f"coverage_{int(alpha*100)}"] = coverage
    
    # Calibration metrics
    residuals = np.abs(y_true - y_pred)
    
    # Expected vs actual uncertainty
    expected_uncertainty = np.mean(y_std)
    actual_uncertainty = np.std(residuals)
    calibration_ratio = actual_uncertainty / expected_uncertainty
    
    # Reliability diagram (simplified)
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0  # Expected Calibration Error
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_std >= bin_lower) & (y_std < bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = (residuals[in_bin] <= bin_upper).mean()
            avg_confidence_in_bin = y_std[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    metrics = {
        **coverage_metrics,
        "calibration_ratio": calibration_ratio,
        "expected_calibration_error": ece,
        "mean_prediction_std": np.mean(y_std),
        "std_prediction_std": np.std(y_std)
    }
    
    return metrics


def bayesian_metrics(
    model,
    X: torch.Tensor,
    y: torch.Tensor,
    num_samples: int = 1000
) -> Dict[str, float]:
    """Calculate Bayesian-specific metrics.
    
    Args:
        model: Trained Bayesian model
        X: Input features
        y: True targets
        num_samples: Number of samples for Monte Carlo estimation
        
    Returns:
        Dictionary of Bayesian metrics
    """
    # Get predictions with uncertainty
    y_pred, y_std = model.predict(X, num_samples=num_samples, return_uncertainty=True)
    
    # Convert to numpy
    y_true = y.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    y_std = y_std.detach().cpu().numpy()
    
    # Log likelihood (approximate)
    log_likelihood = -0.5 * np.sum(
        np.log(2 * np.pi * y_std**2) + (y_true - y_pred)**2 / y_std**2
    )
    
    # Bayesian Information Criterion (approximate)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    bic = -2 * log_likelihood + n_params * np.log(len(y_true))
    
    # Akaike Information Criterion (approximate)
    aic = -2 * log_likelihood + 2 * n_params
    
    # Widely Applicable Information Criterion (approximate)
    waic = -2 * log_likelihood + 2 * np.sum(np.log(y_std))
    
    metrics = {
        "log_likelihood": log_likelihood,
        "bic": bic,
        "aic": aic,
        "waic": waic,
        "n_parameters": n_params
    }
    
    return metrics


def model_comparison_metrics(
    models: Dict[str, any],
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    num_samples: int = 1000
) -> Dict[str, Dict[str, float]]:
    """Compare multiple models and return comprehensive metrics.
    
    Args:
        models: Dictionary of model name -> model instance
        X_test: Test features
        y_test: Test targets
        num_samples: Number of samples for uncertainty estimation
        
    Returns:
        Dictionary of model name -> metrics
    """
    results = {}
    
    for name, model in models.items():
        logging.info(f"Evaluating model: {name}")
        
        # Get predictions
        if hasattr(model, 'predict') and 'uncertainty' in str(type(model.predict)):
            y_pred, y_std = model.predict(X_test, num_samples=num_samples, return_uncertainty=True)
        else:
            y_pred = model.predict(X_test)
            y_std = None
        
        # Calculate metrics
        regression_met = regression_metrics(y_test, y_pred, y_std)
        
        if y_std is not None:
            uncertainty_met = uncertainty_metrics(y_test, y_pred, y_std)
            bayesian_met = bayesian_metrics(model, X_test, y_test, num_samples)
            results[name] = {**regression_met, **uncertainty_met, **bayesian_met}
        else:
            results[name] = regression_met
    
    return results

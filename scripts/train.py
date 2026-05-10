#!/usr/bin/env python3
"""Main training script for probabilistic programming implementation."""

import logging
import argparse
from pathlib import Path
from typing import Dict, Any

import torch
import numpy as np
from omegaconf import OmegaConf

from src.utils import set_seed, get_device, setup_logging
from src.data import generate_linear_data, generate_nonlinear_data
from src.models import BayesianLinearRegression, BayesianNeuralNetwork
from src.train import PyroTrainer
from src.metrics import model_comparison_metrics
from src.viz import plot_predictions, plot_training_history, plot_model_comparison


def train_bayesian_linear_regression(
    config: Dict[str, Any],
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor
) -> Dict[str, Any]:
    """Train Bayesian Linear Regression model.
    
    Args:
        config: Configuration dictionary
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        
    Returns:
        Model and results
    """
    logging.info("Training Bayesian Linear Regression...")
    
    # Initialize model
    model = BayesianLinearRegression(
        input_dim=X_train.shape[1],
        prior_std=config.model.prior_std,
        noise_prior_std=config.model.noise_prior_std,
        device=get_device(config.experiment.device)
    )
    
    # Train model
    training_results = model.fit(
        X_train, y_train,
        num_epochs=config.training.num_epochs,
        learning_rate=config.training.learning_rate,
        verbose=True
    )
    
    # Make predictions
    y_pred, y_std = model.predict(X_test, return_uncertainty=True)
    
    # Calculate metrics
    from src.metrics import regression_metrics, uncertainty_metrics
    regression_met = regression_metrics(y_test, y_pred, y_std)
    uncertainty_met = uncertainty_metrics(y_test, y_pred, y_std)
    
    results = {
        "model": model,
        "training_results": training_results,
        "predictions": y_pred,
        "uncertainty": y_std,
        "metrics": {**regression_met, **uncertainty_met}
    }
    
    return results


def train_bayesian_neural_network(
    config: Dict[str, Any],
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor
) -> Dict[str, Any]:
    """Train Bayesian Neural Network model.
    
    Args:
        config: Configuration dictionary
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        
    Returns:
        Model and results
    """
    logging.info("Training Bayesian Neural Network...")
    
    # Initialize model
    model = BayesianNeuralNetwork(
        input_dim=X_train.shape[1],
        hidden_dim=config.model.hidden_dim,
        num_layers=config.model.num_layers,
        prior_std=config.model.prior_std,
        noise_prior_std=config.model.noise_prior_std,
        activation=config.model.activation,
        dropout=config.model.dropout,
        device=get_device(config.experiment.device)
    )
    
    # Train model
    training_results = model.fit(
        X_train, y_train,
        num_epochs=config.training.num_epochs,
        learning_rate=config.training.learning_rate,
        verbose=True
    )
    
    # Make predictions
    y_pred, y_std = model.predict(X_test, return_uncertainty=True)
    
    # Calculate metrics
    from src.metrics import regression_metrics, uncertainty_metrics
    regression_met = regression_metrics(y_test, y_pred, y_std)
    uncertainty_met = uncertainty_metrics(y_test, y_pred, y_std)
    
    results = {
        "model": model,
        "training_results": training_results,
        "predictions": y_pred,
        "uncertainty": y_std,
        "metrics": {**regression_met, **uncertainty_met}
    }
    
    return results


def train_classical_baselines(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor
) -> Dict[str, Dict[str, Any]]:
    """Train classical baseline models.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        
    Returns:
        Dictionary of baseline results
    """
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    logging.info("Training classical baselines...")
    
    baselines = {
        "OLS": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.1),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for name, model in baselines.items():
        logging.info(f"Training {name}...")
        
        # Convert to numpy for sklearn
        X_train_np = X_train.detach().cpu().numpy()
        y_train_np = y_train.detach().cpu().numpy()
        X_test_np = X_test.detach().cpu().numpy()
        y_test_np = y_test.detach().cpu().numpy()
        
        # Train model
        model.fit(X_train_np, y_train_np)
        
        # Make predictions
        y_pred = model.predict(X_test_np)
        
        # Calculate metrics
        mse = mean_squared_error(y_test_np, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_np, y_pred)
        r2 = r2_score(y_test_np, y_pred)
        
        results[name] = {
            "model": model,
            "predictions": torch.tensor(y_pred),
            "metrics": {
                "mse": mse,
                "rmse": rmse,
                "mae": mae,
                "r2": r2
            }
        }
    
    return results


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train probabilistic models")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--data-type", type=str, default="linear",
                       choices=["linear", "nonlinear"],
                       help="Type of synthetic data to generate")
    parser.add_argument("--output-dir", type=str, default="assets",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Load configuration
    config = OmegaConf.load(args.config)
    
    # Setup
    set_seed(config.experiment.seed)
    setup_logging(config.logging.level, config.logging.log_dir)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    logging.info("Starting probabilistic programming experiment")
    logging.info(f"Configuration: {config}")
    
    # Generate data
    if args.data_type == "linear":
        X_train, X_test, y_train, y_test = generate_linear_data(
            n_samples=config.data.n_samples,
            n_features=config.data.n_features,
            noise_std=config.data.noise_std,
            true_slope=config.data.true_slope,
            true_intercept=config.data.true_intercept,
            x_range=tuple(config.data.x_range),
            test_split=config.data.test_split,
            random_seed=config.experiment.seed
        )
    else:
        X_train, X_test, y_train, y_test = generate_nonlinear_data(
            n_samples=config.data.n_samples,
            noise_std=config.data.noise_std,
            function_type="sine",
            test_split=config.data.test_split,
            random_seed=config.experiment.seed
        )
    
    logging.info(f"Data shape: Train {X_train.shape}, Test {X_test.shape}")
    
    # Train models
    all_results = {}
    
    # Classical baselines
    baseline_results = train_classical_baselines(X_train, y_train, X_test, y_test)
    all_results.update(baseline_results)
    
    # Bayesian models
    if config.model._target_.endswith("BayesianLinearRegression"):
        bayesian_results = train_bayesian_linear_regression(
            config, X_train, y_train, X_test, y_test
        )
        all_results["BayesianLinearRegression"] = bayesian_results
    else:
        bayesian_results = train_bayesian_neural_network(
            config, X_train, y_train, X_test, y_test
        )
        all_results["BayesianNeuralNetwork"] = bayesian_results
    
    # Extract metrics for comparison
    comparison_metrics = {}
    for name, results in all_results.items():
        comparison_metrics[name] = results["metrics"]
    
    # Create visualizations
    logging.info("Creating visualizations...")
    
    # Plot predictions for Bayesian model
    bayesian_name = "BayesianLinearRegression" if "BayesianLinearRegression" in all_results else "BayesianNeuralNetwork"
    bayesian_results = all_results[bayesian_name]
    
    plot_predictions(
        X_test.detach().cpu().numpy(),
        y_test.detach().cpu().numpy(),
        bayesian_results["predictions"].detach().cpu().numpy(),
        bayesian_results["uncertainty"].detach().cpu().numpy(),
        title=f"{bayesian_name} Predictions",
        save_path=output_dir / "predictions.png"
    )
    
    # Plot training history
    plot_training_history(
        bayesian_results["training_results"]["losses"],
        title=f"{bayesian_name} Training History",
        save_path=output_dir / "training_history.png"
    )
    
    # Plot model comparison
    plot_model_comparison(
        comparison_metrics,
        title="Model Comparison",
        save_path=output_dir / "model_comparison.png"
    )
    
    # Save results
    results_path = output_dir / "results.yaml"
    OmegaConf.save(comparison_metrics, results_path)
    
    logging.info(f"Results saved to {results_path}")
    logging.info("Experiment completed successfully!")
    
    # Print summary
    print("\n" + "="*50)
    print("EXPERIMENT SUMMARY")
    print("="*50)
    
    for name, metrics in comparison_metrics.items():
        print(f"\n{name}:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")
    
    print(f"\nResults saved to: {output_dir}")
    print("Author: kryptologyst — GitHub: https://github.com/kryptologyst")


if __name__ == "__main__":
    main()

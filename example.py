#!/usr/bin/env python3
"""Simple example script demonstrating probabilistic programming."""

import torch
import numpy as np
import matplotlib.pyplot as plt

from src.utils import set_seed
from src.data import generate_linear_data
from src.models import BayesianLinearRegression
from src.metrics import regression_metrics, uncertainty_metrics
from src.viz import plot_predictions


def main():
    """Run a simple probabilistic programming example."""
    print("🎲 Probabilistic Programming Example")
    print("=" * 50)
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Generate synthetic data
    print("Generating synthetic linear data...")
    X_train, X_test, y_train, y_test = generate_linear_data(
        n_samples=100,
        noise_std=1.0,
        true_slope=2.0,
        true_intercept=1.0,
        random_seed=42
    )
    
    print(f"Data shape: Train {X_train.shape}, Test {X_test.shape}")
    
    # Initialize Bayesian Linear Regression model
    print("\nInitializing Bayesian Linear Regression model...")
    model = BayesianLinearRegression(
        input_dim=X_train.shape[1],
        prior_std=1.0,
        noise_prior_std=1.0
    )
    
    # Train the model
    print("Training model...")
    training_results = model.fit(
        X_train, y_train,
        num_epochs=1000,
        learning_rate=0.01,
        verbose=True
    )
    
    print(f"Final loss: {training_results['final_loss']:.4f}")
    
    # Make predictions with uncertainty
    print("\nMaking predictions...")
    y_pred, y_std = model.predict(X_test, return_uncertainty=True)
    
    # Calculate metrics
    print("\nCalculating metrics...")
    regression_met = regression_metrics(y_test, y_pred, y_std)
    uncertainty_met = uncertainty_metrics(y_test, y_pred, y_std)
    
    # Print results
    print("\n📊 Results:")
    print("-" * 30)
    print(f"RMSE: {regression_met['rmse']:.4f}")
    print(f"MAE: {regression_met['mae']:.4f}")
    print(f"R²: {regression_met['r2']:.4f}")
    print(f"95% Coverage: {uncertainty_met['coverage_95']:.3f}")
    print(f"Calibration Ratio: {uncertainty_met['calibration_ratio']:.3f}")
    
    # Create visualization
    print("\nCreating visualization...")
    plot_predictions(
        X_test.detach().cpu().numpy(),
        y_test.detach().cpu().numpy(),
        y_pred.detach().cpu().numpy(),
        y_std.detach().cpu().numpy(),
        title="Bayesian Linear Regression Predictions"
    )
    
    print("\n✅ Example completed successfully!")
    print("Author: kryptologyst — GitHub: https://github.com/kryptologyst")


if __name__ == "__main__":
    main()

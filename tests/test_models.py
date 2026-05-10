"""Tests for probabilistic programming implementation."""

import pytest
import torch
import numpy as np
from src.utils import set_seed, get_device
from src.data import generate_linear_data, generate_nonlinear_data
from src.models import BayesianLinearRegression, BayesianNeuralNetwork
from src.metrics import regression_metrics, uncertainty_metrics


class TestDataGeneration:
    """Test data generation functions."""
    
    def test_generate_linear_data(self):
        """Test linear data generation."""
        X_train, X_test, y_train, y_test = generate_linear_data(
            n_samples=100,
            noise_std=1.0,
            random_seed=42
        )
        
        assert X_train.shape[0] == 80  # 80% train split
        assert X_test.shape[0] == 20   # 20% test split
        assert X_train.shape[1] == 1   # Single feature
        assert isinstance(X_train, torch.Tensor)
        assert isinstance(y_train, torch.Tensor)
    
    def test_generate_nonlinear_data(self):
        """Test nonlinear data generation."""
        X_train, X_test, y_train, y_test = generate_nonlinear_data(
            n_samples=100,
            noise_std=0.1,
            random_seed=42
        )
        
        assert X_train.shape[0] == 80
        assert X_test.shape[0] == 20
        assert X_train.shape[1] == 1
        assert isinstance(X_train, torch.Tensor)
        assert isinstance(y_train, torch.Tensor)


class TestBayesianModels:
    """Test Bayesian model implementations."""
    
    def test_bayesian_linear_regression_init(self):
        """Test Bayesian Linear Regression initialization."""
        model = BayesianLinearRegression(
            input_dim=1,
            prior_std=1.0,
            noise_prior_std=1.0
        )
        
        assert model.input_dim == 1
        assert model.device == torch.device("cpu")
    
    def test_bayesian_neural_network_init(self):
        """Test Bayesian Neural Network initialization."""
        model = BayesianNeuralNetwork(
            input_dim=1,
            hidden_dim=64,
            output_dim=1,
            num_layers=2
        )
        
        assert model.input_dim == 1
        assert model.hidden_dim == 64
        assert model.num_layers == 2
    
    def test_bayesian_linear_regression_training(self):
        """Test Bayesian Linear Regression training."""
        set_seed(42)
        
        # Generate simple data
        X_train, X_test, y_train, y_test = generate_linear_data(
            n_samples=50,
            noise_std=0.5,
            random_seed=42
        )
        
        # Initialize model
        model = BayesianLinearRegression(input_dim=1)
        
        # Train model
        training_results = model.fit(
            X_train, y_train,
            num_epochs=100,
            learning_rate=0.01,
            verbose=False
        )
        
        assert "final_loss" in training_results
        assert training_results["final_loss"] > 0
    
    def test_bayesian_linear_regression_prediction(self):
        """Test Bayesian Linear Regression prediction."""
        set_seed(42)
        
        # Generate data
        X_train, X_test, y_train, y_test = generate_linear_data(
            n_samples=50,
            noise_std=0.5,
            random_seed=42
        )
        
        # Train model
        model = BayesianLinearRegression(input_dim=1)
        model.fit(X_train, y_train, num_epochs=100, verbose=False)
        
        # Make predictions
        y_pred, y_std = model.predict(X_test, return_uncertainty=True)
        
        assert y_pred.shape == y_test.shape
        assert y_std.shape == y_test.shape
        assert torch.all(y_std >= 0)  # Uncertainty should be non-negative


class TestMetrics:
    """Test evaluation metrics."""
    
    def test_regression_metrics(self):
        """Test regression metrics calculation."""
        y_true = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = torch.tensor([1.1, 1.9, 3.1, 3.9, 5.1])
        y_std = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1])
        
        metrics = regression_metrics(y_true, y_pred, y_std)
        
        assert "mse" in metrics
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics
        assert "coverage_95" in metrics
        assert metrics["mse"] > 0
        assert metrics["rmse"] > 0
    
    def test_uncertainty_metrics(self):
        """Test uncertainty metrics calculation."""
        y_true = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = torch.tensor([1.1, 1.9, 3.1, 3.9, 5.1])
        y_std = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1])
        
        metrics = uncertainty_metrics(y_true, y_pred, y_std)
        
        assert "coverage_95" in metrics
        assert "calibration_ratio" in metrics
        assert "expected_calibration_error" in metrics
        assert 0 <= metrics["coverage_95"] <= 1


class TestUtils:
    """Test utility functions."""
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        
        # Generate two random numbers
        x1 = torch.randn(1)
        set_seed(42)
        x2 = torch.randn(1)
        
        # Should be the same with same seed
        assert torch.allclose(x1, x2)
    
    def test_get_device(self):
        """Test device selection."""
        device = get_device("cpu")
        assert device == torch.device("cpu")
        
        device_auto = get_device("auto")
        assert isinstance(device_auto, torch.device)


if __name__ == "__main__":
    pytest.main([__file__])

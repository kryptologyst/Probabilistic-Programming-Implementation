"""Synthetic data generation functions."""

import logging
from typing import Tuple, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


def generate_linear_data(
    n_samples: int = 100,
    n_features: int = 1,
    noise_std: float = 1.0,
    true_slope: float = 2.0,
    true_intercept: float = 1.0,
    x_range: Tuple[float, float] = (0, 10),
    test_split: float = 0.2,
    random_seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate synthetic linear regression data.
    
    Args:
        n_samples: Number of samples to generate
        n_features: Number of input features
        noise_std: Standard deviation of noise
        true_slope: True slope parameter
        true_intercept: True intercept parameter
        x_range: Range for x values
        test_split: Fraction of data for testing
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test) tensors
    """
    np.random.seed(random_seed)
    
    # Generate x values
    X = np.random.uniform(x_range[0], x_range[1], (n_samples, n_features))
    
    # Generate y values with linear relationship and noise
    y = true_slope * X[:, 0] + true_intercept + np.random.normal(0, noise_std, n_samples)
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_split, random_state=random_seed
    )
    
    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    
    logging.info(f"Generated {len(X_train)} training and {len(X_test)} test samples")
    logging.info(f"True parameters: slope={true_slope}, intercept={true_intercept}, noise_std={noise_std}")
    
    return X_train, X_test, y_train, y_test


def generate_nonlinear_data(
    n_samples: int = 100,
    noise_std: float = 0.1,
    function_type: str = "sine",
    x_range: Tuple[float, float] = (0, 2 * np.pi),
    test_split: float = 0.2,
    random_seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate synthetic nonlinear regression data.
    
    Args:
        n_samples: Number of samples to generate
        noise_std: Standard deviation of noise
        function_type: Type of nonlinear function ("sine", "quadratic", "exponential")
        x_range: Range for x values
        test_split: Fraction of data for testing
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test) tensors
    """
    np.random.seed(random_seed)
    
    # Generate x values
    X = np.random.uniform(x_range[0], x_range[1], (n_samples, 1))
    
    # Generate y values based on function type
    if function_type == "sine":
        y = np.sin(X[:, 0]) + np.random.normal(0, noise_std, n_samples)
    elif function_type == "quadratic":
        y = X[:, 0] ** 2 + np.random.normal(0, noise_std, n_samples)
    elif function_type == "exponential":
        y = np.exp(X[:, 0]) + np.random.normal(0, noise_std, n_samples)
    else:
        raise ValueError(f"Unknown function type: {function_type}")
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_split, random_state=random_seed
    )
    
    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    
    logging.info(f"Generated {len(X_train)} training and {len(X_test)} test samples")
    logging.info(f"Function type: {function_type}, noise_std: {noise_std}")
    
    return X_train, X_test, y_train, y_test


def create_data_loader(
    X: torch.Tensor,
    y: torch.Tensor,
    batch_size: int = 32,
    shuffle: bool = True
) -> DataLoader:
    """Create a PyTorch DataLoader.
    
    Args:
        X: Input features
        y: Target values
        batch_size: Batch size
        shuffle: Whether to shuffle data
        
    Returns:
        DataLoader object
    """
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

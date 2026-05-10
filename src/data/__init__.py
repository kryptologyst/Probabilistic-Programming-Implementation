"""Data generation and loading utilities."""

from .synthetic import generate_linear_data, generate_nonlinear_data, create_data_loader

__all__ = ["generate_linear_data", "generate_nonlinear_data", "create_data_loader"]
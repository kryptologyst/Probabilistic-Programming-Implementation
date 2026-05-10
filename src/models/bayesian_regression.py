"""Bayesian regression models using Pyro."""

import logging
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam


class BayesianLinearRegression(PyroModule):
    """Bayesian Linear Regression using Pyro.
    
    This model treats both the coefficients and noise as probabilistic quantities,
    allowing for uncertainty quantification in predictions.
    """
    
    def __init__(
        self,
        input_dim: int = 1,
        prior_std: float = 1.0,
        noise_prior_std: float = 1.0,
        device: Optional[torch.device] = None
    ):
        """Initialize Bayesian Linear Regression model.
        
        Args:
            input_dim: Number of input features
            prior_std: Standard deviation of weight priors
            noise_prior_std: Standard deviation of noise prior
            device: Device to run on
        """
        super().__init__()
        self.input_dim = input_dim
        self.device = device or torch.device("cpu")
        
        # Define probabilistic parameters
        self.weight = PyroSample(
            dist.Normal(0.0, prior_std).expand([input_dim]).to_event(1)
        )
        self.bias = PyroSample(dist.Normal(0.0, prior_std))
        self.noise = PyroSample(dist.HalfNormal(noise_prior_std))
        
        self.to(self.device)
        
    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass of the model.
        
        Args:
            x: Input features
            y: Target values (optional, for training)
            
        Returns:
            Predicted values
        """
        # Linear transformation
        mean = torch.matmul(x, self.weight) + self.bias
        
        # Sample observations if targets provided
        if y is not None:
            with pyro.plate("data", x.shape[0]):
                pyro.sample("obs", dist.Normal(mean, self.noise), obs=y)
        
        return mean
    
    def guide(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> None:
        """Variational guide for inference.
        
        Args:
            x: Input features
            y: Target values (optional)
        """
        # Variational parameters for weights
        weight_loc = pyro.param(
            "weight_loc", 
            torch.zeros(self.input_dim, device=self.device)
        )
        weight_scale = pyro.param(
            "weight_scale", 
            torch.ones(self.input_dim, device=self.device),
            constraint=torch.distributions.constraints.positive
        )
        
        # Variational parameters for bias
        bias_loc = pyro.param("bias_loc", torch.tensor(0.0, device=self.device))
        bias_scale = pyro.param(
            "bias_scale", 
            torch.tensor(1.0, device=self.device),
            constraint=torch.distributions.constraints.positive
        )
        
        # Variational parameters for noise
        noise_loc = pyro.param(
            "noise_loc", 
            torch.tensor(1.0, device=self.device),
            constraint=torch.distributions.constraints.positive
        )
        
        # Sample from variational distributions
        pyro.sample("weight", dist.Normal(weight_loc, weight_scale).to_event(1))
        pyro.sample("bias", dist.Normal(bias_loc, bias_scale))
        pyro.sample("noise", dist.HalfNormal(noise_loc))
    
    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        num_epochs: int = 1000,
        learning_rate: float = 0.01,
        verbose: bool = True
    ) -> Dict[str, float]:
        """Fit the Bayesian model using variational inference.
        
        Args:
            X: Training features
            y: Training targets
            num_epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            verbose: Whether to print training progress
            
        Returns:
            Dictionary with training metrics
        """
        # Move data to device
        X = X.to(self.device)
        y = y.to(self.device)
        
        # Setup optimizer and inference
        optimizer = Adam({"lr": learning_rate})
        svi = SVI(self.model, self.guide, optimizer, loss=Trace_ELBO())
        
        # Training loop
        losses = []
        for epoch in range(num_epochs):
            loss = svi.step(X, y)
            losses.append(loss)
            
            if verbose and epoch % 100 == 0:
                logging.info(f"Epoch {epoch}, Loss: {loss:.4f}")
        
        # Get learned parameters
        params = {
            "weight_mean": pyro.param("weight_loc").detach().cpu().numpy(),
            "weight_std": pyro.param("weight_scale").detach().cpu().numpy(),
            "bias_mean": pyro.param("bias_loc").item(),
            "bias_std": pyro.param("bias_scale").item(),
            "noise_mean": pyro.param("noise_loc").item(),
            "final_loss": losses[-1]
        }
        
        if verbose:
            logging.info(f"Final parameters: {params}")
        
        return params
    
    def model(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Model function for Pyro inference."""
        return self.forward(x, y)
    
    def predict(
        self, 
        X: torch.Tensor, 
        num_samples: int = 100,
        return_uncertainty: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Make predictions with uncertainty quantification.
        
        Args:
            X: Input features
            num_samples: Number of samples for prediction
            return_uncertainty: Whether to return uncertainty estimates
            
        Returns:
            Predictions and optionally uncertainty estimates
        """
        X = X.to(self.device)
        
        # Generate samples from posterior
        predictions = []
        for _ in range(num_samples):
            with torch.no_grad():
                pred = self.forward(X)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        mean_pred = predictions.mean(dim=0)
        
        if return_uncertainty:
            std_pred = predictions.std(dim=0)
            return mean_pred, std_pred
        else:
            return mean_pred


class BayesianNeuralNetwork(PyroModule):
    """Bayesian Neural Network using Pyro.
    
    A neural network where weights are treated as random variables,
    enabling uncertainty quantification through Bayesian inference.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 1,
        num_layers: int = 2,
        prior_std: float = 1.0,
        noise_prior_std: float = 1.0,
        activation: str = "relu",
        dropout: float = 0.1,
        device: Optional[torch.device] = None
    ):
        """Initialize Bayesian Neural Network.
        
        Args:
            input_dim: Number of input features
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
            num_layers: Number of hidden layers
            prior_std: Standard deviation of weight priors
            noise_prior_std: Standard deviation of noise prior
            activation: Activation function
            dropout: Dropout rate
            device: Device to run on
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.device = device or torch.device("cpu")
        
        # Define activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        self.dropout = nn.Dropout(dropout)
        
        # Define probabilistic layers
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(
            PyroModule[nn.Linear](input_dim, hidden_dim)
        )
        self.layers[-1].weight = PyroSample(
            dist.Normal(0.0, prior_std).expand([hidden_dim, input_dim]).to_event(2)
        )
        self.layers[-1].bias = PyroSample(
            dist.Normal(0.0, prior_std).expand([hidden_dim]).to_event(1)
        )
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(
                PyroModule[nn.Linear](hidden_dim, hidden_dim)
            )
            self.layers[-1].weight = PyroSample(
                dist.Normal(0.0, prior_std).expand([hidden_dim, hidden_dim]).to_event(2)
            )
            self.layers[-1].bias = PyroSample(
                dist.Normal(0.0, prior_std).expand([hidden_dim]).to_event(1)
            )
        
        # Output layer
        self.layers.append(
            PyroModule[nn.Linear](hidden_dim, output_dim)
        )
        self.layers[-1].weight = PyroSample(
            dist.Normal(0.0, prior_std).expand([output_dim, hidden_dim]).to_event(2)
        )
        self.layers[-1].bias = PyroSample(
            dist.Normal(0.0, prior_std).expand([output_dim]).to_event(1)
        )
        
        # Noise parameter
        self.noise = PyroSample(dist.HalfNormal(noise_prior_std))
        
        self.to(self.device)
    
    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input features
            y: Target values (optional, for training)
            
        Returns:
            Network output
        """
        # Forward through layers
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.activation(x)
            x = self.dropout(x)
        
        # Output layer
        mean = self.layers[-1](x)
        
        # Sample observations if targets provided
        if y is not None:
            with pyro.plate("data", x.shape[0]):
                pyro.sample("obs", dist.Normal(mean.squeeze(), self.noise), obs=y)
        
        return mean.squeeze()
    
    def guide(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> None:
        """Variational guide for inference."""
        # Guide for each layer
        for i, layer in enumerate(self.layers):
            # Weight parameters
            weight_loc = pyro.param(
                f"weight_loc_{i}",
                torch.zeros_like(layer.weight),
                event_dim=2
            )
            weight_scale = pyro.param(
                f"weight_scale_{i}",
                torch.ones_like(layer.weight),
                constraint=torch.distributions.constraints.positive,
                event_dim=2
            )
            
            # Bias parameters
            bias_loc = pyro.param(
                f"bias_loc_{i}",
                torch.zeros_like(layer.bias),
                event_dim=1
            )
            bias_scale = pyro.param(
                f"bias_scale_{i}",
                torch.ones_like(layer.bias),
                constraint=torch.distributions.constraints.positive,
                event_dim=1
            )
            
            # Sample from variational distributions
            pyro.sample(f"weight_{i}", dist.Normal(weight_loc, weight_scale).to_event(2))
            pyro.sample(f"bias_{i}", dist.Normal(bias_loc, bias_scale).to_event(1))
        
        # Noise parameter
        noise_loc = pyro.param(
            "noise_loc",
            torch.tensor(1.0, device=self.device),
            constraint=torch.distributions.constraints.positive
        )
        pyro.sample("noise", dist.HalfNormal(noise_loc))
    
    def model(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Model function for Pyro inference."""
        return self.forward(x, y)
    
    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        num_epochs: int = 1000,
        learning_rate: float = 0.01,
        verbose: bool = True
    ) -> Dict[str, float]:
        """Fit the Bayesian neural network."""
        X = X.to(self.device)
        y = y.to(self.device)
        
        optimizer = Adam({"lr": learning_rate})
        svi = SVI(self.model, self.guide, optimizer, loss=Trace_ELBO())
        
        losses = []
        for epoch in range(num_epochs):
            loss = svi.step(X, y)
            losses.append(loss)
            
            if verbose and epoch % 100 == 0:
                logging.info(f"Epoch {epoch}, Loss: {loss:.4f}")
        
        return {"final_loss": losses[-1], "losses": losses}
    
    def predict(
        self, 
        X: torch.Tensor, 
        num_samples: int = 100,
        return_uncertainty: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Make predictions with uncertainty quantification."""
        X = X.to(self.device)
        
        predictions = []
        for _ in range(num_samples):
            with torch.no_grad():
                pred = self.forward(X)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        mean_pred = predictions.mean(dim=0)
        
        if return_uncertainty:
            std_pred = predictions.std(dim=0)
            return mean_pred, std_pred
        else:
            return mean_pred

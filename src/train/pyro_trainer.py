"""Training utilities for Pyro models."""

import logging
from typing import Dict, Any, Optional, Callable

import torch
import pyro
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam, ClippedAdam


class PyroTrainer:
    """Trainer for Pyro-based probabilistic models."""
    
    def __init__(
        self,
        model: Any,
        guide: Callable,
        learning_rate: float = 0.01,
        optimizer: str = "adam",
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None
    ):
        """Initialize Pyro trainer.
        
        Args:
            model: Pyro model function
            guide: Pyro guide function
            learning_rate: Learning rate for optimizer
            optimizer: Optimizer type ("adam", "clipped_adam")
            scheduler: Learning rate scheduler (optional)
            device: Device to run on
        """
        self.model = model
        self.guide = guide
        self.device = device or torch.device("cpu")
        
        # Setup optimizer
        if optimizer == "adam":
            self.optimizer = Adam({"lr": learning_rate})
        elif optimizer == "clipped_adam":
            self.optimizer = ClippedAdam({"lr": learning_rate})
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")
        
        self.scheduler = scheduler
        
        # Setup SVI
        self.svi = SVI(model, guide, self.optimizer, loss=Trace_ELBO())
        
        self.training_history = []
    
    def train_epoch(
        self,
        X: torch.Tensor,
        y: torch.Tensor
    ) -> float:
        """Train for one epoch.
        
        Args:
            X: Input features
            y: Target values
            
        Returns:
            Loss value
        """
        loss = self.svi.step(X, y)
        return loss
    
    def train(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        num_epochs: int = 1000,
        verbose: bool = True,
        log_interval: int = 100
    ) -> Dict[str, Any]:
        """Train the model.
        
        Args:
            X: Training features
            y: Training targets
            num_epochs: Number of training epochs
            verbose: Whether to print progress
            log_interval: Interval for logging
            
        Returns:
            Training history and final metrics
        """
        # Move data to device
        X = X.to(self.device)
        y = y.to(self.device)
        
        # Training loop
        losses = []
        for epoch in range(num_epochs):
            loss = self.train_epoch(X, y)
            losses.append(loss)
            
            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Logging
            if verbose and epoch % log_interval == 0:
                logging.info(f"Epoch {epoch}, Loss: {loss:.4f}")
        
        self.training_history = losses
        
        return {
            "final_loss": losses[-1],
            "best_loss": min(losses),
            "losses": losses
        }
    
    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """Get learned parameters from the guide.
        
        Returns:
            Dictionary of parameter names and values
        """
        return dict(pyro.get_param_store())


class EarlyStopping:
    """Early stopping utility for training."""
    
    def __init__(
        self,
        patience: int = 50,
        min_delta: float = 1e-6,
        monitor: str = "loss",
        mode: str = "min"
    ):
        """Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            monitor: Metric to monitor
            mode: "min" or "max" for improvement direction
        """
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        
        self.best_score = float('inf') if mode == "min" else float('-inf')
        self.counter = 0
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """Check if training should stop.
        
        Args:
            score: Current score to evaluate
            
        Returns:
            True if training should stop
        """
        if self.mode == "min":
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True
        
        return self.early_stop


class ModelCheckpoint:
    """Model checkpointing utility."""
    
    def __init__(
        self,
        filepath: str,
        monitor: str = "loss",
        mode: str = "min",
        save_best_only: bool = True
    ):
        """Initialize model checkpointing.
        
        Args:
            filepath: Path to save checkpoints
            monitor: Metric to monitor
            mode: "min" or "max" for improvement direction
            save_best_only: Whether to save only the best model
        """
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        
        self.best_score = float('inf') if mode == "min" else float('-inf')
    
    def __call__(
        self,
        model: Any,
        optimizer: Any,
        epoch: int,
        score: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Save checkpoint if conditions are met.
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            epoch: Current epoch
            score: Current score
            metadata: Additional metadata
        """
        should_save = False
        
        if self.save_best_only:
            if self.mode == "min" and score < self.best_score:
                should_save = True
                self.best_score = score
            elif self.mode == "max" and score > self.best_score:
                should_save = True
                self.best_score = score
        else:
            should_save = True
        
        if should_save:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict() if hasattr(model, 'state_dict') else None,
                "optimizer_state_dict": optimizer.state_dict() if hasattr(optimizer, 'state_dict') else None,
                "score": score,
                "metadata": metadata or {}
            }
            torch.save(checkpoint, self.filepath)
            logging.info(f"Checkpoint saved to {self.filepath}")


def train_with_callbacks(
    trainer: PyroTrainer,
    X: torch.Tensor,
    y: torch.Tensor,
    num_epochs: int = 1000,
    callbacks: Optional[list] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """Train model with callbacks.
    
    Args:
        trainer: Pyro trainer instance
        X: Training features
        y: Training targets
        num_epochs: Number of training epochs
        callbacks: List of callback functions
        verbose: Whether to print progress
        
    Returns:
        Training results
    """
    callbacks = callbacks or []
    
    # Move data to device
    X = X.to(trainer.device)
    y = y.to(trainer.device)
    
    losses = []
    for epoch in range(num_epochs):
        loss = trainer.train_epoch(X, y)
        losses.append(loss)
        
        # Call callbacks
        for callback in callbacks:
            if hasattr(callback, '__call__'):
                callback(epoch, loss, trainer)
        
        # Logging
        if verbose and epoch % 100 == 0:
            logging.info(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    return {
        "final_loss": losses[-1],
        "best_loss": min(losses),
        "losses": losses,
        "parameters": trainer.get_parameters()
    }

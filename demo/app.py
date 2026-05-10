"""Interactive Streamlit demo for probabilistic programming."""

import streamlit as st
import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.utils import set_seed, get_device
from src.data import generate_linear_data, generate_nonlinear_data
from src.models import BayesianLinearRegression, BayesianNeuralNetwork
from src.metrics import regression_metrics, uncertainty_metrics
from src.viz import create_interactive_plot


# Page configuration
st.set_page_config(
    page_title="Probabilistic Programming Demo",
    page_icon="🎲",
    layout="wide"
)

# Title and description
st.title("🎲 Probabilistic Programming Implementation")
st.markdown("""
**Author:** kryptologyst — [GitHub](https://github.com/kryptologyst)

This interactive demo showcases Bayesian regression models with uncertainty quantification using Pyro.
""")

# Disclaimer
st.warning("""
⚠️ **DISCLAIMER:** This is a research/educational demonstration. 
Not for production decisions or control systems. Results should not be used for critical decision-making.
""")

# Sidebar for configuration
st.sidebar.header("Configuration")

# Data generation parameters
st.sidebar.subheader("Data Generation")
data_type = st.sidebar.selectbox(
    "Data Type",
    ["linear", "nonlinear"],
    help="Type of synthetic data to generate"
)

n_samples = st.sidebar.slider(
    "Number of Samples",
    min_value=50,
    max_value=500,
    value=100,
    help="Number of data points to generate"
)

noise_std = st.sidebar.slider(
    "Noise Standard Deviation",
    min_value=0.1,
    max_value=2.0,
    value=1.0,
    step=0.1,
    help="Amount of noise in the data"
)

if data_type == "linear":
    true_slope = st.sidebar.slider(
        "True Slope",
        min_value=-5.0,
        max_value=5.0,
        value=2.0,
        step=0.1,
        help="True slope parameter"
    )
    true_intercept = st.sidebar.slider(
        "True Intercept",
        min_value=-5.0,
        max_value=5.0,
        value=1.0,
        step=0.1,
        help="True intercept parameter"
    )

# Model parameters
st.sidebar.subheader("Model Parameters")
model_type = st.sidebar.selectbox(
    "Model Type",
    ["Bayesian Linear Regression", "Bayesian Neural Network"],
    help="Type of Bayesian model to use"
)

prior_std = st.sidebar.slider(
    "Prior Standard Deviation",
    min_value=0.1,
    max_value=2.0,
    value=1.0,
    step=0.1,
    help="Standard deviation of weight priors"
)

noise_prior_std = st.sidebar.slider(
    "Noise Prior Standard Deviation",
    min_value=0.1,
    max_value=2.0,
    value=1.0,
    step=0.1,
    help="Standard deviation of noise prior"
)

if model_type == "Bayesian Neural Network":
    hidden_dim = st.sidebar.slider(
        "Hidden Dimension",
        min_value=16,
        max_value=128,
        value=64,
        step=16,
        help="Size of hidden layers"
    )
    num_layers = st.sidebar.slider(
        "Number of Layers",
        min_value=1,
        max_value=4,
        value=2,
        help="Number of hidden layers"
    )
    activation = st.sidebar.selectbox(
        "Activation Function",
        ["relu", "tanh", "sigmoid"],
        help="Activation function for hidden layers"
    )

# Training parameters
st.sidebar.subheader("Training Parameters")
num_epochs = st.sidebar.slider(
    "Number of Epochs",
    min_value=100,
    max_value=2000,
    value=1000,
    step=100,
    help="Number of training epochs"
)

learning_rate = st.sidebar.slider(
    "Learning Rate",
    min_value=0.001,
    max_value=0.1,
    value=0.01,
    step=0.001,
    format="%.3f",
    help="Learning rate for optimization"
)

# Set random seed
seed = st.sidebar.number_input(
    "Random Seed",
    min_value=0,
    max_value=1000,
    value=42,
    help="Random seed for reproducibility"
)

# Main content
if st.button("🚀 Run Experiment", type="primary"):
    
    # Set seed
    set_seed(seed)
    
    # Generate data
    with st.spinner("Generating data..."):
        if data_type == "linear":
            X_train, X_test, y_train, y_test = generate_linear_data(
                n_samples=n_samples,
                noise_std=noise_std,
                true_slope=true_slope,
                true_intercept=true_intercept,
                random_seed=seed
            )
        else:
            X_train, X_test, y_train, y_test = generate_nonlinear_data(
                n_samples=n_samples,
                noise_std=noise_std,
                function_type="sine",
                random_seed=seed
            )
    
    st.success(f"Generated {len(X_train)} training and {len(X_test)} test samples")
    
    # Train model
    with st.spinner("Training model..."):
        device = get_device("auto")
        
        if model_type == "Bayesian Linear Regression":
            model = BayesianLinearRegression(
                input_dim=X_train.shape[1],
                prior_std=prior_std,
                noise_prior_std=noise_prior_std,
                device=device
            )
        else:
            model = BayesianNeuralNetwork(
                input_dim=X_train.shape[1],
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                prior_std=prior_std,
                noise_prior_std=noise_prior_std,
                activation=activation,
                device=device
            )
        
        # Train model
        training_results = model.fit(
            X_train, y_train,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            verbose=False
        )
    
    st.success("Model training completed!")
    
    # Make predictions
    with st.spinner("Making predictions..."):
        y_pred, y_std = model.predict(X_test, return_uncertainty=True)
    
    # Calculate metrics
    regression_met = regression_metrics(y_test, y_pred, y_std)
    uncertainty_met = uncertainty_metrics(y_test, y_pred, y_std)
    
    # Display results
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Regression Metrics")
        st.metric("RMSE", f"{regression_met['rmse']:.4f}")
        st.metric("MAE", f"{regression_met['mae']:.4f}")
        st.metric("R²", f"{regression_met['r2']:.4f}")
        st.metric("MAPE", f"{regression_met['mape']:.2f}%")
    
    with col2:
        st.subheader("🎯 Uncertainty Metrics")
        st.metric("95% Coverage", f"{uncertainty_met['coverage_95']:.3f}")
        st.metric("Calibration Ratio", f"{uncertainty_met['calibration_ratio']:.3f}")
        st.metric("ECE", f"{uncertainty_met['expected_calibration_error']:.4f}")
        st.metric("Mean Uncertainty", f"{uncertainty_met['mean_prediction_std']:.4f}")
    
    # Create interactive plot
    st.subheader("📈 Interactive Predictions")
    
    fig = create_interactive_plot(
        X_test.detach().cpu().numpy(),
        y_test.detach().cpu().numpy(),
        y_pred.detach().cpu().numpy(),
        y_std.detach().cpu().numpy(),
        title=f"{model_type} Predictions"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Training history
    st.subheader("📉 Training History")
    
    fig_history = go.Figure()
    fig_history.add_trace(go.Scatter(
        y=training_results["losses"],
        mode='lines',
        name='Loss',
        line=dict(color='blue', width=2)
    ))
    
    fig_history.update_layout(
        title="Training Loss",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_history, use_container_width=True)
    
    # Residual analysis
    st.subheader("🔍 Residual Analysis")
    
    residuals = y_test.detach().cpu().numpy() - y_pred.detach().cpu().numpy()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Residuals vs predictions
        fig_residuals = go.Figure()
        fig_residuals.add_trace(go.Scatter(
            x=y_pred.detach().cpu().numpy(),
            y=residuals,
            mode='markers',
            name='Residuals',
            marker=dict(color='blue', size=6, opacity=0.6)
        ))
        fig_residuals.add_hline(y=0, line_dash="dash", line_color="red")
        fig_residuals.update_layout(
            title="Residuals vs Predictions",
            xaxis_title="Predictions",
            yaxis_title="Residuals"
        )
        st.plotly_chart(fig_residuals, use_container_width=True)
    
    with col2:
        # Histogram of residuals
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=residuals,
            nbinsx=20,
            name='Residuals',
            marker=dict(color='blue', opacity=0.7)
        ))
        fig_hist.update_layout(
            title="Distribution of Residuals",
            xaxis_title="Residuals",
            yaxis_title="Frequency"
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    # Uncertainty calibration
    st.subheader("🎯 Uncertainty Calibration")
    
    # Coverage vs confidence level
    confidence_levels = np.linspace(0.1, 0.99, 20)
    coverages = []
    
    for conf_level in confidence_levels:
        z_score = {0.1: 0.13, 0.2: 0.25, 0.3: 0.39, 0.4: 0.52, 0.5: 0.67,
                  0.6: 0.84, 0.7: 1.04, 0.8: 1.28, 0.9: 1.64, 0.95: 1.96, 0.99: 2.58}
        
        z = np.interp(conf_level, list(z_score.keys()), list(z_score.values()))
        
        lower_bound = y_pred.detach().cpu().numpy() - z * y_std.detach().cpu().numpy()
        upper_bound = y_pred.detach().cpu().numpy() + z * y_std.detach().cpu().numpy()
        coverage = np.mean((y_test.detach().cpu().numpy() >= lower_bound) & 
                          (y_test.detach().cpu().numpy() <= upper_bound))
        coverages.append(coverage)
    
    fig_calibration = go.Figure()
    fig_calibration.add_trace(go.Scatter(
        x=confidence_levels,
        y=coverages,
        mode='lines+markers',
        name='Actual Coverage',
        line=dict(color='blue', width=2)
    ))
    fig_calibration.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Perfect Calibration',
        line=dict(color='red', dash='dash')
    ))
    fig_calibration.update_layout(
        title="Coverage vs Confidence Level",
        xaxis_title="Confidence Level",
        yaxis_title="Coverage",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1])
    )
    
    st.plotly_chart(fig_calibration, use_container_width=True)
    
    # Model information
    st.subheader("ℹ️ Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Model Type:** {model_type}")
        st.write(f"**Training Samples:** {len(X_train)}")
        st.write(f"**Test Samples:** {len(X_test)}")
        st.write(f"**Training Epochs:** {num_epochs}")
    
    with col2:
        st.write(f"**Final Loss:** {training_results['final_loss']:.4f}")
        st.write(f"**Best Loss:** {training_results['best_loss']:.4f}")
        st.write(f"**Device:** {device}")
        st.write(f"**Random Seed:** {seed}")

# Footer
st.markdown("---")
st.markdown("""
**Probabilistic Programming Implementation**  
Author: kryptologyst — [GitHub](https://github.com/kryptologyst)

Built with Pyro, PyTorch, and Streamlit for educational and research purposes.
""")

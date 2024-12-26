import streamlit as st
import numpy as np
import joblib
import os

# Page Config
st.set_page_config(
    page_title="Sales Predictor",
    page_icon="🔮",
    layout="wide"
)

# Page Title
st.title("🔮 Sales Prediction")
st.write("Enter the advertising budgets below to predict sales:")

# Load Models

def load_model(model_name):
    try:
        model_path = os.path.join(os.path.dirname(__file__), f"../models/{model_name}.pkl")
        import warnings
        from sklearn.exceptions import InconsistentVersionWarning

        warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model '{model_name}': {e}")
        return None

models = {
    "Linear Regression": load_model("linear_regression"),
    "Ridge Regression": load_model("ridge_regression"),
    "Lasso Regression": load_model("lasso_regression"),
    "ElasticNet Regression": load_model("elasticnet_regression"),
    "Bayesian Ridge Regression": load_model("bayesian_ridge_regression"),
    "Decision Tree": load_model("decision_tree"),
    "Random Forest": load_model("random_forest"),
    "Gradient Boosting": load_model("gradient_boosting"),
    "Support Vector Regressor (SVR) (Scaled)": load_model("support_vector_regressor_svr"),
    "K-Nearest Neighbors": load_model("k-nearest_neighbors"),
    "AdaBoost": load_model("adaboost"),
    "Multi-Layer Perceptron (MLP)": load_model("multi-layer_perceptron_mlp")
}

# Remove any models that failed to load
models = {name: model for name, model in models.items() if model is not None}

# Model Performance Metrics
model_metrics = {
    "Linear Regression": {"MSE": 10.23, "R²": 0.89},
    "Ridge Regression": {"MSE": 9.87, "R²": 0.91},
    "Lasso Regression": {"MSE": 11.45, "R²": 0.87},
    "ElasticNet Regression": {"MSE": 10.91, "R²": 0.88},
    "Bayesian Ridge Regression": {"MSE": 10.65, "R²": 0.89},
    "Decision Tree": {"MSE": 12.34, "R²": 0.85},
    "Random Forest": {"MSE": 8.76, "R²": 0.92},
    "Gradient Boosting": {"MSE": 7.89, "R²": 0.959},
    "Support Vector Regressor (SVR) (Scaled)": {"MSE": 9.45, "R²": 0.90},
    "K-Nearest Neighbors": {"MSE": 10.12, "R²": 0.89},
    "AdaBoost": {"MSE": 9.32, "R²": 0.91},
    "Multi-Layer Perceptron (MLP)": {"MSE": 8.23, "R²": 0.94}
}

# Model Selection
model_choice = st.selectbox("Choose a Model for Prediction", list(models.keys()))

# Display Model Details
st.write(f"**Selected Model:** {model_choice}")
st.write(f"**MSE:** {model_metrics[model_choice]['MSE']}")
st.write(f"**R²:** {model_metrics[model_choice]['R²']}")

st.divider()

# Input Fields
tv_budget = st.number_input("TV Advertising Budget ($)", min_value=0.0, step=0.1)
radio_budget = st.number_input("Radio Advertising Budget ($)", min_value=0.0, step=0.1)
newspaper_budget = st.number_input("Newspaper Advertising Budget ($)", min_value=0.0, step=0.1)

# Prediction Button
if st.button("Predict Sales"):
    # Prepare input
    inputs = np.array([[tv_budget, radio_budget, newspaper_budget]])
    
    # Handle scaling for SVR
    if "SVR" in model_choice:
        scaler_path = os.path.join(os.path.dirname(__file__), "../models/scaler.pkl")
        try:
            scaler = joblib.load(scaler_path)
            inputs = scaler.transform(inputs)
        except Exception as e:
            st.error(f"Error loading scaler: {e}")
            st.stop()
    
    # Predict using selected model
    model = models.get(model_choice)
    if model:
        try:
            sales_prediction = model.predict(inputs)[0]
            st.success(f"Predicted Sales: ${sales_prediction:.2f}")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
    else:
        st.error("Selected model is unavailable or failed to load.")


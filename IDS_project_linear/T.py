import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Load dataset
data = pd.read_csv("data.csv")  # Ensure the dataset is in the same folder or provide the correct path

# Splitting dataset into features and target
X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1),
    "ElasticNet Regression": ElasticNet(alpha=0.1, l1_ratio=0.5),
    "Bayesian Ridge Regression": BayesianRidge(),
    "Decision Tree": DecisionTreeRegressor(max_depth=5),
    "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
    "Support Vector Regressor (SVR)": SVR(kernel='rbf', C=100, gamma=0.1),
    "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=5),
    "AdaBoost": AdaBoostRegressor(n_estimators=100),
    "Multi-Layer Perceptron (MLP)": MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
}

# Scale data for SVR
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Directory to save models
models_dir = "models"
os.makedirs(models_dir, exist_ok=True)

# Train models and save them
for name, model in models.items():
    print(f"Training {name}...")
    if name == "Support Vector Regressor (SVR)":
        model.fit(X_train_scaled, y_train)
    else:
        model.fit(X_train, y_train)
    
    # Save model
    model_filename = os.path.join(models_dir, f"{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.pkl")
    joblib.dump(model, model_filename)
    print(f"Saved model: {model_filename}")

# Save scaler for SVR
scaler_filename = os.path.join(models_dir, "scaler.pkl")
joblib.dump(scaler, scaler_filename)
print(f"Saved scaler: {scaler_filename}")

# Evaluate models
results = []
for name, model in models.items():
    print(f"Evaluating {name}...")
    if name == "Support Vector Regressor (SVR)":
        predictions = model.predict(X_test_scaled)
    else:
        predictions = model.predict(X_test)
    
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    results.append({"Model": name, "MSE": mse, "R²": r2})

# Save results to CSV
results_df = pd.DataFrame(results)
results_csv_path = os.path.join(models_dir, "retrained_model_metrics.csv")
results_df.to_csv(results_csv_path, index=False)
print(f"Saved results to {results_csv_path}")

import os

models_dir = os.path.join(os.path.dirname(__file__), "../models")
svr_scaled_path = os.path.join(models_dir, "svr_scaled.pkl")
print(f"SVR Model Exists: {os.path.exists(svr_scaled_path)}")

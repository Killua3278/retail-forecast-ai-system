# generate_model.py
# This script generates and saves a dummy model.pkl compatible with the Streamlit retail app

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import make_regression
import joblib
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic training data
X, y = make_regression(n_samples=100, n_features=514, noise=0.2)

# Ensure consistent data types
X = np.asarray(X, dtype=np.float32)
y = np.asarray(y, dtype=np.float32)

# Train the model
model = GradientBoostingRegressor().fit(X, y)

# Save the model to disk
joblib.dump(model, "model.pkl")

print("✅ model.pkl has been created with 514 input features.")

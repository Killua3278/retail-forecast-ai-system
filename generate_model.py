from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import make_regression
import joblib
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Generate synthetic data with 514 features (512 + 2)
X, y = make_regression(n_samples=100, n_features=514, noise=0.2)

# Train model
model = GradientBoostingRegressor().fit(X, y)

# Save model
joblib.dump(model, "model.pkl")

print("✅ model.pkl created with 514 input features.")

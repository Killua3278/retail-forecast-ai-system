from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import make_regression
import joblib
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic regression data with 514 features (512 image + 2 extra)
X, y = make_regression(n_samples=100, n_features=514, noise=0.2)

# Ensure the data is float32 for compatibility
X = np.asarray(X, dtype=np.float32)
y = np.asarray(y, dtype=np.float32)

# Train the Gradient Boosting model
model = GradientBoostingRegressor().fit(X, y)

# Save the trained model to 'model.pkl'
joblib.dump(model, "model.pkl")

print("✅ model.pkl created successfully with 514 input features.")


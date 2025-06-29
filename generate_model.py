from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import make_regression
import joblib
import numpy as np

np.random.seed(42)
X, y = make_regression(n_samples=100, n_features=514, noise=0.2)
model = GradientBoostingRegressor().fit(X, y)
joblib.dump(model, "model.pkl")
print("Model trained with 514 features saved as model.pkl")

import pickle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Scale data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 5. Evaluate
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))

# 6. Save model and scaler
import os

# Create model directory if it doesn't exist
os.makedirs(os.path.join(os.path.dirname(__file__), "..", "model"), exist_ok=True)

# Save model and scaler with absolute paths
model_path = os.path.join(os.path.dirname(__file__), "..", "model", "model.pkl")
scaler_path = os.path.join(os.path.dirname(__file__), "..", "model", "scaler.pkl")

pickle.dump(model, open(model_path, "wb"))
pickle.dump(scaler, open(scaler_path, "wb"))

print(f"Model saved to: {model_path}")
print(f"Scaler saved to: {scaler_path}")
print("Model & Scaler Saved Successfully!")

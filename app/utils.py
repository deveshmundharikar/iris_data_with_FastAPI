import os
import pickle
import numpy as np

# Get the current file's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Lazy-load scaler (only load when first needed)
_scaler = None


def get_scaler():
    """Lazy-load the scaler to avoid import-time errors when model doesn't exist."""
    global _scaler
    if _scaler is None:
        scaler_path = os.path.join(BASE_DIR, "..", "model", "scaler.pkl")
        with open(scaler_path, "rb") as f:
            _scaler = pickle.load(f)
    return _scaler


def preprocess_input(input_data):
    """
    Preprocess input features for prediction.

    Args:
        input_data: List or numpy array of shape (4,) containing [sepal_length, sepal_width, petal_length, petal_width]

    Returns:
        numpy.ndarray: Scaled features of shape (1, 4)
    """
    if not isinstance(input_data, (list, np.ndarray)):
        raise ValueError("Input must be a list or numpy array")
    if len(input_data) != 4:
        raise ValueError("Input must have exactly 4 features")

    # Convert to 2D array and scale
    input_array = np.array(input_data).reshape(1, -1)
    scaler = get_scaler()
    return scaler.transform(input_array)

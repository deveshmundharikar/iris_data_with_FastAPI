import os
import pickle
import numpy as np

# Get the current file's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load scaler safely using absolute path
scaler_path = os.path.join(BASE_DIR, "..", "model", "scaler.pkl")
with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

def preprocess_input(data):
    """
    Preprocess input features for prediction.
    
    Parameters:
    data : list or array-like
        [sepal_length, sepal_width, petal_length, petal_width]
    
    Returns:
    np.array : Scaled feature array ready for model prediction
    """
    arr = np.array(data).reshape(1, -1)
    scaled = scaler.transform(arr)
    return scaled

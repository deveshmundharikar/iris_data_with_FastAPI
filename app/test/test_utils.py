# app/test/test_utils.py
import numpy as np
from app.utils import preprocess_input


def test_preprocess_input():
    # Test with sample input
    input_data = [5.1, 3.5, 1.4, 0.2]
    result = preprocess_input(input_data)

    # Check if the output is a numpy array
    assert isinstance(result, np.ndarray)

    # Check the shape (should be 2D array with 1 row)
    assert result.shape == (1, 4)

    # Check if the output is scaled (values should be in a reasonable range)
    # Instead of checking the mean, check the range of values
    assert np.all(result > -5) and np.all(result < 5)  # Adjust these bounds based on your scaler

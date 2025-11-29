from fastapi import FastAPI
from pydantic import BaseModel
import os
import pickle

app = FastAPI()


# Pydantic model for request body
class Features(BaseModel):
    features: list[float]


# Get current file directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Lazy-load model (only load when first needed)
_model = None


def get_model():
    """Lazy-load the model to avoid import-time errors when model doesn't exist."""
    global _model
    if _model is None:
        model_path = os.path.join(BASE_DIR, "..", "model", "model.pkl")
        with open(model_path, "rb") as f:
            _model = pickle.load(f)
    return _model


@app.get("/")
def home():
    return {"message": "Welcome to the Iris Classifier API"}


@app.post("/predict")
def predict(data: Features):
    """
    data.features = [sepal_length, sepal_width, petal_length, petal_width]
    """
    from app.utils import preprocess_input

    processed = preprocess_input(data.features)
    model = get_model()
    prediction = int(model.predict(processed)[0])

    species = ["Setosa", "Versicolor", "Virginica"]

    return {"prediction": prediction, "class_name": species[prediction]}

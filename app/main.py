from fastapi import FastAPI
from pydantic import BaseModel
import os
import pickle
from app.utils import preprocess_input

app = FastAPI()

# Pydantic model for request body
class Features(BaseModel):
    features: list[float]

# Get current file directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load trained model safely
model_path = os.path.join(BASE_DIR, "..", "model", "model.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)


@app.get("/")
def home():
    return {"message": "Welcome to the Iris Classifier API"}

@app.post("/predict")
def predict(data: Features):
    """
    data.features = [sepal_length, sepal_width, petal_length, petal_width]
    """
    processed = preprocess_input(data.features)
    prediction = int(model.predict(processed)[0])
    
    species = ["Setosa", "Versicolor", "Virginica"]

    return {
        "prediction": prediction,
        "class_name": species[prediction]
    }

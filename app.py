from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd
import os
import dagshub

app = FastAPI(title="Wine Quality MLOps API")

# 1. Define what data the API expects to receive from the user
class WineFeatures(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float

model = None

# 2. When the server starts, load the model from DagsHub
@app.on_event("startup")
def load_model():
    global model
    # We will securely store your username in Hugging Face later!
    github_username = os.environ.get("GITHUB_USERNAME")
    
    print("Connecting to DagsHub...")
    dagshub.init(repo_owner=github_username, repo_name="mlops-wine-pipeline", mlflow=True)
    
    print("Finding the latest model...")
    # Automatically find the most recent training run
    runs = mlflow.search_runs()
    latest_run_id = runs.iloc[0]["run_id"]
    
    print(f"Downloading model from run: {latest_run_id}")
    model = mlflow.pyfunc.load_model(f"runs:/{latest_run_id}/model")
    print("Model loaded successfully!")

# 3. Create a basic home page
@app.get("/")
def home():
    return {"message": "Hello! The MLOps Wine Quality API is running and the model is loaded."}

# 4. Create the prediction endpoint
@app.post("/predict")
def predict(wine: WineFeatures):
    # Convert the user's input into a Pandas DataFrame
    data = pd.DataFrame([wine.dict()])
    
    # The original CSV had spaces in the column names, so we replace the underscores
    data.columns = [col.replace("_", " ") for col in data.columns]
    
    # Ask the model to predict the wine quality!
    prediction = model.predict(data)
    
    return {"predicted_quality_score": float(prediction[0])}

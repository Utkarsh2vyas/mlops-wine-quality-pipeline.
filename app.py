from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd
import os
import dagshub
from mlflow.tracking import MlflowClient

app = FastAPI(title="Wine Quality MLOps API")

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

@app.on_event("startup")
def load_model():
    global model
    github_username = os.environ.get("GITHUB_USERNAME")
    
    print("Connecting to DagsHub...")
    dagshub.init(repo_owner=github_username, repo_name="mlops-wine-pipeline", mlflow=True)
    
    print("Finding the latest model...")
    runs = mlflow.search_runs()
    latest_run_id = runs.iloc[0]["run_id"]
    
    # --- THE SMART DOWNLOADER ---
    client = MlflowClient()
    artifacts = client.list_artifacts(latest_run_id)
    
    # Automatically find the folder the model is hiding in
    model_folder = ""
    for art in artifacts:
        if art.is_dir:
            model_folder = art.path
            break
            
    print(f"Auto-discovered model folder: '{model_folder}'")
    print(f"Downloading model from run: {latest_run_id}")
    
    # Load the model using the auto-discovered path!
    model = mlflow.pyfunc.load_model(f"runs:/{latest_run_id}/{model_folder}")
    print("Model loaded successfully!")

@app.get("/")
def home():
    return {"message": "Hello! The MLOps Wine Quality API is running and the model is loaded."}

@app.post("/predict")
def predict(wine: WineFeatures):
    data = pd.DataFrame([wine.dict()])
    data.columns = [col.replace("_", " ") for col in data.columns]
    prediction = model.predict(data)
    return {"predicted_quality_score": float(prediction[0])}

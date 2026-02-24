from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd
import os
import dagshub
from mlflow.artifacts import download_artifacts

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
    
    print("Downloading all run artifacts to local server...")
    local_dir = download_artifacts(run_id=latest_run_id)
    
    model_path = None
    for root, dirs, files in os.walk(local_dir):
        if "MLmodel" in files:
            model_path = root
            break
            
    if model_path is None:
        raise Exception("Could not find the MLmodel file anywhere in the downloaded artifacts!")
        
    print(f"Found actual model at: {model_path}")
    
    model = mlflow.pyfunc.load_model(model_path)
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

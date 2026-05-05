from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI()

# Load model and scaler
model = pickle.load(open("model/diabetes_model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))

# Input schema
class DiabetesInput(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

@app.get("/")
def home():
    return {"message": "Diabetes Prediction API is running"}

@app.post("/predict")
def predict(data: DiabetesInput):
    
    input_data = np.array([[
        data.Pregnancies,
        data.Glucose,
        data.BloodPressure,
        data.SkinThickness,
        data.Insulin,
        data.BMI,
        data.DiabetesPedigreeFunction,
        data.Age
    ]])
    
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    
    return {
        "prediction": int(prediction[0]),
        "result": "Diabetes" if prediction[0] == 1 else "No Diabetes"
    }
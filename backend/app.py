from fastapi import FastAPI,Query,HTTPException 
from scraper import get_salary_selenium
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
import pandas as pd

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change "*" to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],
)
class SalaryRequest(BaseModel):
    age: int
    gender: str
    education: str
    job_title: str
    experience: float

try:
    model = joblib.load("data_processing/gb_model.pkl")
    feature_scaler = joblib.load("data_processing/feature_scaler.pkl")  
    salary_scaler = joblib.load("data_processing/salary_scaler.pkl")  
    X_columns = joblib.load("data_processing/X_columns.pkl")
    if isinstance(X_columns, pd.DataFrame):
        X_columns = X_columns.columns.tolist()  # Convert to list
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None
    feature_scaler = None   
    salary_scaler = None
    X_columns = None

@app.get("/")
def home():
    return {"message": "Salary Predictor API is running!"}

def predict_salary(age, gender, education, job_title, experience):
    print(f"Loaded X_columns shape: {len(X_columns)}")
    print(f"Expected input features: {model.n_features_in_}")

    x = np.zeros(len(X_columns)) 
    
    scaled_features = feature_scaler.transform([[age, experience]])[0]
    x[0:2] = scaled_features  
    
    for col_name in [f"Gender_{gender}", f"Education Level_{education}", f"Job Title_{job_title}"]:
        
        if col_name in X_columns:
            x[X_columns.index(col_name)] = 1

    # Convert input to DataFrame
    x_df = pd.DataFrame([x], columns=X_columns)

    # Predict scaled salary
    predicted_salary_scaled = model.predict(x_df)[0]

    # Convert back to original salary scale
    predicted_salary = salary_scaler.inverse_transform([[predicted_salary_scaled]])[0][0]

    return predicted_salary

@app.post("/predict/")
# def predict_salary(job_title: str=Query(), experience: int=Query()):
def predict_salary_api(data: SalaryRequest):
    
    predicted_salary = predict_salary(data.age, data.gender, data.education, data.job_title, data.experience)
    # Get real-time salary data
    # chrome_driver_path = r"C:\Users\kp121\Documents\vs code project\chromedriver\chromedriver-win64\chromedriver.exe"

    # service = Service(chrome_driver_path)

    # driver = webdriver.Chrome(service=service)
    real_salary_data = ""
    # get_salary_selenium(job_title,driver)

    return {
        "job_title": data.job_title,
        "predicted_salary": round(predicted_salary, 2),
        "real_time_salary": real_salary_data,
    }
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app)
    # uvicorn app:app --reload
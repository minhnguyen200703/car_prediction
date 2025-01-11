from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Initialize FastAPI
app = FastAPI()

# Define the input schema using Pydantic
class CarModel(BaseModel):
    mileage_v2: float
    manufacture_date: int
    seats: int
    gearbox: str
    fuel: str
    color: str
    brand: str
    type: str

# Load the trained model and scaler
model = joblib.load('baseline_mlp.joblib')
scaler = joblib.load('scaler.joblib')

# Load dummy columns for categorical features
with open("dummy_cols.txt", "r", encoding="utf-8") as f:
    dummy_cols = [line.strip() for line in f]

# Define categorical columns
categorical_cols = ['brand', 'fuel', 'color', 'gearbox', 'type']

@app.post("/")
async def scoring_endpoint(car: CarModel):
    # Convert input into DataFrame
    input_data = pd.DataFrame([car.dict().values()], columns=car.dict().keys())
    
    # Add the `car_age` feature and drop `manufacture_date`
    input_data['car_age'] = 2025 - input_data['manufacture_date']
    input_data = input_data.drop(columns=['manufacture_date'])

    # Standardize categorical feature values
    input_data['type'] = input_data['type'].str.strip().str.lower()
    input_data['gearbox'] = input_data['gearbox'].str.strip().str.upper()
    input_data['fuel'] = input_data['fuel'].str.strip().str.lower()

    # Perform one-hot encoding for categorical features
    input_data = pd.get_dummies(input_data, columns=categorical_cols, drop_first=True)

    # Ensure all dummy columns exist in the input data
    for col in dummy_cols:
        if col not in input_data.columns:
            input_data[col] = 0

    # Reorder columns to match training data
    input_data = input_data[dummy_cols]

    # Scale numeric columns
    numeric_cols_to_scale = ['mileage_v2', 'car_age']
    input_data[numeric_cols_to_scale] = scaler.transform(input_data[numeric_cols_to_scale])

    # Make predictions
    yhat = model.predict(input_data)

    # Return the prediction as JSON
    return {"prediction": float(yhat)}

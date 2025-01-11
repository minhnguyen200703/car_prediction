# Bring in lightweight dependencies
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd


app = FastAPI()

class CarModel(BaseModel):
    mileage_v2: float # -0.342622511
    manufacture_date: int # 1980
    seats: int # 4
    gearbox: str # MT
    fuel: str # petrol
    color: str # green
    brand: str # Jeep
    type: str # suv / cross over

# with open('baseline_mlp.pkl','rb') as f:
#     model = pickle.load(f)

model = joblib.load('baseline_mlp.joblib')
scaler = joblib.load('scaler.joblib')

with open("dummy_cols.txt", "r", encoding="utf-8") as f:
    dummy_cols = [line.strip() for line in f]

categorical_cols = ['brand', 'fuel', 'color', 'gearbox', 'type']

@app.post("/")
async def scoring_endpoint(car: CarModel):
    input_data = pd.DataFrame([car.dict().values()], columns=car.dict().keys())
    
    input_data['car_age'] = 2025 - input_data['manufacture_date']
    input_data = input_data.drop(columns=['manufacture_date'])  # Replace with car_age

    input_data['type'] = input_data['type'].str.strip().str.lower()
    input_data['gearbox'] = input_data['gearbox'].str.strip().str.upper()
    input_data['fuel'] = input_data['fuel'].str.strip().str.lower()

    input_data = pd.get_dummies(input_data, columns=categorical_cols, drop_first=True)

    for col in dummy_cols:
        if col not in input_data.columns:
            input_data[col] = 0
    input_data = input_data[dummy_cols]  

    numeric_cols_to_scale = ['mileage_v2', 'car_age']
    input_data[numeric_cols_to_scale] = scaler.transform(input_data[numeric_cols_to_scale])

    yhat = model.predict(input_data)
    return {"prediction": float(yhat)}
# Bring in lightweight dependencies
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
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

with open('baseline_mlp.pkl','rb') as f:
    model = pickle.load(f)

@app.post('/')
async def scoring_endpoint(car:CarModel):
    df = pd.DataFrame([car.dict().values()], columns=car.dict().keys())
    yhat = model.predict(df)
    return {"prediction":int(yhat)}
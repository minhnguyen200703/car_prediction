# Bring in lightweight dependencies
from fastapi import FastAPI
from pydantic import BaseModel

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


@app.post('/')
async def scoring_endpoint(car:CarModel):
    return car
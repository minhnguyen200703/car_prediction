from fastapi import FastAPI, HTTPException
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
try:
    model = joblib.load('./model/baseline_mlp.joblib')
    scaler = joblib.load('./model/scaler.joblib')
except Exception as e:
    raise RuntimeError(f"Error loading model or scaler: {str(e)}")

# Load dummy columns for categorical features
try:
    with open("./data/dummy_cols.txt", "r", encoding="utf-8") as f:
        dummy_cols = [line.strip() for line in f]
except FileNotFoundError:
    raise RuntimeError("Dummy columns file not found. Ensure 'dummy_cols.txt' exists in the correct path.")
except Exception as e:
    raise RuntimeError(f"Error reading dummy columns file: {str(e)}")

# Define categorical columns
categorical_cols = ['brand', 'model', 'origin', 'type', 'gearbox', 'fuel', 'color', 'condition']

@app.post("/")
async def scoring_endpoint(car: CarModel):
    try:
        # Convert input into DataFrame
        input_data = pd.DataFrame([car.dict().values()], columns=car.dict().keys())

        # Add the `car_age` feature and drop `manufacture_date`
        input_data['car_age'] = 2025 - input_data['manufacture_date']
        input_data = input_data.drop(columns=['manufacture_date'])

        # Standardize categorical feature values
        input_data['type'] = input_data['type'].str.strip().str.lower()
        input_data['gearbox'] = input_data['gearbox'].str.strip().str.upper()
        input_data['fuel'] = input_data['fuel'].str.strip().str.lower()

        # Initialize all dummy columns with 0 (False)
        for col in dummy_cols:
            input_data[col] = 0

        # Set the corresponding dummy columns to 1 (True) based on the input
        for col in categorical_cols:
            if col in input_data.columns:  # Ensure the column exists in input_data
                feature_value = input_data[col].iloc[0]
                dummy_col_name = f"{col}_{feature_value}"
                if dummy_col_name in dummy_cols:
                    input_data[dummy_col_name] = 1

        # Reorder columns to match the training data
        input_data = input_data.reindex(columns=dummy_cols, fill_value=0)

        # Scale numeric columns
        numeric_cols_to_scale = ['mileage_v2', 'car_age']
        input_data[numeric_cols_to_scale] = scaler.transform(input_data[numeric_cols_to_scale])

        # Make predictions
        yhat = model.predict(input_data)

        # Return the prediction as JSON
        return {"prediction": float(yhat)}
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing or invalid input feature: {str(e)}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Error processing input: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

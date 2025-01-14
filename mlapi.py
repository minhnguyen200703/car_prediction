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
model = joblib.load('./model/baseline_mlp.pkl')
scaler = joblib.load('./model/scaler.joblib')

# Load dummy columns for categorical features
with open("./data/dummy_cols.txt", "r", encoding="utf-8") as f:
    dummy_cols = [line.strip() for line in f]

# Define categorical columns
categorical_cols = ['brand', 'model', 'origin', 'type', 'gearbox', 'fuel', 'color', 'condition']

@app.post("/")
async def scoring_endpoint(car: CarModel):
    import logging

    # Configure logging
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("scoring_endpoint")

    # Convert input into DataFrame
    input_data = pd.DataFrame([car.dict().values()], columns=car.dict().keys())
    logger.debug(f"Initial Input Data:\n{input_data}")

    # Add the `car_age` feature and drop `manufacture_date`
    input_data['car_age'] = 2025 - input_data['manufacture_date']
    input_data = input_data.drop(columns=['manufacture_date'])
    logger.debug(f"Input Data After Adding car_age:\n{input_data}")

    # Standardize categorical feature values
    input_data['type'] = input_data['type'].astype(str).str.strip().str.lower()
    input_data['gearbox'] = input_data['gearbox'].astype(str).str.strip().str.upper()
    input_data['fuel'] = input_data['fuel'].astype(str).str.strip().str.lower()
    logger.debug(f"Input Data After Standardizing Categorical Values:\n{input_data}")

    # Initialize all dummy columns with 0
    for col in dummy_cols:
        input_data[col] = 0
    logger.debug(f"Input Data After Initializing Dummy Columns:\n{input_data}")

    # Set the corresponding dummy columns to 1 based on the input
    for col in categorical_cols:
        if col in input_data.columns:
            feature_value = input_data[col].iloc[0]
            dummy_col_name = f"{col}_{feature_value}"
            if dummy_col_name in dummy_cols:
                input_data[dummy_col_name] = 1
    logger.debug(f"Input Data After Setting Dummy Columns:\n{input_data}")

    # Reindex columns to match the training data
    input_data = input_data.reindex(columns=dummy_cols, fill_value=0)
    logger.debug(f"Input Data After Reindexing to Match Dummy Columns:\n{input_data}")

    # Validate numeric columns before scaling
    numeric_cols_to_scale = ['mileage_v2', 'car_age']
    for col in numeric_cols_to_scale:
        if col not in input_data.columns:
            input_data[col] = 0  # Default value for missing numeric columns

    # Log numeric columns before scaling
    logger.debug(f"Numeric Columns Before Scaling:\n{input_data[numeric_cols_to_scale]}")

    # Scale numeric columns
    try:
        input_data[numeric_cols_to_scale] = scaler.transform(input_data[numeric_cols_to_scale])
    except Exception as e:
        logger.error(f"Error During Scaling: {e}")
        raise e
    logger.debug(f"Input Data After Scaling Numeric Columns:\n{input_data}")

    # Make predictions
    try:
        yhat = model.predict(input_data)
    except Exception as e:
        logger.error(f"Error During Prediction: {e}")
        raise e

    # Log final input data and prediction
    logger.debug(f"Final Input Data for Prediction:\n{input_data}")
    logger.debug(f"Prediction: {yhat}")

    # Return the prediction as JSON
    return {"prediction": float(yhat)}

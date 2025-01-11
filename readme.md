# README

## Overview
The `Car_Predictions.ipynb` notebook includes all the steps for data preprocessing, data processing, and model training. The following models have been implemented:

- **Random Forest**
- **Gradient Boosting**
- **Multilayer Perceptron (MLP)**
- **Neural Network (TensorFlow)**

Additionally, a fine-tuned version of these models was planned, but could not be executed due to time constraints.

### Metrics Defined
The evaluation of the models was based on the following metrics:

1. **RMSE (Root Mean Square Error)**: Measures the average magnitude of errors between predicted and actual values. Lower values indicate better performance.
2. **MSE (Mean Squared Error)**: Measures the average squared difference between predicted and actual values. Lower values indicate better performance.
3. **R² (R-squared)**: Indicates the proportion of variance in the target variable explained by the model. Higher values (closer to 1) indicate better performance.

### Results
The baseline results of the models are summarized below:

| Model                  | Version   | RMSE      | MSE       | R²       |
|------------------------|-----------|-----------|-----------|----------|
| Random Forest          | Baseline  | 0.416608  | 0.173562  | 0.700296 |
| Gradient Boosting      | Baseline  | 0.413656  | 0.171112  | 0.705780 |
| Multi-Layer Perceptron | Baseline  | 0.353815  | 0.125185  | 0.784221 |

Based on these metrics, the **Multi-Layer Perceptron (MLP)** was selected as the best-performing model.

### Notes
- There was an issue with the results from the Neural Network (TensorFlow) model, which remains unresolved due to limited time (the results of some folds are negative). The details and intermediate results have been logged in the Jupyter notebook for future reference.

---

## RESTful API (Question 3)
A RESTful API has been created in the `mlapi.py` file, which uses **FastAPI**. The API is hosted at: [https://car-prediction-kutt.onrender.com](https://car-prediction-kutt.onrender.com).

### Functionality
- Accepts input as a JSON data payload.
- Returns predictions based on the trained Multi-Layer Perceptron model (saved as `baseline_mlp.pkl`).

To test the API, send a POST request to the endpoint with the required input format, and the API will return the predicted car price or other relevant outputs.

### Data analyze questions:
- I have put the answers in the `EDA.ipynb` notebook with the charts and plots to support my answers.

### Github link for references:
This is the github that I have for this project: [https://github.com/minhnguyen200703/car_prediction](https://github.com/minhnguyen200703/car_prediction).

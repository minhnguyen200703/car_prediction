from joblib import dump
import pickle

# Load the model using pickle
with open('baseline_mlp.pkl', 'rb') as f:
    model = pickle.load(f)

# Save the model as a .joblib file
dump(model, 'baseline_mlp.joblib')

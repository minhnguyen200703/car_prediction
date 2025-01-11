from joblib import dump
import pickle

# Load the model using pickle
with open('./data/baseline_mlp.pkl', 'rb') as f:
    model = pickle.load(f)

# Save the model as a .joblib file
dump(model, './data/baseline_mlp.joblib')

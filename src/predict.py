import pandas as pd
from src.model import load_model

def predict_single(input_dict: dict, model_path='models/model.joblib'):
    model = load_model(model_path)
    df = pd.DataFrame([input_dict])
    return model.predict(df)[0], model.predict_proba(df)[0][1]
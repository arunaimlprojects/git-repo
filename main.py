from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# Load model and preprocessing pipeline
model = joblib.load("artifacts/model/model.pkl")
preprocessor = joblib.load("artifacts/processed_data/preprocessor.pkl")

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    X_processed = preprocessor.transform(df)
    pred = model.predict(X_processed)[0]  # take first row
    return {
        "math_score": round(pred[0], 2),
        "reading_score": round(pred[1], 2),
        "writing_score": round(pred[2], 2)
    }

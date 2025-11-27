import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import joblib
import os
import json

class ModelEvaluation:
    def __init__(self, model_path, features_path, target_path, output_dir="artifacts/evaluation"):
        self.model_path = model_path
        self.features_path = features_path
        self.target_path = target_path
        self.output_dir = output_dir

    def run(self):
        print("\nStarting Model Evaluation...")

        # Load trained model
        model = joblib.load(self.model_path)

        # Load features and target
        X = pd.read_csv(self.features_path)
        y = pd.read_csv(self.target_path)

        # Train/Test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Predictions
        y_pred = model.predict(X_test)

        # Metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        metrics = {
            "MSE": mse,
            "MAE": mae,
            "R2": r2
        }

        # Save metrics
        os.makedirs(self.output_dir, exist_ok=True)
        with open(os.path.join(self.output_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)

        print(f"Model Evaluation Completed! Metrics saved at {self.output_dir}")
        print("Metrics:", metrics)

# Run Evaluation
if __name__ == "__main__":
    evaluator = ModelEvaluation(
        model_path="artifacts/model/model.pkl",
        features_path="artifacts/processed_data/X.csv",
        target_path="artifacts/processed_data/y.csv"
    )
    evaluator.run()

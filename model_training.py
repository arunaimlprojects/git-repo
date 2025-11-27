import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

class ModelTraining:
    def __init__(self, features_path, target_path, model_dir="artifacts/model"):
        self.features_path = features_path
        self.target_path = target_path
        self.model_dir = model_dir

    def run(self):
        print("\nStarting Model Training...")

        # ✅ Load features and target separately
        X = pd.read_csv(self.features_path)
        y = pd.read_csv(self.target_path)

        # ✅ Train/Test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # ✅ Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # ✅ Save model
        os.makedirs(self.model_dir, exist_ok=True)
        joblib.dump(model, os.path.join(self.model_dir, "model.pkl"))

        print(f"Model Training Completed! Model saved at {self.model_dir}")

# Run training
if __name__ == "__main__":
    trainer = ModelTraining(
        features_path="artifacts/processed_data/X.csv",
        target_path="artifacts/processed_data/y.csv"
    )
    trainer.run()

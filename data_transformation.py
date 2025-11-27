import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import os
import joblib

class DataTransformation:
    def __init__(self, raw_data_path, output_dir="artifacts/processed_data"):
        self.raw_data_path = raw_data_path
        self.output_dir = output_dir

    def run(self):
        print("\nStarting Data Transformation...")

        # 1️⃣ Load raw data
        df = pd.read_csv(self.raw_data_path)

        # 2️⃣ Separate target columns
        target_cols = ["math score", "reading score", "writing score"]
        y = df[target_cols]
        X = df.drop(columns=target_cols)

        # 3️⃣ Define categorical and numerical columns
        categorical_cols = ["gender", "race/ethnicity", "parental level of education",
                            "lunch", "test preparation course"]
        numerical_cols = [col for col in X.columns if col not in categorical_cols]

        # 4️⃣ Preprocessing pipeline
        preprocessor = ColumnTransformer(transformers=[
            ('cat', OneHotEncoder(), categorical_cols),
            ('num', StandardScaler(), numerical_cols)
        ])

        # Fit & transform
        X_processed = preprocessor.fit_transform(X)

        # ✅ Save preprocessor
        os.makedirs(self.output_dir, exist_ok=True)
        joblib.dump(preprocessor, os.path.join(self.output_dir, "preprocessor.pkl"))

        # 5️⃣ Save processed data
        # Features
        if hasattr(X_processed, "toarray"):  # OneHotEncoder returns sparse matrix
            X_processed_df = pd.DataFrame(X_processed.toarray())
        else:
            X_processed_df = pd.DataFrame(X_processed)

        X_processed_df.to_csv(os.path.join(self.output_dir, "X.csv"), index=False)
        # Target
        y.to_csv(os.path.join(self.output_dir, "y.csv"), index=False)

        print(f"Data Transformation Completed! Features, target and preprocessor saved at {self.output_dir}")


# Run Transformation
if __name__ == "__main__":
    transformer = DataTransformation("artifacts/raw_data.csv")
    transformer.run()

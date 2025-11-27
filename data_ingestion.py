import pandas as pd
import requests
import os

class DataIngestion:
    def __init__(self, source_path, output_path):
        self.source_path = source_path
        self.output_path = output_path

    def read_data(self):
        df = pd.read_csv(self.source_path)
        return df

    def save_raw(self, df):
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        df.to_csv(self.output_path, index=False)
        print(f"Raw data saved to: {self.output_path}")

    def run(self):
        print("Starting Data Ingestion...")
        df = self.read_data()
        self.save_raw(df)
        print("Data Ingestion Completed Successfully!")

# Run Pipeline
if __name__ == "__main__":
    ingestion = DataIngestion(
        source_path="C:\\Users\\Atthah\\Desktop\\my_own\\git-repo\\data\\StudentsPerformance.csv",
        output_path="artifacts/raw_data.csv"
    )
    ingestion.run()

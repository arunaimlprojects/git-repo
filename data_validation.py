import pandas as pd
import os

class DataValidation:
    def __init__(self, raw_data_path):
        self.raw_data_path = raw_data_path

    # Check if file exists
    def validate_exists(self):
        return os.path.exists(self.raw_data_path)

    # Check if columns match exactly
    def validate_schema(self, df, expected_columns):
        return list(df.columns) == expected_columns

    # Run validation
    def run(self):
        print("\nStarting Data Validation...")

        if not self.validate_exists():
            raise Exception("Raw data file does not exist!")

        df = pd.read_csv(self.raw_data_path)

        # Expected columns based on your dataset
        expected_columns = [
            "gender",
            "race/ethnicity",
            "parental level of education",
            "lunch",
            "test preparation course",
            "math score",
            "reading score",
            "writing score"
        ]

        if not self.validate_schema(df, expected_columns):
            raise Exception("Schema Validation Failed! Columns do not match expected ones.")

        print("Data Validation Successful!")
        return df

# Run validation
if __name__ == "__main__":
    validator = DataValidation("artifacts/raw_data.csv")
    df = validator.run()

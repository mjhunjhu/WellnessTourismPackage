import pandas as pd
import os
import sys

def register_data():
    data_path = "tourism_project/data/tourism.csv"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        sys.exit(1)
    
    # Load and validate schema
    try:
        df = pd.read_csv(data_path)
        print("Dataset loaded successfully.")
        print(f"Dataset columns: {list(df.columns)}")
        print(f"Total records registered: {len(df)}")
    except Exception as e:
        print(f"Validation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    register_data()

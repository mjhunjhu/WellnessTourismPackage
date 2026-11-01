import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import os
import sys

def prepare_data():
    # Use relative paths from project root
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "tourism.csv"
    output_dir = project_root / "data" / "splits"

    if not data_path.exists():
        print(f"Error: {data_path} not found.")
        sys.exit(1)

    # Load dataset
    df = pd.read_csv(data_path)
    print("Dataset loaded successfully. Shape:", df.shape)

    # Remove unnecessary columns
    columns_to_drop = ["Unnamed: 0", "CustomerID"]
    df = df.drop(columns=columns_to_drop, errors="ignore")

    # Define target variable
    TARGET = "ProdTaken"
    if TARGET not in df.columns:
        print(f"Error: Target column {TARGET} not found.")
        sys.exit(1)

    # Split features and target
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Recombine features and target
    train_df = X_train.copy()
    train_df[TARGET] = y_train

    test_df = X_test.copy()
    test_df[TARGET] = y_test

    # Save output splits
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_dir / "train.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    print("Data preparation completed. Splits saved successfully.")
    print("Train size:", train_df.shape, "| Test size:", test_df.shape)

if __name__ == "__main__":
    prepare_data()

from pathlib import Path
import json
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

# 1. Define project paths
project_root = Path(__file__).resolve().parent.parent
train_path = project_root / "data" / "splits" / "train.csv"
test_path = project_root / "data" / "splits" / "test.csv"
deployment_dir = project_root / "deployment"
deployment_dir.mkdir(parents=True, exist_ok=True)
model_path = deployment_dir / "best_model.joblib"

# 2. Load datasets
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
TARGET = "ProdTaken"

X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]
X_test = test_df.drop(columns=[TARGET])
y_test = test_df[TARGET]

# 3. Identify numerical and categorical columns
numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X_train.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

# 4. Preprocessing
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# 5. Define model pipeline
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(random_state=42, class_weight="balanced"))
])

param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 10, 20],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1, 2]
}

# 6. Configure MLflow and start training
mlflow.set_tracking_uri((project_root / "mlruns").as_uri())
mlflow.set_experiment("Tourism_Prediction")

with mlflow.start_run(run_name="Random_Forest_Tuning"):
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, scoring="f1", n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    
    # Log params & metrics
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_metric("best_cv_f1", grid_search.best_score_)
    
    y_pred = best_model.predict(X_test)
    mlflow.log_metrics({
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_precision": precision_score(y_test, y_pred, zero_division=0),
        "test_recall": recall_score(y_test, y_pred, zero_division=0),
        "test_f1": f1_score(y_test, y_pred, zero_division=0)
    })
    
    # Save local model and log to MLflow
    joblib.dump(best_model, model_path)
    mlflow.sklearn.log_model(best_model, "best_model")
    
    print("Model training completed and saved to GitHub staging area!")

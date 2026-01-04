import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from preprocessing import preprocess_data, split_and_scale
import joblib
import os

def train(data_path, model_type, max_depth=None, n_estimators=100, C=1.0):
    """
    Train a model and log to MLflow.
    """
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    print("Preprocessing data...")
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)
    
    # Start MLflow run
    with mlflow.start_run():
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("data_path", data_path)
        
        if model_type == "logistic_regression":
            mlflow.log_param("C", C)
            model = LogisticRegression(C=C, random_state=42)
        elif model_type == "random_forest":
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        else:
            raise ValueError("Invalid model_type")
            
        print(f"Training {model_type}...")
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        try:
            roc = roc_auc_score(y_test, y_prob)
        except:
            roc = 0.0
            
        print(f"Metrics: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, ROC={roc:.4f}")
        
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("roc_auc", roc)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Save scaler
        os.makedirs("models", exist_ok=True)
        joblib.dump(scaler, "models/scaler.joblib")
        mlflow.log_artifact("models/scaler.joblib")

        # Save model locally for easy access in API/Docker
        joblib.dump(model, "models/model.pkl")
        print("Model saved locally to models/model.pkl")
        
        print(f"Run complete. Model logged to MLflow.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/raw/heart_disease.csv")
    parser.add_argument("--model_type", default="logistic_regression", choices=["logistic_regression", "random_forest"])
    parser.add_argument("--max_depth", type=int, default=None)
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--C", type=float, default=1.0)
    
    args = parser.parse_args()
    
    train(args.data_path, args.model_type, args.max_depth, args.n_estimators, args.C)

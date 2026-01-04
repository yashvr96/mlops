import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from preprocessing import preprocess_data, split_and_scale
import joblib
import os
import numpy as np

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
        
        # Cross Validation (5-fold)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        print(f"Cross-Validation Accuracy Scores: {cv_scores}")
        print(f"Mean CV Accuracy: {cv_scores.mean():.4f}")
        mlflow.log_metric("cv_accuracy_mean", cv_scores.mean())
        mlflow.log_metric("cv_accuracy_std", cv_scores.std())

        # Fit on full training set
        model.fit(X_train, y_train)
        
        # Evaluate on Test Set
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        try:
            roc = roc_auc_score(y_test, y_prob)
        except:
            roc = 0.0
            
        print(f"Test Metrics: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, ROC={roc:.4f}")
        
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("roc_auc", roc)
        
        # --- PLOTS ---
        # 1. Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        fig, ax = plt.subplots(figsize=(6,6))
        disp.plot(ax=ax, cmap='Blues')
        plt.title(f"Confusion Matrix - {model_type}")
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        plt.close()
        
        # 2. ROC Curve
        if hasattr(model, "predict_proba"):
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            plt.figure(figsize=(8,6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {model_type}')
            plt.legend(loc="lower right")
            plt.savefig("roc_curve.png")
            mlflow.log_artifact("roc_curve.png")
            plt.close()

        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Save scaler
        os.makedirs("models", exist_ok=True)
        joblib.dump(scaler, "models/scaler.joblib")
        mlflow.log_artifact("models/scaler.joblib")

        # Save model locally for easy access in API/Docker
        joblib.dump(model, "models/model.pkl")
        print("Model saved locally to models/model.pkl")
        
        # Cleanup temp images
        if os.path.exists("confusion_matrix.png"):
            os.remove("confusion_matrix.png")
        if os.path.exists("roc_curve.png"):
            os.remove("roc_curve.png")
            
        print(f"Run complete. Model and Artifacts logged to MLflow.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/raw/heart_disease.csv")
    parser.add_argument("--model_type", default="logistic_regression", choices=["logistic_regression", "random_forest"])
    parser.add_argument("--max_depth", type=int, default=None)
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--C", type=float, default=1.0)
    
    args = parser.parse_args()
    
    train(args.data_path, args.model_type, args.max_depth, args.n_estimators, args.C)

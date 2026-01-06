# Heart Disease Prediction MLOps Pipeline

This repository contains an end-to-end MLOps project for predicting heart disease risk. It demonstrates a full machine learning lifecycle including data acquisition, experimentation, containerization, and deployment on Kubernetes.

📄 **[Read the Full Project Report](./REPORT.md)** for detailed architecture, modeling choices, and experiments.

## 🚀 Key Features
*   **Automated Data Pipeline**: Fetches and processes data from the UCI Repository.
*   **Experiment Tracking**: Uses **MLflow** to log parameters, metrics, and model artifacts.
*   **Containerization**: **Docker** image with a production-ready **FastAPI** application.
*   **Deployment**: **Kubernetes** manifests for high-availability deployment with LoadBalancer.
*   **Monitoring**: Integrated **Prometheus** metrics and **Grafana** dashboard support.
*   **CI/CD**: GitHub Actions pipeline for automated linting, testing, and artifact generation.

## 🛠️ Quick Start

### 1. Install Dependencies
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows
# source .venv/bin/activate    # Linux/Mac
pip install -r requirements.txt
```

### 2. Run Training
```bash
python src/data_loader.py
python src/train.py --model_type logistic_regression
```

### 3. Run API Locally (Docker)
```bash
docker build -t mlops-heart-disease:latest .
docker run -p 8000:8000 mlops-heart-disease:latest
```
Access Swagger UI at: [http://localhost:8000/docs](http://localhost:8000/docs)

### 4. Deploy to Kubernetes
```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

## 📂 Project Structure
*   `src/` - Source code for API, training, and data loading.
*   `tests/` - Unit tests for the application.
*   `k8s/` - Kubernetes deployment and monitoring manifests.
*   `notebooks/` - Jupyter notebooks for EDA.
*   `.github/workflows/` - CI/CD pipeline configuration.

---
*Created by Yash Verma for MLOps Assignment 1.*
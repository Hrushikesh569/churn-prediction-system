---
title: Churn Prediction Dashboard
emoji: 🚀
colorFrom: green
colorTo: red
sdk: docker
pinned: false
license: mit
short_description: Built a machine learning-powered dashboard to predict customer churn
---

# Churn Prediction System

This project is a complete Machine Learning pipeline and web application designed to predict customer churn. It uses an ensemble voting classifier to predict whether a customer will leave based on their demographics, account information, and service usage.

## 🏗️ Architecture

The system is composed of the following key components:

1. **FastAPI Backend (`app/app.py`)**: A high-performance web server that exposes a `/predict` endpoint. It handles incoming customer data, runs the data through the preprocessing pipeline, and uses pre-trained models to generate churn predictions.
2. **Machine Learning Pipeline (`src/`)**: 
   - **Data Processing (`src/data/`)**: Cleans raw data and handles missing values.
   - **Feature Engineering (`src/features/`)**: Builds features, caps outliers, and prepares the dataset.
   - **Model Training & Evaluation (`src/models/`)**: Code to train the models (Logistic Regression, Random Forest, XGBoost) and combine them into a Voting Classifier.
3. **Frontend (`app/static/`)**: A responsive web interface allowing users to input customer details and instantly view churn probability and risk levels.
4. **Configuration (`config/config.yaml`)**: Centralized configuration management for model hyperparameters, cross-validation settings, data URLs, and output paths.

## 🔄 Workflow

1. **Data Ingestion & Preprocessing**: Raw data is fetched (e.g., Telco Customer Churn dataset) and goes through rigorous cleaning (handling data types, imputing missing values, and outlier capping).
2. **Feature Engineering**: Transforming categorical and continuous variables, utilizing `StandardScaler` and `OneHotEncoder`.
3. **Model Training**: A pipeline trains Logistic Regression, Random Forest, and XGBoost models. Hyperparameters are tuned, and the best estimators are combined into a Voting Classifier.
4. **Model Serialization**: The trained models, preprocessor, and outlier cap values are saved in the `outputs/models/` directory as `.joblib` files.
5. **Inference / Deployment**: 
   - The FastAPI app loads the serialized models on startup.
   - Users submit data via the frontend.
   - The backend processes the input identically to the training phase and returns the churn probability, prediction, and risk level.

## 🚀 Running Locally

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the FastAPI server:
   ```bash
   uvicorn app.app:app --host 0.0.0.0 --port 8000
   ```
3. Open your browser and navigate to `http://localhost:8000`.

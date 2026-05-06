# 🩺 MLOps Project: End-to-End Diabetes Prediction System

##  Solution Overview

This project is an end-to-end Diabetes Prediction MLOps system built using Python, FastAPI, Scikit-learn, GitHub Actions, and Render.

The system takes patient health data as input through a FastAPI REST API and predicts whether the patient is likely to have diabetes using a trained Machine Learning model.

The project includes several MLOps practices such as:

Cloud deployment using Render
CI pipeline using GitHub Actions
Prediction monitoring and logging
Model versioning with timestamp-based saving
Automated retraining workflow using GitHub Actions
REST API serving using FastAPI

Whenever code is pushed to GitHub, the CI pipeline automatically validates the project and checks whether the backend works correctly.

The project also includes a manual retraining pipeline where GitHub Actions can automatically run the model training script in the cloud and generate new model versions without manually running the training code locally.


# 🎯 Quick Highlights

## ✅ Production-Ready ML System

Built a complete ML pipeline capable of:

* Training models
* Serving predictions through APIs
* Monitoring predictions
* Retraining automatically
* Deploying to the cloud

---

## ✅ Cloud-Native Deployment

The project is deployed using Render and accessible through a public API endpoint.

---

## ✅ CI/CD Automation

Implemented GitHub Actions workflows for:

* Continuous Integration (CI)
* Automated Retraining Pipeline

---

## ✅ Model Lifecycle Management

Implemented:

* Model versioning
* Logging & monitoring
* Prediction tracking
* Automated model generation

---

# 📚 Table of Contents

* [Live Deployment](#-live-deployment)
* [Project Overview](#-project-overview)
* [Tech Stack](#-tech-stack)
* [Project Architecture](#-project-architecture)
* [End-to-End Workflow](#️-end-to-end-workflow)
* [API Example](#-api-example)
* [How to Run Locally](#️-how-to-run-locally)
* [CI/CD Workflows](#-cicd-workflows)
* [Challenges & Solutions](#-challenges--solutions)
* [Future Improvements](#-future-improvements)
* [Learning Outcomes](#-learning-outcomes)
* [Final Project Status](#-final-project-status)
* [Author](#-author)

---

# 🌍 Live Deployment

## 🔗 Live API URL

```text
https://diabetes-mlops-7g8v.onrender.com
```

## 📖 Swagger Documentation

```text
https://diabetes-mlops-7g8v.onrender.com/docs
```

---

# 🧠 Project Overview

This project predicts whether a patient is likely to have diabetes based on medical attributes such as:

* Glucose
* BMI
* Blood Pressure
* Insulin
* Age
* Pregnancies

The system uses a RandomForestClassifier trained on the Pima Indians Diabetes Dataset.

The ML model is exposed through a FastAPI REST API and deployed on Render.

---

# 🛠️ Tech Stack

## Core ML & Data Processing

| Technology   | Usage                          |
| ------------ | ------------------------------ |
| Python       | Core programming language      |
| Pandas       | Data preprocessing             |
| NumPy        | Numerical operations           |
| Scikit-learn | ML model training & evaluation |

---

## Backend & Deployment

| Technology | Usage            |
| ---------- | ---------------- |
| FastAPI    | API framework    |
| Uvicorn    | ASGI server      |
| Render     | Cloud deployment |

---

## MLOps & Automation

| Technology     | Usage                      |
| -------------- | -------------------------- |
| GitHub Actions | CI/CD workflows            |
| GitHub         | Version control            |
| Logging        | Monitoring & observability |

---

# 📂 Project Architecture

```text
Diabetes/
│
├── backend/
│   └── main.py                    # FastAPI backend
│
├── data/
│   └── diabetes.csv              # Dataset
│
├── model/
│   ├── train_model.py            # Model training pipeline
│   ├── diabetes_model.pkl        # Current production model
│   ├── scaler.pkl                # Current scaler
│   ├── model_*.pkl               # Versioned models
│   └── scaler_*.pkl              # Versioned scalers
│
├── scripts/
│   └── retrain.py                # Retraining script
│
├── notebooks/
│   └── experimentation.ipynb     # Experiments & EDA
│
├── .github/
│   └── workflows/
│       ├── ci.yml                # CI pipeline
│       └── retrain.yml           # Retraining workflow
│
├── requirements.txt
├── Procfile
└── README.md
```

---

# ⚙️ End-to-End Workflow

## 1️⃣ Data Preprocessing

The dataset is:

* Loaded using Pandas
* Cleaned by handling invalid values
* Standardized using StandardScaler

---

## 2️⃣ Model Training

The project uses:

```text
RandomForestClassifier
```

Training steps:

* Train-test split
* Feature scaling
* Model fitting
* Threshold tuning
* Evaluation using confusion matrix & classification report

---

## 3️⃣ API Serving

FastAPI exposes the ML model using:

```text
POST /predict
```

The API accepts medical inputs and returns:

* Prediction label
* Diabetes / No Diabetes result

---

## 4️⃣ Monitoring & Logging

The project includes monitoring features such as:

* Prediction count tracking
* Input logging
* Prediction logging
* Server activity logs

This provides basic observability for the deployed ML system.

---

## 5️⃣ Model Versioning

Each retraining generates uniquely versioned model files:

```text
model_YYYYMMDD_HHMMSS.pkl
```

Benefits:

* Prevents overwriting
* Supports rollback
* Tracks model history

---

## 6️⃣ CI Pipeline

GitHub Actions automatically:

* Installs dependencies
* Validates backend code
* Checks application structure

on every push to GitHub.

---

## 7️⃣ Automated Retraining Pipeline

A manual GitHub Actions workflow enables:

* Automated model retraining
* Cloud-based training execution
* Automatic generation of versioned models

---

# 🧪 API Example

## Sample Input

```json
{
  "Pregnancies": 2,
  "Glucose": 120,
  "BloodPressure": 70,
  "SkinThickness": 20,
  "Insulin": 80,
  "BMI": 25.0,
  "DiabetesPedigreeFunction": 0.5,
  "Age": 30
}
```

## Sample Output

```json
{
  "prediction": 0,
  "result": "No Diabetes"
}
```

---

# ▶️ How to Run Locally

## 1️⃣ Clone Repository

```bash
git clone <your-repository-url>
```

---

## 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3️⃣ Run FastAPI Server

```bash
python -m uvicorn backend.main:app --reload
```

---

## 4️⃣ Open Swagger UI

```text
http://127.0.0.1:8000/docs
```

---

# 🔁 CI/CD Workflows

## Continuous Integration (CI)

The CI pipeline:

* Installs dependencies
* Validates backend syntax
* Checks project integrity

Triggered automatically on every push.

---

## Retraining Workflow

The retraining workflow:

* Runs training pipeline automatically
* Generates new model versions
* Simulates automated retraining behavior

Triggered manually through GitHub Actions.

---

# 🛠️ Challenges & Solutions

## 1️⃣ Dependency & Deployment Issues

### Problem

The initial `requirements.txt` contained unnecessary packages such as TensorFlow, Jupyter, and other heavy dependencies which caused deployment failures on Render.

### Solution

The dependency list was cleaned and reduced to only production-required packages:

* FastAPI
* Uvicorn
* NumPy
* Pandas
* Scikit-learn
* Pydantic

### Result

Successful deployment of the ML API on Render.

---

## 2️⃣ Model Overwriting During Retraining

### Problem

Every retraining process overwrote the previous model file:

```text
model/diabetes_model.pkl
```

This removed older model versions and made rollback impossible.

### Solution

Implemented timestamp-based model versioning:

```text
model_YYYYMMDD_HHMMSS.pkl
```

### Result

* Preserved model history
* Enabled rollback capability
* Improved reproducibility

---

## 3️⃣ Monitoring & Logging

### Problem

Initially there was no visibility into API usage or predictions.

### Solution

Implemented:

* Prediction logging
* Input logging
* Prediction count tracking
* Server activity logs

### Result

Added basic monitoring and observability to the ML system.

---

## 4️⃣ CI/CD Workflow Setup

### Problem

Understanding GitHub Actions workflows and automation pipelines was initially challenging.

### Solution

Implemented:

* CI pipeline using `ci.yml`
* Automated retraining workflow using `retrain.yml`

### Result

The project now supports:

* Automated validation
* Cloud-based retraining
* Workflow automation

---

## 5️⃣ Deployment & GitHub Workflow Debugging

### Problem

Several deployment and GitHub workflow issues occurred during setup:

* Uvicorn not recognized
* Render build failures
* Workflow execution issues
* Git/GitHub confusion during commits and pushes

### Solution

Systematically debugged:

* Python environment setup
* Dependency installation
* Git commands and workflow
* Render deployment configuration
* GitHub Actions configuration

### Result

Successfully deployed a fully working ML system with CI/CD and retraining pipelines.

---

# 📈 Future Improvements

* Frontend UI Integration
* MLflow Integration
* Docker Containerization
* Advanced Monitoring Dashboard
* Auto Deployment after Retraining
* Kubernetes Scaling

---

# 🎯 Learning Outcomes

This project helped in understanding:

* End-to-End ML Systems
* FastAPI Backend Development
* Cloud Deployment
* CI/CD Pipelines
* Monitoring & Logging
* Model Versioning
* Automated Retraining
* Basic MLOps Concepts

---

# 🏆 Final Project Status

This project successfully demonstrates:

✅ End-to-End ML Pipeline

✅ Cloud Deployment

✅ CI/CD Automation

✅ Monitoring & Logging

✅ Model Versioning

✅ Automated Retraining

✅ Basic MLOps Workflow

---

# 👨‍💻 Author

Shishir M S

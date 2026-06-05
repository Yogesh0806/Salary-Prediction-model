# 💰 Adult Income Classification System

An end-to-end Machine Learning web application that predicts whether an individual's annual income exceeds $50,000 based on demographic and employment-related attributes from the Adult Census Income dataset.

Built using Python, Scikit-Learn, XGBoost, and Streamlit.

---

## 🚀 Features

* Automated data preprocessing pipeline
* Missing value handling and data cleaning
* Exploratory Data Analysis (EDA)
* Training and comparison of 7 machine learning models
* Automatic best-model selection
* Model persistence using Pickle
* Interactive Streamlit dashboard
* Real-time income prediction
* Feature importance visualization
* Model performance analytics

---

## 🛠️ Tech Stack

* Python
* Streamlit
* Pandas
* NumPy
* Scikit-Learn
* XGBoost
* Plotly
* Matplotlib
* Seaborn
* Joblib / Pickle

---

## 📊 Machine Learning Models

* Logistic Regression
* Ridge Classifier
* Decision Tree Classifier
* Random Forest Classifier
* Gradient Boosting Classifier
* XGBoost Classifier
* Linear SVC

The application automatically evaluates all models and selects the best-performing model based on test accuracy.

---

## 📂 Project Structure

```text
├── app.py
├── adult_3.csv
├── requirements.txt
├── README.md
├── best_model.pkl
└── model_meta.pkl
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone <repository-url>
cd <repository-name>
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## 📈 Workflow

```text
Dataset
   ↓
Data Cleaning
   ↓
Feature Engineering
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Best Model Selection
   ↓
Model Saving
   ↓
Streamlit Deployment
```

---

## 📋 Dataset

Dataset: Adult Census Income Dataset

Target Variable:

* <=50K
* > 50K

The goal is to classify whether an individual's income exceeds $50,000 annually based on census attributes.

---

## 🎯 Key Learning Outcomes

* End-to-end Machine Learning pipeline development
* Data preprocessing and feature engineering
* Classification model evaluation and comparison
* Model deployment using Streamlit
* Interactive dashboard creation
* Production-ready project structure

---

## 🔮 Future Improvements

* Hyperparameter tuning
* SHAP explainability
* Docker containerization
* Cloud deployment (AWS/Azure/GCP)
* Batch prediction support
* User authentication

---

## 👨‍💻 Author

Yogesh Yadav

Data Science & Analytics Student | Machine Learning Enthusiast

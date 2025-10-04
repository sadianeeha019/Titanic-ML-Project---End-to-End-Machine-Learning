# 🚢 Titanic ML Project - End-to-End Machine Learning

![Python](https://img.shields.io/badge/Python-3.9-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)
![Colab](https://img.shields.io/badge/Open%20in-Colab-brightgreen)

## 📌 Project Overview
An **end-to-end Machine Learning project** predicting Titanic passenger survival.  
This project demonstrates:
- Data cleaning & preprocessing
- Feature engineering (FamilySize, IsAlone, Titles from names)
- Model training & hyperparameter tuning
- Evaluation (ROC, PR, Confusion Matrix)
- Deployment with **Streamlit App**
- Reproducible workflow with Colab & GitHub

## 📊 Dataset
We use the classic **Kaggle Titanic dataset**:  
[Download Titanic Data](https://www.kaggle.com/c/titanic/data)

## ⚙️ Features
- Exploratory Data Analysis (EDA) with visualizations
- Multiple Models: Logistic Regression, Random Forest, XGBoost
- Feature Importance ranking
- Hyperparameter Tuning with GridSearchCV
- Export best model (`joblib`)

## 🚀 Quick Start

### Run in Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR-USERNAME/titanic-ml-enhanced/blob/main/notebooks/colab_titanic_ml_enhanced.ipynb)

### Run Streamlit Demo
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py

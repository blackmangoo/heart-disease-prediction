# ❤️ Heart Disease Prediction

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red?logo=streamlit)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue?logo=docker)

> Predict heart disease risk using patient health data and classification models with explainable AI.

## 🎯 Objective

Build a model to predict whether a person is at **risk of heart disease** based on health metrics like age, cholesterol, blood pressure, etc. Use multiple classification models and explain predictions using SHAP.

## 📚 What You'll Learn

- **Exploratory Data Analysis (EDA)** on medical data
- **Data preprocessing** (handling missing values, encoding, scaling)
- **Classification modeling** (Logistic Regression, Decision Tree, Random Forest, SVM)
- **Model evaluation** (Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix)
- **Feature importance** and **SHAP explainability**
- **Handling class imbalance**
- **Building a health risk assessment UI** with Streamlit

## 🧠 Concepts to Revise Before Starting

| Concept | Resource |
|---------|----------|
| Logistic Regression | [StatQuest Video](https://www.youtube.com/watch?v=yIYKR4sgzI8) |
| Decision Trees | [StatQuest Video](https://www.youtube.com/watch?v=_L39rN6gz7Y) |
| Confusion Matrix | [StatQuest Video](https://www.youtube.com/watch?v=Kdsp6soqA7o) |
| ROC Curve & AUC | [StatQuest Video](https://www.youtube.com/watch?v=4jRBRDbJemM) |
| Precision vs Recall | [Google ML Guide](https://developers.google.com/machine-learning/crash-course/classification/precision-and-recall) |
| SHAP Values | [SHAP Intro Video](https://www.youtube.com/watch?v=VB9uV-x0veo) |
| Feature Scaling | [When to Scale](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_scaling_importance.html) |

## 📁 Project Structure

```
heart-disease-prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── Dockerfile
├── notebooks/
│   └── heart_disease_analysis.ipynb   ← Main analysis notebook
├── src/
│   ├── __init__.py
│   ├── data_loader.py                 ← Load & clean dataset
│   ├── eda.py                         ← Exploratory Data Analysis
│   ├── model.py                       ← Train & evaluate models
│   └── visualize.py                   ← Plotting functions
├── data/                               ← UCI Heart Disease dataset
├── models/                             ← Saved trained models
├── results/                            ← Plots and metrics
└── app/
    └── streamlit_app.py                ← Heart risk assessment UI
```

## 🚀 Step-by-Step Implementation Guide

### Step 1: Setup Environment
```bash
conda create -n heart-disease python=3.11 -y
conda activate heart-disease
pip install -r requirements.txt
```

### Step 2: Get the Dataset
- Download from [Kaggle - Heart Disease UCI](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-dataset)
- Place in `data/` folder
- Or load directly using `sklearn.datasets` or `ucimlrepo`

### Step 3: Data Loading & Cleaning (`src/data_loader.py`)
- Load the CSV dataset
- Check for missing values and handle them
- Check data types and fix if needed
- Encode categorical variables (if any)
- Split into features (X) and target (y)

### Step 4: Exploratory Data Analysis (`src/eda.py`)
- Distribution of target variable (balanced or imbalanced?)
- Correlation heatmap
- Feature distributions by class (heart disease vs no heart disease)
- Box plots for outlier detection
- Age vs heart disease distribution
- Pairplot for key features

### Step 5: Model Training (`src/model.py`)
- Split data into train/test (80/20, stratified)
- Scale features using StandardScaler
- Train multiple models:
  1. Logistic Regression
  2. Decision Tree
  3. Random Forest
  4. (Bonus) SVM
- For each model:
  - Calculate accuracy, precision, recall, F1
  - Generate confusion matrix
  - Generate ROC curve
- Compare all models in a summary table

### Step 6: Explainability
- Feature importance from tree-based models
- SHAP values for the best model
- SHAP summary plot and force plots
- Identify top risk factors

### Step 7: Visualization (`src/visualize.py`)
- ROC curves for all models (overlay plot)
- Confusion matrices (heatmap style)
- Feature importance bar chart
- SHAP summary plot
- Model comparison bar chart

### Step 8: Jupyter Notebook (`notebooks/heart_disease_analysis.ipynb`)
- Full end-to-end analysis with markdown explanations
- All plots inline
- Model comparison and final recommendation
- Clinical interpretation of results

### Step 9: Streamlit App (`app/streamlit_app.py`)
- Input form for patient health data
- Risk prediction with confidence percentage
- Risk gauge visualization (green/yellow/red)
- Feature importance chart
- SHAP explanation for individual prediction
- Beautiful medical-themed UI

### Step 10: Docker
```bash
docker build -t heart-disease .
docker run -p 8501:8501 heart-disease
```

## 🎯 Extra Challenges (Bonus Learning)

- [ ] Implement cross-validation (5-fold stratified)
- [ ] Hyperparameter tuning with GridSearchCV
- [ ] Handle class imbalance using SMOTE
- [ ] Add Gradient Boosting model
- [ ] Create a patient risk report PDF generator

## 📊 Results

### Model Performance
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|-----|---------|
| Logistic Regression | - | - | - | - | - |
| Decision Tree | - | - | - | - | - |
| Random Forest | - | - | - | - | - |

### Screenshots
<!-- Add screenshots of your Streamlit app and key plots here -->

## 🔗 Links

- **Dataset:** [Heart Disease UCI - Kaggle](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-dataset)
- **Internship:** DevelopersHub Corporation AI/ML Engineering

---
*Built as part of DevelopersHub Corporation AI/ML Engineering Internship*
import json, os

cells = []

def md(source):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in source.split("\n")]})

def code(source):
    cells.append({"cell_type": "code", "metadata": {}, "source": [line + "\n" for line in source.split("\n")], "outputs": [], "execution_count": None})

# ===== TITLE =====
md("# 🫀 Heart Disease Prediction — End-to-End ML Pipeline\n\n**Intern:** Ammar Akbar | **Organization:** DevelopersHub Corporation\n\n**Objective:** Predict whether a person is at risk of heart disease based on their health data using Logistic Regression and Decision Trees.\n\n---")

# ===== 1. IMPORTS =====
md("## 1. Setup & Imports\n\nWe import all necessary libraries:\n- `pandas` / `numpy` — data manipulation\n- `sklearn` — machine learning models & metrics\n- `plotly` / `seaborn` — interactive charts\n- `shap` — model explainability")

code("# Standard libraries\nimport pandas as pd\nimport numpy as np\nimport warnings\nwarnings.filterwarnings('ignore')\n\n# Machine Learning\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.tree import DecisionTreeClassifier\nfrom sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, classification_report\n\n# Visualization\nimport plotly.express as px\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport shap\n\nprint('All libraries imported successfully!')")

# ===== 2. DATA LOADING =====
md("## 2. Data Loading\n\nWe load the **Heart Disease UCI Dataset**.\n\n### What data do we get?\n- **age**: age in years\n- **sex**: (1 = male; 0 = female)\n- **cp**: chest pain type (4 values)\n- **trestbps**: resting blood pressure\n- **chol**: serum cholestoral in mg/dl\n- **fbs**: fasting blood sugar > 120 mg/dl\n- **restecg**: resting electrocardiographic results (values 0,1,2)\n- **thalach**: maximum heart rate achieved\n- **exang**: exercise induced angina\n- **oldpeak**: ST depression induced by exercise relative to rest\n- **slope**: the slope of the peak exercise ST segment\n- **ca**: number of major vessels (0-3) colored by flourosopy\n- **thal**: 3 = normal; 6 = fixed defect; 7 = reversable defect\n- **target**: 1 = disease risk, 0 = no risk")

code("# ============================================\n# STEP 2: Load the cleaned data\n# ============================================\ndf = pd.read_csv('../data/heart.csv')\n\nprint(f'Rows: {df.shape[0]}, Columns: {df.shape[1]}')\ndf.head()")

# ===== 3. EDA =====
md("## 3. Exploratory Data Analysis (EDA)\n\nLet's understand the distribution of our target variable and features.")

code("# Target Distribution\nfig = px.pie(df, names='target', title='Target Distribution (0 = Healthy, 1 = Heart Disease Risk)', hole=0.3)\nfig.show()")

code("# Correlation Heatmap\nplt.figure(figsize=(12, 10))\nsns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')\nplt.title('Feature Correlation Heatmap')\nplt.show()")

# ===== 4. DATA SPLITTING & SCALING =====
md("## 4. Train-Test Split & Scaling\n\nWe split the data 80/20. We use `StandardScaler` to scale our features because Logistic Regression is sensitive to feature scales.")

code("X = df.drop(columns=['target'])\ny = df['target']\n\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)\n\nscaler = StandardScaler()\nX_train_scaled = scaler.fit_transform(X_train)\nX_test_scaled = scaler.transform(X_test)\n\nprint('Data scaled successfully.')")

# ===== 5. MODELING =====
md("## 5. Model Training: Logistic Regression vs Decision Tree")

code("# Logistic Regression\nlr = LogisticRegression(random_state=42)\nlr.fit(X_train_scaled, y_train)\nlr_preds = lr.predict(X_test_scaled)\nlr_probs = lr.predict_proba(X_test_scaled)[:, 1]\n\n# Decision Tree\ndt = DecisionTreeClassifier(max_depth=5, random_state=42)\ndt.fit(X_train, y_train)\ndt_preds = dt.predict(X_test)\ndt_probs = dt.predict_proba(X_test)[:, 1]\n\nprint('Models trained.')")

# ===== 6. EVALUATION =====
md("## 6. Evaluation\n\nLet's look at the metrics: Accuracy, ROC-AUC, and Confusion Matrix.")

code("print('=== Logistic Regression ===')\nprint(f'Accuracy: {accuracy_score(y_test, lr_preds):.4f}')\nprint(f'ROC-AUC:  {roc_auc_score(y_test, lr_probs):.4f}')\nprint(classification_report(y_test, lr_preds))\n\nprint('\\n=== Decision Tree ===')\nprint(f'Accuracy: {accuracy_score(y_test, dt_preds):.4f}')\nprint(f'ROC-AUC:  {roc_auc_score(y_test, dt_probs):.4f}')\nprint(classification_report(y_test, dt_preds))")

code("# Confusion Matrix for best model (Logistic Regression)\ncm = confusion_matrix(y_test, lr_preds)\nplt.figure(figsize=(6,4))\nsns.heatmap(cm, annot=True, fmt='d', cmap='Blues')\nplt.title('Logistic Regression Confusion Matrix')\nplt.xlabel('Predicted')\nplt.ylabel('Actual')\nplt.show()")

# ===== 7. FEATURE IMPORTANCE (SHAP) =====
md("## 7. Model Explainability with SHAP\n\nSHAP values tell us exactly *why* the model makes its decisions and which features are most important.")

code("explainer = shap.LinearExplainer(lr, X_train_scaled)\nshap_values = explainer.shap_values(X_test_scaled)\n\nshap.summary_plot(shap_values, X_test, feature_names=X.columns)")

# Write notebook
os.makedirs('notebooks', exist_ok=True)
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"}
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open('notebooks/heart_disease.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)

print("Notebook generated at notebooks/heart_disease.ipynb")

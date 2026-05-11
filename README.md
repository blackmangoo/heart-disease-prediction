# 🫀 CardioCare AI — Heart Disease Predictor

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-purple?logo=python)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue?logo=docker)

> An end-to-end Machine Learning pipeline and diagnostic dashboard that predicts the risk of heart disease based on patient vitals. Features a highly accurate Logistic Regression model (86.89% Accuracy, 0.95 ROC-AUC) and state-of-the-art SHAP value explainability.

## 📋 Task Objective

Build a binary classification model that:
- Cleans and preprocesses the **UCI Heart Disease Dataset**
- Performs Exploratory Data Analysis (EDA)
- Trains and compares **Logistic Regression** and **Decision Tree** classifiers
- Integrates **SHAP** values for model interpretability
- Deploys an interactive **Streamlit dashboard** with a premium dark glassmorphism UI

## 📁 Project Structure

```
heart-disease-prediction/
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
├── Dockerfile                       # Docker containerization
├── .gitignore
├── app/
│   └── streamlit_app.py             # 🎨 Premium Streamlit dashboard with SHAP
├── src/
│   ├── data_loader.py               # Fetch & clean UCI dataset via API
│   ├── model.py                     # Train & evaluate ML models
│   └── create_notebook.py           # Auto-generate grading notebook
├── notebooks/
│   └── heart_disease.ipynb          # Full EDA & ML pipeline walkthrough
├── data/                            # Downloaded dataset
└── models/                          # Saved trained models & scalers
```

## 📊 Results

### Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Logistic Regression** 🏆 | **86.89%** | **81.25%** | **92.86%** | **86.67%** | **0.9513** |
| Decision Tree | 78.69% | 72.73% | 85.71% | 78.69% | 0.8047 |

**Winner: Logistic Regression** — Logistic Regression strongly outperformed the Decision Tree, likely because medical vitals (like blood pressure, cholesterol, and heart rate) have linear or monotonic relationships with disease risk. The extremely high ROC-AUC of 0.95 shows excellent separability between healthy and at-risk patients.

### Key Insights (from SHAP)

1.  **Chest Pain Type (`cp`)** is consistently the most critical predictor of heart disease risk.
2.  **Number of major vessels (`ca`)** and **Thalassemia (`thal`)** are significant secondary indicators.
3.  **Maximum Heart Rate Achieved (`thalach`)** shows a clear pattern where lower max heart rate under exercise correlates with higher risk.

## 🚀 Quick Start

### Local Setup
```bash
# Clone the repository
git clone https://github.com/blackmangoo/heart-disease-prediction.git
cd heart-disease-prediction

# Create environment & install dependencies
pip install -r requirements.txt

# Run the data pipeline and train the model
python src/data_loader.py
python src/model.py

# Run the dashboard
streamlit run app/streamlit_app.py
```

## 🛠 Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.11 | Core language |
| ucimlrepo | Dataset fetching API |
| pandas / numpy | Data manipulation |
| scikit-learn | Machine Learning |
| Plotly / Seaborn | Interactive charts |
| SHAP | Model Explainability |
| Streamlit | Web dashboard |
| Docker | Containerization |

## 📄 License

This project is licensed under the MIT License.

---

**Internship Task 3** — DevelopersHub Corporation AI/ML Engineering Internship | May 2026

Built by [Ammar Akbar](https://github.com/blackmangoo)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# ==========================================
# 1. PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(page_title="CardioCare AI", page_icon="🫀", layout="wide")

st.markdown("""
<style>
    /* Premium Dark Theme with Glassmorphism */
    :root {
        --primary-bg: #0f172a;
        --card-bg: rgba(30, 41, 59, 0.7);
        --accent: #ef4444; /* Red accent for heart theme */
        --text-main: #f8fafc;
        --text-muted: #94a3b8;
    }
    
    .stApp {
        background-color: var(--primary-bg);
        color: var(--text-main);
    }
    
    .glass-card {
        background: var(--card-bg);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
    }
    
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #ef4444, #f43f5e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    
    .hero-subtitle {
        color: var(--text-muted);
        text-align: center;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: var(--accent);
    }
    
    .metric-label {
        color: var(--text-muted);
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    hr {
        border-color: rgba(255, 255, 255, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. LOAD MODELS & DATA
# ==========================================
@st.cache_resource
def load_models():
    lr_model = joblib.load('models/logistic_regression.joblib')
    dt_model = joblib.load('models/decision_tree.joblib')
    scaler = joblib.load('models/scaler.joblib')
    X_train = pd.read_csv('models/X_train.csv')
    return lr_model, dt_model, scaler, X_train

try:
    lr_model, dt_model, scaler, X_train = load_models()
except FileNotFoundError:
    st.error("Models not found. Please run `python src/model.py` first.")
    st.stop()

# Initialize SHAP Explainer (using Logistic Regression coefficients for linear explainer)
explainer = shap.LinearExplainer(lr_model, scaler.transform(X_train))

# ==========================================
# 3. HEADER
# ==========================================
st.markdown('<div class="hero-title">CardioCare AI 🫀</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Advanced Heart Disease Risk Prediction using Machine Learning</div>', unsafe_allow_html=True)

# ==========================================
# 4. SIDEBAR - PATIENT INPUT
# ==========================================
st.sidebar.markdown("## 📋 Patient Vitals Form")

# Create input fields based on the UCI dataset features
age = st.sidebar.slider("Age", 20, 100, 50)
sex = st.sidebar.selectbox("Sex", options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
cp = st.sidebar.selectbox("Chest Pain Type (cp)", options=[1, 2, 3, 4], 
                          help="1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic")
trestbps = st.sidebar.slider("Resting Blood Pressure (mm Hg)", 90, 200, 120)
chol = st.sidebar.slider("Serum Cholestoral (mg/dl)", 100, 600, 200)
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[1, 0], format_func=lambda x: "True" if x == 1 else "False")
restecg = st.sidebar.selectbox("Resting ECG Results", options=[0, 1, 2], 
                               help="0: normal, 1: ST-T wave abnormality, 2: probable/definite left ventricular hypertrophy")
thalach = st.sidebar.slider("Maximum Heart Rate Achieved", 60, 220, 150)
exang = st.sidebar.selectbox("Exercise Induced Angina", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
oldpeak = st.sidebar.slider("ST depression induced by exercise", 0.0, 6.0, 1.0, step=0.1)
slope = st.sidebar.selectbox("Slope of peak exercise ST segment", options=[1, 2, 3],
                             help="1: upsloping, 2: flat, 3: downsloping")
ca = st.sidebar.slider("Number of major vessels colored by flourosopy", 0, 3, 0)
thal = st.sidebar.selectbox("Thalassemia", options=[3, 6, 7],
                            help="3: normal, 6: fixed defect, 7: reversable defect")

# Compile input into a dataframe
input_data = pd.DataFrame({
    'age': [age], 'sex': [sex], 'cp': [cp], 'trestbps': [trestbps], 'chol': [chol],
    'fbs': [fbs], 'restecg': [restecg], 'thalach': [thalach], 'exang': [exang],
    'oldpeak': [oldpeak], 'slope': [slope], 'ca': [ca], 'thal': [thal]
})

# ==========================================
# 5. PREDICTION LOGIC
# ==========================================
# Scale the input
input_scaled = scaler.transform(input_data)

# Predict
prediction_prob = lr_model.predict_proba(input_scaled)[0][1]
prediction = int(prediction_prob > 0.5)

# ==========================================
# 6. MAIN CONTENT AREA
# ==========================================
tab1, tab2, tab3 = st.tabs(["Diagnostics", "AI Explainability (SHAP)", "Model Performance"])

with tab1:
    st.markdown("### 🔍 Real-Time AI Diagnosis")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if prediction == 1:
            st.error("🚨 **ELEVATED RISK DETECTED**")
            st.markdown("The AI model has detected patterns consistent with heart disease.")
        else:
            st.success("✅ **LOW RISK DETECTED**")
            st.markdown("The AI model indicates a low probability of heart disease.")
            
    with col2:
        # Gauge chart for probability
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prediction_prob * 100,
            title = {'text': "Risk Probability %"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "rgba(239, 68, 68, 0.8)" if prediction == 1 else "rgba(34, 197, 94, 0.8)"},
                'steps': [
                    {'range': [0, 40], 'color': "rgba(34, 197, 94, 0.2)"},
                    {'range': [40, 60], 'color': "rgba(234, 179, 8, 0.2)"},
                    {'range': [60, 100], 'color': "rgba(239, 68, 68, 0.2)"}
                ]
            }
        ))
        fig.update_layout(height=250, margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("### 🧠 Why did the AI make this prediction?")
    st.markdown("We use **SHAP (SHapley Additive exPlanations)** to break down exactly how each of your vital signs influenced the final risk probability.")
    
    # Calculate SHAP values for the specific input
    shap_values = explainer.shap_values(input_scaled)
    
    # Create a nice waterfall/bar plot using plotly
    # Since it's a single prediction, we can plot the contribution of each feature
    features = input_data.columns
    contributions = shap_values[0]
    
    # Sort by absolute contribution
    sorted_idx = np.argsort(np.abs(contributions))
    
    fig2 = go.Figure(go.Bar(
        x=contributions[sorted_idx],
        y=[features[i] for i in sorted_idx],
        orientation='h',
        marker_color=['rgba(239, 68, 68, 0.8)' if x > 0 else 'rgba(34, 197, 94, 0.8)' for x in contributions[sorted_idx]]
    ))
    
    fig2.update_layout(
        title="Feature Contributions to Risk Score (Red = Increased Risk, Green = Decreased Risk)",
        xaxis_title="Impact on Prediction (Log Odds)",
        yaxis_title="Feature",
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "white"}
    )
    
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.markdown("### 📊 Under the Hood: Model Performance")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-label">Model Architecture</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-value" style="font-size: 1.5rem;">Logistic Regression</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-label">Test Accuracy</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-value">86.89%</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-label">ROC-AUC Score</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-value">0.9513</div>', unsafe_allow_html=True)
        
    st.markdown("<br><p style='color: var(--text-muted);'>Trained on the UCI Heart Disease Dataset. Features 80/20 chronological split with standard scaling.</p>", unsafe_allow_html=True)

st.markdown("<div style='text-align: center; color: var(--text-muted); margin-top: 50px;'>Built with ❤️ by Ammar Akbar - DevelopersHub AI/ML Internship</div>", unsafe_allow_html=True)

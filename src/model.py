import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

def train_and_evaluate(data_path='data/heart.csv', models_dir='models'):
    """
    Trains Logistic Regression and Decision Tree models on the Heart Disease dataset.
    Evaluates their performance and saves the best model.
    """
    print("Loading data for training...")
    df = pd.read_csv(data_path)
    
    # Split into features (X) and target (y)
    X = df.drop(columns=['target'])
    y = df['target']
    
    # Train/Test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Feature Scaling (Crucial for Logistic Regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\n--- Training Logistic Regression ---")
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)
    
    lr_preds = lr_model.predict(X_test_scaled)
    lr_probs = lr_model.predict_proba(X_test_scaled)[:, 1]
    
    lr_metrics = {
        'Accuracy': accuracy_score(y_test, lr_preds),
        'Precision': precision_score(y_test, lr_preds),
        'Recall': recall_score(y_test, lr_preds),
        'F1-Score': f1_score(y_test, lr_preds),
        'ROC-AUC': roc_auc_score(y_test, lr_probs)
    }
    
    for metric, value in lr_metrics.items():
        print(f"{metric}: {value:.4f}")
        
    print("\n--- Training Decision Tree ---")
    dt_model = DecisionTreeClassifier(random_state=42, max_depth=5)
    dt_model.fit(X_train, y_train) # Tree models don't require scaling, but we'll train on unscaled X
    
    dt_preds = dt_model.predict(X_test)
    dt_probs = dt_model.predict_proba(X_test)[:, 1]
    
    dt_metrics = {
        'Accuracy': accuracy_score(y_test, dt_preds),
        'Precision': precision_score(y_test, dt_preds),
        'Recall': recall_score(y_test, dt_preds),
        'F1-Score': f1_score(y_test, dt_preds),
        'ROC-AUC': roc_auc_score(y_test, dt_probs)
    }
    
    for metric, value in dt_metrics.items():
        print(f"{metric}: {value:.4f}")

    # Determine Winner (based on Accuracy)
    winner = "Logistic Regression" if lr_metrics['Accuracy'] >= dt_metrics['Accuracy'] else "Decision Tree"
    print(f"\nWinner: {winner}")
    
    # Save the models and the scaler
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(lr_model, os.path.join(models_dir, 'logistic_regression.joblib'))
    joblib.dump(dt_model, os.path.join(models_dir, 'decision_tree.joblib'))
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.joblib'))
    
    # Also save the training data for SHAP explainer initialization later
    X_train.to_csv(os.path.join(models_dir, 'X_train.csv'), index=False)
    
    print("\nModels and scaler saved successfully in 'models/' directory.")
    
    return {
        'Logistic Regression': lr_metrics,
        'Decision Tree': dt_metrics
    }

if __name__ == "__main__":
    train_and_evaluate()

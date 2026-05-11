import pandas as pd
import numpy as np
import os
from ucimlrepo import fetch_ucirepo

def get_heart_disease_data(save_path='data/heart.csv'):
    """
    Fetches the Heart Disease dataset directly from UCI Machine Learning Repository.
    Cleans missing values and binarizes the target variable.
    """
    print("Fetching Heart Disease dataset from UCI...")
    try:
        # Fetch dataset by ID
        heart_disease = fetch_ucirepo(id=45) 
        
        # Extract features and targets
        X = heart_disease.data.features 
        y = heart_disease.data.targets 
        
        # Combine into a single DataFrame
        df = pd.concat([X, y], axis=1)
        
        # The target in the original dataset is 'num' (0 = no disease, 1-4 = disease)
        # We need to convert this to a binary classification problem (0 or 1)
        target_col = 'num' if 'num' in df.columns else 'target'
        df['target'] = (df[target_col] > 0).astype(int)
        
        # Drop the original multi-class target if it was named 'num'
        if 'num' in df.columns:
            df = df.drop(columns=['num'])
            
        # Clean missing values: The Cleveland dataset has some missing values in 'ca' and 'thal'
        # Convert any potential string representations of missing values to NaN
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col].replace('?', np.nan), errors='coerce')
        
        # Impute missing values with the median of the column
        df = df.fillna(df.median())
        
        # Save to disk so we don't have to fetch it from the internet every time
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"Dataset successfully saved to {save_path}")
        
        return df
        
    except Exception as e:
        print(f"Error fetching data: {e}")
        # If fetching fails (e.g., no internet), try to load from local cache
        if os.path.exists(save_path):
            print("Loading from local cache...")
            return pd.read_csv(save_path)
        else:
            raise Exception("No local data found and failed to fetch from UCI.")

if __name__ == "__main__":
    df = get_heart_disease_data()
    print("Dataset shape:", df.shape)
    print("Target distribution:\n", df['target'].value_counts())
    print("\nMissing values:\n", df.isnull().sum())

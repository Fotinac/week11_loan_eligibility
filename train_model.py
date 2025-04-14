"""
train_model.py
Trains a Random Forest model using one-hot encoded features.
Modularized for reuse in both main app and testing.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
import os
from utils.logger import setup_logger

logger = setup_logger()

def train_rf_model():
    try:
        logger.info("Starting training process...")

        # Load and preprocess data
        df = pd.read_csv("data/credit.csv")
        logger.info("Dataset loaded.")
        df = df.drop(columns=["Loan_ID"]).dropna()
        df["Dependents"] = df["Dependents"].replace("3+", "3")
        categorical = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
        df[categorical] = df[categorical].astype(str)
        df_encoded = pd.get_dummies(df, columns=categorical, dtype=int)

        X = df_encoded.drop(columns=["Loan_Approved"])
        y = df_encoded["Loan_Approved"].map({'Y': 1, 'N': 0})

        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        logger.info("Model trained successfully.")

        os.makedirs("models", exist_ok=True)
        with open("models/rf_model.pkl", "wb") as f:
            pickle.dump(model, f)
        logger.info("Model saved to models/rf_model.pkl")

        pd.DataFrame([X.columns.tolist()]).to_csv("models/feature_names.csv", index=False, header=False)
        logger.info("Feature names saved.")

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

# Correct position: only run if this file is called directly
if __name__ == "__main__":
    train_rf_model()


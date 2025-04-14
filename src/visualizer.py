import streamlit as st
import matplotlib.pyplot as plt
import pickle
import pandas as pd

def plot_feature_importance():
    # Load model and feature names
    with open("models/rf_model.pkl", "rb") as f:
        model = pickle.load(f)

    feature_names = pd.read_csv("models/feature_names.csv", header=None).squeeze().tolist()
    importances = model.feature_importances_

    # Sort by importance
    sorted_idx = sorted(range(len(importances)), key=lambda i: importances[i])
    sorted_names = [feature_names[i] for i in sorted_idx]
    sorted_importances = [importances[i] for i in sorted_idx]

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.barh(sorted_names, sorted_importances)
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance Chart")
    st.pyplot(fig)

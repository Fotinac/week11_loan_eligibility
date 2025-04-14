"""
main.py
Core logic that ties together input preprocessing, model prediction, and visualizations.
Used by app.py to keep Streamlit code clean and focused.
"""

from src.data_loader import preprocess_input
from src.model import load_model, predict
from src.visualizer import plot_feature_importance
from utils.logger import setup_logger

logger = setup_logger()

def predict_loan_eligibility(user_input: dict):
    model = load_model()
    X = preprocess_input(user_input)
    return predict(model, X)

def show_feature_importance_chart():
    plot_feature_importance()

## Live App  
[Click here to open the app](https://week11loaneligibility-fjxcwrbxlbtvvsq8gvtsbj.streamlit.app)


# Credit Loan Eligibility Predictor

A modular, Streamlit-based web application that predicts whether an applicant is eligible for a loan based on financial and demographic inputs. Built using Python and Random Forest, this project demonstrates model training, one-hot encoding, logging, error handling, and interactive UI development.

---

## Folder Structure

```
week11_loan_eligibility_app/
├── app.py                   # Streamlit user interface
├── main.py                  # Core logic: prediction + visualization
├── train_model.py           # Model training script
├── requirements.txt         # Required Python packages
├── README.md                # Project documentation

├── data/
│   └── credit.csv           # Original dataset

├── models/
│   ├── rf_model.pkl         # Trained model
│   └── feature_names.csv    # Column order used for prediction

├── logs/
│   └── app.log              # Log file for debugging

├── src/
│   ├── __init__.py
│   ├── data_loader.py       # Input preprocessing (one-hot encoding)
│   ├── model.py             # Model loading and prediction logic
│   └── visualizer.py        # Feature importance chart
```

---

## How to Run the App

1. **Install dependencies**  
```bash
pip install -r requirements.txt
```

2. **Train the model** (optional — only needed if retraining)  
```bash
python train_model.py
```

3. **Launch the Streamlit app**  
```bash
streamlit run app.py
```

---

## Model & Dataset

- Model: Random Forest Classifier (sklearn)
- Dataset: `credit.csv`, includes features like gender, income, loan amount, and credit history
- Encoding: One-hot encoding for categorical variables
- Target: `Loan_Approved` (Y/N → 1/0)

---

## Modularization

All major components are separated into independent modules:
- `train_model.py`: trains and saves the model + feature names
- `data_loader.py`: handles input cleaning and one-hot encoding
- `model.py`: loads model and makes predictions
- `visualizer.py`: displays feature importances using matplotlib
- `main.py`: ties everything together for app use
- `app.py`: clean Streamlit UI, no business logic inside

---

## Logging & Error Handling

- All core functions include `try/except` blocks with logging
- Logs are stored in `logs/app.log` for easier debugging
- `logger` is initialized via `utils/logger.py` and shared across modules

Example log output:
```
2025-04-13 21:45:23 - INFO - Model trained successfully.
2025-04-13 21:47:11 - INFO - Prediction result: 1
```

---

## Dependencies

```
pandas
numpy
scikit-learn
streamlit
matplotlib
```

---

## Author

Developed by Fotina Cao as part of Week 11 project in the CST2213 Business Intelligence Programming course.

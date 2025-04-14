import pandas as pd

def preprocess_input(input_dict):
    """
    Preprocess the user input to match the model’s one-hot encoded features.
    """
    # Convert single row to DataFrame
    df = pd.DataFrame([input_dict])

    # Clean values to match training format
    df['Dependents'] = df['Dependents'].replace('3+', '3')

    # Convert column types to string so get_dummies works correctly
    categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
    df[categorical_cols] = df[categorical_cols].astype(str)

    # One-hot encode
    df_encoded = pd.get_dummies(df, columns=categorical_cols, dtype=int)

    # Align with training features (fill missing columns with 0s)
    feature_names = pd.read_csv("models/feature_names.csv", header=None).squeeze().tolist()
    for col in feature_names:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    # Reorder columns to match model input
    df_encoded = df_encoded[feature_names]

    return df_encoded

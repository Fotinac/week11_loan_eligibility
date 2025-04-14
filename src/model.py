import pickle

def load_model(path="models/rf_model.pkl"):
    with open(path, "rb") as file:
        return pickle.load(file)

def predict(model, X):
    return model.predict(X)[0]

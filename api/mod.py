from sklearn.externals import joblib


def predict(model, data):
    if isinstance(model, str):
        model = joblib.load(model)
    y_pred = model.predict_proba(data)[:, 1]
    return y_pred

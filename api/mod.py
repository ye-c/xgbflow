from sklearn.externals import joblib
import numpy as np


def predict_df(model_path, df):
    model = joblib.load(model_path)
    y_pred = [x[1] for x in model.predict_proba(df)]
    return y_pred

from sklearn.externals import joblib
import pandas as pd
import numpy as np


def predict(model, data):
    if isinstance(model, str):
        model = joblib.load(model)
    y_pred = model.predict_proba(data)[:, 1]
    return y_pred


def accuracy_print(label, score):
    df = pd.DataFrame()
    df['label'] = label
    df['score'] = score

    ret = []
    for i in np.arange(max(score), 0, -0.001):
        tmp = df[df.score >= i]
        num = tmp.shape[0]
        real = tmp[tmp.label == 1].shape[0]
        rat = (real * 100 / num)
        print('%-30s %-10s %-10s %s' % (i, num, real, '%.2f%%' % rat))
        ret.append((i, '%.2f%%' % rat))
    return ret

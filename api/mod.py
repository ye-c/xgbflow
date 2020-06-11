import joblib
from sklearn import metrics
import xgboost as xgb
import pandas as pd
import numpy as np


def auc(label, pre):
    return metrics.roc_auc_score(label, pre)


def ks(label, pre):
    fpr, tpr, thresholds = metrics.roc_curve(label, pre)
    ks = max(tpr - fpr)
    return fpr, tpr, ks


def predict_proba(model, data):
    '''
    data = np.array([[x1, x2, ...],])
    '''
    if isinstance(model, str):
        model = joblib.load(model)
    y_pred = model.predict_proba(data)[:, 1]
    return y_pred


def predict(model, data):
    if isinstance(model, str):
        model = joblib.load(model)
    y_pred = model.predict(data)
    return y_pred


def bst_predict(bst, data):
    if isinstance(bst, str):
        bst = xgb.Booster(model_file=bst)
    dtest = xgb.DMatrix(data)
    y_pred = bst.predict(dtest)
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


def dump(model, savePath='./dump.raw.txt'):
    '''
    1. 将训练好的模型，用 save_model 方法保存为模型文件；
    2. 使用 xgb.Booster 读取模型文件，joblib保存的模型文件读取会报错；
    3. bst.dump_model 方法得到模型描述文件；

    '''
    try:
        bst = xgb.Booster({'nthread':4})
        bst.load_model(model)
        bst.dump_model(savePath)
    except Exception as e:
        print('Error:', e)


if __name__ == '__main__':
    mod = '/Users/yec/welab/ing/h5_model_monitor/h5_antifraud/model/xgb_h5_release_dump.m'
    sav = '/Users/yec/welab/ing/h5_model_monitor/h5_antifraud/model/dump.raw.txt'
    dump(mod, sav)

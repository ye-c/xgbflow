import os
import numpy as np
import pandas as pd
import xgboost as xgb
from xgbflow.api import mod, draw
from xgbflow.utils.markitdown import MarkitDown


# def classifier_train(data, classifier=None, do_eval=True):
#     '''
#     data = [
#         ('train', trn_x.values, trn_y),
#         ('test', tet_x.values, tet_y),
#         ('v1', vr1_x.values, vr1_y),
#         ('v2', vr2_x.values, vr2_y)
#     ]
#     model = xflow.classifier_train(data)
#     joblib.dump(model, 'model/xgb_h5_classifier.model')
#     xflow.classifier_verify(model, data)
#     '''
#     if not classifier:
#         classifier = xgb.XGBClassifier(
#             max_depth=3,
#             learning_rate=0.1,
#             n_estimators=50,
#             # verbosity=1,
#             objective='binary:logistic',
#             booster='gbtree',
#             # n_jobs=1,
#             # nthread=None,
#             gamma=0.5,
#             min_child_weight=2,
#             max_delta_step=0,
#             subsample=0.85,
#             # colsample_bytree=1,
#             # colsample_bylevel=1,
#             # colsample_bynode=1,
#             reg_alpha=30,
#             reg_lambda=15,
#             scale_pos_weight=3,
#             # base_score=0.5,
#             # random_state=0,
#             seed=0,
#             silent=1
#         )
#     n_train, x_train, y_train = data[0]
#     eval_list = [(x, y) for n, x, y in data] if do_eval else None
#     classifier.fit(x_train, y_train,
#                    eval_set=eval_list,
#                    eval_metric='auc',
#                    # early_stopping_rounds=10,
#                    )
#     return classifier


def classifier_train(data, classifier=None, label='y', do_eval=True):
    '''
    data = {
        'train': trn_df,
        'test': tet_df,
        'v1': vr1_df,
        'v2': vr2_d,
    }
    model = xflow.classifier_train(data)
    joblib.dump(model, 'model/xgb_h5_classifier.model')
    xflow.classifier_verify(model, data)
    '''
    if not classifier:
        classifier = xgb.XGBClassifier(
            max_depth=3,
            learning_rate=0.1,
            n_estimators=50,
            # verbosity=1,
            objective='binary:logistic',
            booster='gbtree',
            # n_jobs=1,
            # nthread=None,
            gamma=0.5,
            min_child_weight=2,
            max_delta_step=0,
            subsample=0.85,
            # colsample_bytree=1,
            # colsample_bylevel=1,
            # colsample_bynode=1,
            reg_alpha=30,
            reg_lambda=15,
            scale_pos_weight=3,
            # base_score=0.5,
            # random_state=0,
            seed=0,
            silent=1
        )
    train_df = data['train']
    x_train = train_df.drop([label], axis=1).values
    y_train = train_df[label]
    eval_list = [(df.drop([label], axis=1).values, df[label])
                 for df in data.values()] if do_eval else None
    classifier.fit(
        x_train, y_train,
        eval_set=eval_list,
        eval_metric='auc',
        # early_stopping_rounds=10,
    )
    return classifier


def classifier_verify(model, data, label='y', verify=None,
                      draw_out=False, title='title'):
    '''
    data = {
        'train': trn_df,
        'test': tet_df,
        'v1': vr1_df,
        'v2': vr2_d,
    }
    aklist, top_data, pltlist = xflow.classifier_verify(
        model, data,
        label='y',
        verify='vrf1',
        draw_out=True,
        title='H5 Model'
    )
    '''
    aklist, pltlist, top_data = [], [], []
    for name, df in data.items():
        x = df.drop([label], axis=1).values
        y = df[label]
        y_pred = model.predict_proba(x)[:, 1]
        aucks, plts = calc_aucks(y, y_pred,
                                 is_draw=draw_out,
                                 title=title,
                                 save_as=name)
        aucks.insert(0, name)
        aklist.append(aucks)
        plts.insert(0, name)
        pltlist.append(plts)

        if name == verify:
            top_data = top_status(y, y_pred)

    return aklist, top_data, pltlist


def calc_aucks(label, pre, is_draw=False, title='default', save_as='default'):
    auc = mod.auc(label, pre)
    fpr, tpr, ks = mod.ks(label, pre)
    auc, ks = round(auc, 4), round(ks, 4)
    pltls = []
    if is_draw:
        pltdir = os.path.abspath('.') + '/plt'
        if not os.path.exists(pltdir):
            os.makedirs(pltdir)
        save_path = os.path.join(pltdir, save_as)
        pltls.append('![](%s)' % draw.auc(fpr, tpr, auc, ks, title, save_path))
        pltls.append('![](%s)' % draw.ks(fpr, tpr, ks, save_path))
        pltls.append('![](%s)' % draw.score_dist(pre, save_path))
    return [auc, ks], pltls


def feature_importance(features, model):
    if isinstance(features, dict):
        imp = {k: v for k, v in zip(
            features.values(), model.feature_importances_)}
    else:
        imp = {k: v for k, v in zip(features, model.feature_importances_)}
    imp = sorted(imp.items(), key=lambda item: item[1])
    imp.reverse()
    for i in range(len(imp)):
        imp[i] = (i + 1,) + imp[i]
    return imp


def top_status(label, score):
    '''
    columns=['TOP N', 'CutOff', '真欺诈', '总数', '准确率', '召回率']
    '''
    index_map = {'95%': 'TOP 5%', '90%': 'TOP 10%', '85%': 'TOP 15%',
                 '80%': 'TOP 20%', '70%': 'TOP 30%', '50%': 'TOP 50%'}
    df = pd.DataFrame()
    df['score'] = score
    df['label'] = label
    bad_num = df[df.label == 1].shape[0]
    desc = df.score.describe(percentiles=[0.7, 0.8, 0.85, 0.9, 0.95])
    desc = desc[4:10]
    res = []
    for index in desc.index:
        dd = df[df.score >= desc[index]]
        num = dd.shape[0]
        real = dd[dd.label == 1].shape[0]
        res.append((index_map[index],
                    '%.4f' % desc[index],
                    real,
                    num,
                    '%.2f%%' % (real * 100 / num),
                    '%.2f%%' % (real * 100 / bad_num)))
    res.reverse()
    return res


def flow_feature_all(n_est, data):
    model = xgb.XGBClassifier(
        max_depth=3,
        learning_rate=0.1,
        n_estimators=n_est,
        # verbosity=1,
        objective='binary:logistic',
        booster='gbtree',
        # n_jobs=1,
        # nthread=None,
        gamma=0.5,
        min_child_weight=2,
        max_delta_step=0,
        subsample=0.85,
        # colsample_bytree=1,
        # colsample_bylevel=1,
        # colsample_bynode=1,
        reg_alpha=30,
        reg_lambda=15,
        scale_pos_weight=3,
        # base_score=0.5,
        # random_state=0,
        seed=0,
        silent=1
    )
    n_train, x_train, y_train = data[0]
    eval_list = [(x.values, y) for n, x, y in data]
    model.fit(x_train.values, y_train,
              eval_set=eval_list,
              eval_metric='auc')
    return model


def flow_feature_onebyone(model, feature_importance_list, data, testi=1, veri=2):
    imp_list = {}
    for f in feature_importance_list.index:
        print(f)
        xdata = [(n, x[[f]].values, y) for n, x, y in data]
        model.fit(xdata[0][1], xdata[0][2])
        aklist, top_data, pltlist = classifier_verify(model, xdata)
        for n, auc, ks in aklist:
            print('%-10s %-5s' % (n, ks))
        test2veri = aklist[testi][2] - aklist[veri][2]
        imp_list[f] = test2veri
        print('%-10s - %-10s = %s' %
              (xdata[testi][0], xdata[veri][0], test2veri))
    imp_rank = sorted(imp_list.items(), key=lambda imp_list: imp_list[1])
    return imp_rank


def flow_feature_combine(model, imp_rank, data, testi=1, veri=2):
    res = []
    for k, v in imp_rank:
        res.append(k)
        xdata = [(n, x[res].values, y) for n, x, y in data]
        model.fit(xdata[0][1], xdata[0][2])
        aklist, top_data, pltlist = classifier_verify(model, xdata)
        test2veri = aklist[testi][2] - aklist[veri][2]

        ksls = [ak[2] for ak in aklist]
        if test2veri <= 0.05:
            print(res)
            print(len(res), ksls, test2veri)
        else:
            print(len(res), ksls, test2veri, k)
            res.remove(k)
    return res


def steady_flow(n_est, data):
    print('All feature trainning')
    model = flow_feature_all(n_est, data)

    fimp = pd.Series(model.feature_importances_,
                     index=data[0][1].columns.tolist())
    fimp = fimp[fimp > 0]
    print('Importance feature:', len(fimp))

    print('Single feature trainning')
    imp_rank = flow_feature_onebyone(model, fimp, data)
    print(imp_rank)

    # print('Feature combining')
    # res = flow_feature_combine(model, imp_rank, data)

    # return res


# def steady_flow(n_est, x_train, y_train, x_test, y_test, x_verify, y_verify):
#     print('All feature trainning')
#     model = xgb.XGBClassifier(max_depth=3,
#                               learning_rate=0.1,
#                               n_estimators=n_est,
#                               # verbosity=1,
#                               objective='binary:logistic',
#                               booster='gbtree',
#                               # n_jobs=1,
#                               # nthread=None,
#                               gamma=0.5,
#                               min_child_weight=2,
#                               max_delta_step=0,
#                               subsample=0.85,
#                               # colsample_bytree=1,
#                               # colsample_bylevel=1,
#                               # colsample_bynode=1,
#                               reg_alpha=30,
#                               reg_lambda=15,
#                               scale_pos_weight=3,
#                               # base_score=0.5,
#                               # random_state=0,
#                               seed=0,
#                               silent=1)
#     model.fit(x_train.values, y_train,
#               eval_set=[(x_train.values, y_train),
#                         (x_test.values, y_test),
#                         (x_verify.values, y_verify)],
#               eval_metric='auc')

#     fimp = pd.Series(model.feature_importances_,
#                      index=x_train.columns.tolist())
#     fimp = fimp[fimp > 0]
#     print('importance feature:', len(fimp))

#     print('Single feature trainning')
#     imp_list = {}
#     for f in fimp.index:
#         x_tr = x_train[[f]].values
#         x_te = x_test[[f]].values
#         x_ve = x_verify[[f]].values
#         model.fit(x_tr, y_train)
#         aklist = classifier_verify(
#             model, x_tr, y_train, x_te, y_test, x_ve, y_verify)[0]
#         test2veri = aklist[1][2] - aklist[2][2]
#         imp_list[f] = test2veri
#         print('%-80s %s' % (f, test2veri))
#     imp_rank = sorted(imp_list.items(), key=lambda imp_list: imp_list[1])

#     print('Feature combining')
#     res = []
#     for k, v in imp_rank:
#         res.append(k)
#         x_tr = x_train[res].values
#         x_te = x_test[res].values
#         x_ve = x_verify[res].values
#         model.fit(x_tr, y_train)
#         aklist = classifier_verify(
#             model, x_tr, y_train, x_te, y_test, x_ve, y_verify)[0]
#         test2veri = aklist[1][2] - aklist[2][2]

#         trks, teks, veks = aklist[0][2], aklist[1][2], aklist[2][2]
#         if test2veri <= 0.05:
#             print(res)
#             print('%-3s %-4s %-4s %-4s -- %-4s \n' %
#                   (len(res), trks, teks, veks, test2veri))
#         else:
#             print('%-3s %-4s %-4s %-4s -- %-4s %s \n' %
#                   (len(res), trks, teks, veks, test2veri, k))
#             res.remove(k)

#     return res


'''
Booster functions below, unfinished.

'''


def booster_train(dtrain, dtest, dverify, model=None):
    params = {
        # 通用参数
        'booster': 'gbtree',                # 模型 gbtree=树形 gbliner=线性
        'silent': 1,                        # 1=静默 0=输出
        'nthread': -1,                      # 默认值为最大可能的线程数(-1)

        # booster参数
        # 学习率。通过减少每一步的权重，可以提高模型的鲁棒性 default=0.3 (classifier的learning_rate)
        'eta': 0.1,
        'min_child_weight': 2,              # 决定最小叶子节点样本权重和 default=1
        'max_depth': 3,                     # 树的最大深度 default=6
        # 'max_leaf_nodes': 4,              # 树上最大的节点或叶子的数量 如果定义了这个参数，GBM会忽略max_depth参数
        'gamma': 0.5,                       # 指定了节点分裂所需的最小损失函数下降值，值越大，算法越保守
        'alpha': 30,                        # 权重的L1正则化项
        'lambda': 15,                       # 权重的L2正则化项
        # 减小这个参数的值，算法会更加保守，避免过拟。但是，如果这个值设置得过小，它可能会导致欠拟合。典型值：0.5-1
        # 用来控制每棵随机采样的列数的占比(每一列是一个特征)。典型值：0.5-1
        'subsample': 0.85,
        'colsample_bytree': 0.75,
        'max_delta_step': 0,                # 这参数限制每棵树权重改变的最大步长。
        # 如果这个参数的值为0，那就意味着没有约束。如果它被赋予了某个正值，那么它会让这个算法更加保守。
        # 学习目标参数
        'objective': 'binary:logistic',     # 定义需要被最小化的损失函数
        # binary:logistic 二分类的逻辑回归，返回预测的概率(不是类别)。
        # multi:softmax 使用softmax的多分类器，返回预测的类别(不是概率)。需要设置num_class(类别数目)。
        # multi:softprob 和multi:softmax参数一样，但是返回的是每个数据属于各个类别的概率。
        # 'num_class': 0,                   # 类别数目，和objective：multi一起使用
        'eval_metric': ['auc'],             # 对于有效数据的度量（评价）方法。
        # 对于回归问题，默认值是rmse，对于分类问题，默认值是error。
        # 典型值有：
        # rmse 均方根误差(∑Ni=1ϵ2N−−−−−√)
        # mae 平均绝对误差(∑Ni=1|ϵ|N)
        # logloss 负对数似然函数值
        # error 二分类错误率(阈值为0.5)
        # merror 多分类错误率
        # mlogloss 多分类logloss损失函数
        # auc
        # 曲线下面积
        'seed': 0,                          # 随机数的种子，设置它可以复现随机数据的结果，也可以用于调整参数
        'scale_pos_weight': 3,              # 在各类别样本十分不平衡时，把这个参数设定为一个正值，可以使算法更快收敛。default=1
        # 'early_stopping_rounds': 3        # 用于控制在 Out Of Sample 的验证集上连续多少个迭代的分数都没有提高后就提前终止训练。用于防止 Overfitting.
    }

    if params['objective'] != 'binary:logistic':
        params['num_class'] = 2
        params['eval_metric'] = ['merror']

    num_round = 50
    evallist = [(dtrain, 'train'), (dtest, 'eval'), (dverify, 'eval')]

    if model:
        params = {
            'objective': 'binary:logistic',
            'nthread': 4
        }
        bst = xgb.Booster(params)
        bst.load_model(model)
    else:
        bst = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=num_round,
            evals=evallist
        )
    return bst


def booster_verify(dtrain, y_train, dtest, y_test, dverify, y_verify, fea_dict={}):
    for name, data, label in [
            ('训练集', dtrain, y_train),
            ('测试集', dtest, y_test),
            ('验证集', dverify, y_verify),
    ]:
        y_pred = bst.predict(data)
        aucpath, aucks = calc_auc(
            label, y_pred, 'test', save_as='auc_%s' % name)
        print(aucks)

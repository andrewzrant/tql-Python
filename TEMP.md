```https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python
# Class to extend the Sklearn classifier
class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)
    
    def fit(self,x,y):
        return self.clf.fit(x,y)
    
    def feature_importances(self,x,y):
        print(self.clf.fit(x,y).feature_importances_)
```


```
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm


class Stacking(object):
    """二分类"""

    def __init__(self, clf, cv=3, seed=None):
        self._clf = clf
        self.seed = seed  # 不同模型的种子（是否必要？）
        self.cv = cv
        self.clfs = []
        self.clf_by_all = None

    def fit(self, X, y):
        flods = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=666).split(X, y)
        _fit = partial(self._fit_and_score, clf=self._clf, X=X, y=y)

        with ProcessPoolExecutor(self.cv) as pool:
            _ = pool.map(_fit, tqdm(flods, self._clf.__repr__().split('(')[0]))

        preds = []
        scores = []
        for clf, pred, score in _:
            self.clfs.append(clf)
            preds.append(pred)
            scores.append(score)
        print('Score CV : %.4f +/- %.4f' % (np.mean(scores), np.std(scores)))

        preds = np.row_stack(preds)
        oof = preds[np.argsort(preds[:, 0])][:, -1] # 按照X原顺序排序并去掉索引
        print('Score OOF : %.4f' % roc_auc_score(y, oof))

        del scores, preds, oof

    def _fit_and_score(self, iterable, clf, X, y):
        train_index, test_index = iterable
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        #################自定义最优模型###################
        clf.fit(X_train, y_train)
        test_pred = np.column_stack((test_index, clf.predict_proba(X_test)))
        score = roc_auc_score(y_test, test_pred[:, -1])
        ###############################################
        return clf, test_pred, score

    def predict_meta_features(self, X):
        _ = np.column_stack([clf.predict_proba(X)[:, -1] for clf in self.clfs]).mean(1)
        return _
```

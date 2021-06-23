import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

from sklearn.pipeline import Pipeline, FeatureUnion
from imblearn.pipeline import Pipeline as Pipeline_im

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer, average_precision_score, roc_auc_score, precision_score, recall_score, precision_recall_curve, f1_score

from evt_detect.models.trfs import trf_length, trf_pos, trf_sentiment


def model_prep(classifier, scaler=None, fselector=None, oversampler=None, undersampler=None):
    vect = Pipeline([
        ('count', CountVectorizer()),
        ('tfidf', TfidfTransformer())
    ])

    features = FeatureUnion([
        ('vect', vect),
        ('length', trf_length()),
        ('pos', trf_pos()),
        ('sentiment', trf_sentiment())
    ])

    steps = [('features', features)]

    if scaler is not None:
        steps.append(('scaler', scaler()))
    
    if fselector is not None:
        steps.append(('fselector', fselector()))

    if oversampler is not None:
        steps.append(('oversampler', oversampler()))

    if undersampler is not None:
        steps.append(('undersampler', undersampler()))

    steps.append(('classifier', classifier()))

    return Pipeline_im(steps)


class model_eval():
    def __init__(self, data, y_col, x_col='sents'):
        self.data = data
        self.x_col = x_col
        self.y_col = y_col
        self.scores = {
            'pr_auc': 'average_precision',
            'roc_auc': 'roc_auc',
            'precision': 'precision',
            'recall': 'recall'
        }
        self.model_sum = {}
        self.model_compare = []

    def gen_train_test_set(self, test_size=0.2):
        X = self.data[self.x_col]
        y = self.data[self.y_col]
        y.fillna(0, inplace=True)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X,y,test_size=test_size,stratify=y, random_state=2021)
        return self


    def model_tuning(self, model, params, refit_score='pr_auc'):
        model_spec = dict((name, type(step).__name__) for name, step in model.steps[1:])

        param_elgible = model.get_params().keys()
        params = {k:v for k, v in params.items() if k in param_elgible}

        gs = GridSearchCV(model, params, n_jobs=-1, scoring=self.scores, cv=3, return_train_score=True, refit=refit_score)
        gs.fit(self.X_train, self.y_train)

        self.model = gs.best_estimator_
        if self.model_sum:
            self.model_compare.append(self.model_sum)

        self.model_sum = {**model_spec, **gs.best_params_}
        self.model_sum[f'valid_{refit_score}'] = gs.best_score_

        return gs


    def find_best_threshold(self):
        try:
            y_score = self.model.predict_proba(self.X_train)[:,1]
        except:
            y_score = self.model.decision_function(self.X_train)[:,1]

        precisions, recalls, thresholds = precision_recall_curve(self.y_train, y_score)
        f1s = [2*p*r/(p+r) for p, r in zip(precisions, recalls)]
        idx = np.argmax(f1s)
        self.threshold = thresholds[idx]
        self.model_sum['threshold'] = self.threshold

        self.pr_curve = self.plot_best_threshold(precisions, recalls, idx)

        return self

    def plot_best_threshold(self, precisions, recalls, best_idx):
        no_skill = len(self.y_train[self.y_train==1]) / len(self.y_train)
        plt.plot([0,1], [no_skill,no_skill], linestyle='--', label='No Skill')
        plt.plot(recalls, precisions, marker='.')
        plt.scatter(recalls[best_idx], precisions[best_idx], marker='o', color='black', label='Best')
        # axis labels
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend()
        plt.show()
        return plt.gcf()

    def model_predict(self):
        try:
            y_train_score = self.model.predict_proba(self.X_train)[:,1]
            y_test_score = self.model.predict_proba(self.X_test)[:,1]
        except:
            y_train_score = self.model.decision_function(self.X_train)[:,1]
            y_test_score = self.model.decision_function(self.X_test)[:,1]
        
        self.y_train_pred = (y_train_score > self.threshold).astype(int)
        self.y_test_pred = (y_test_score > self.threshold).astype(int)

        model_scores = {
            'train_pr_auc': average_precision_score(self.y_train, y_train_score),
            'test_pr_auc': average_precision_score(self.y_test, y_test_score),
            'train_roc_auc': roc_auc_score(self.y_train, y_train_score),
            'test_roc_auc': roc_auc_score(self.y_test, y_test_score)
        }

        self.model_sum = {**self.model_sum, **model_scores}

        return self

    def predict_error(self):
        err_train = np.not_equal(self.y_train_pred, self.y_train)
        err_test = np.not_equal(self.y_test_pred, self.y_test)

        errs = {}
        errs['train'] = self.X_train[:, err_train]
        errs['test'] = self.X_test[:, err_test]

        return pd.DataFrame.from_dict(errs, orient='index').transpose()

    def model_val(self):
        model_scores = {
            'train_precision': precision_score(self.y_train, self.y_train_pred),
            'test_precision': precision_score(self.y_test, self.y_test_pred),
            'train_recall': recall_score(self.y_train, self.y_train_pred),
            'test_recall': recall_score(self.y_test, self.y_test_pred),
            'train_f1': f1_score(self.y_train, self.y_train_pred),
            'test_f1': f1_score(self.y_test, self.y_test_pred)
        }
        self.model_sum = {**self.model_sum, **model_scores}
        return self

    def model_save(self, save_file):
        joblib.dump(self.model, save_file)

    def models_summary(self):
        return pd.DataFrame(self.model_compare)


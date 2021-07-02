import logging
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.semi_supervised import SelfTrainingClassifier

from sklearn.pipeline import Pipeline, FeatureUnion
from imblearn.pipeline import Pipeline as Pipeline_im

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, precision_recall_curve, f1_score, roc_curve

from evt_detect.models.trfs import trf_length, trf_pos, trf_sentiment
import evt_detect.features.nlp_features as nlp_feat

logger = logging.getLogger(__name__)

def model_prep(classifier, vect_scaler=None, lsa=None, scaler=None, fselector=None, oversampler=None, undersampler=None):
    vect_steps = [
        ('count', CountVectorizer()),
        ('tfidf', TfidfTransformer())
    ]

    if vect_scaler is not None:
        vect_steps.append(('scaler', vect_scaler()))
    
    if lsa is not None:
        vect_steps.append(('lsa', lsa()))

    vect = Pipeline(vect_steps)

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
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.threshold = 0.5
        self.pr_curve = None
        self.y_train_pred = None
        self.y_test_pred = None
        self.refit_score = 'roc_auc'

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

        logger.info('GridSearch Finished')
        self.model = gs.best_estimator_

        self.model_sum = {**model_spec, **gs.best_params_}
        self.model_sum[f'valid_{refit_score}'] = gs.best_score_
        self.refit_score = refit_score

        return gs


    def find_best_threshold(self):
        try:
            y_score = self.model.predict_proba(self.X_train)[:,1]
        except:
            y_score = self.model.decision_function(self.X_train)

        if self.refit_score == 'pr_auc':
            precisions, recalls, thresholds = precision_recall_curve(self.y_train, y_score)
            f1s = [2*p*r/(p+r) for p, r in zip(precisions, recalls)]
            idx = np.argmax(f1s)

            self.threshold_curve = self.plot_best_threshold(recalls, precisions, idx)
            logger.info('PR Curve created')

        elif self.refit_score == 'roc_auc':
            fpr, tpr, thresholds = roc_curve(self.y_train, y_score)
            g_means = np.sqrt(tpr * (1-fpr))
            idx = np.argmax(g_means)

            self.threshold_curve = self.plot_best_threshold(fpr, tpr, idx)
            logger.info('ROC created')

        self.threshold = thresholds[idx]
        self.model_sum['threshold'] = self.threshold
        logger.info('best threshold found')

        return self

    def plot_best_threshold(self, x_values, y_values, best_idx):
        fig = plt.figure(figsize=(6, 4))
        if self.refit_score == 'pr_auc':
            no_skill = len(self.y_train[self.y_train==1]) / len(self.y_train)
            ax = fig.add_subplot()
            ax.plot([0,1], [no_skill,no_skill], linestyle='--', label='No Skill')
            ax.plot(x_values, y_values, marker='.')
            ax.scatter(x_values[best_idx], y_values[best_idx], marker='o', color='black', label='Best')
            # axis labels
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')

        elif self.refit_score == 'roc_auc':
            plt.plot([0,1], [0,1], linestyle='--', label='No Skill')
            plt.plot(x_values, y_values, marker='.')
            plt.scatter(x_values[best_idx], y_values[best_idx], marker='o', color='black', label='Best')
            # axis labels
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')

        plt.legend()
        return fig

    def model_predict(self):
        try:
            y_train_score = self.model.predict_proba(self.X_train)[:,1]
            y_test_score = self.model.predict_proba(self.X_test)[:,1]
        except:
            logger.exception('Cannot call predict_proba')
            
            try:
                y_train_score = self.model.decision_function(self.X_train)
                y_test_score = self.model.decision_function(self.X_test)
            except:
                logger.exception('Cannot call decision_function')
                
        try:
            self.y_train_pred = (y_train_score > self.threshold).astype(int)
            self.y_test_pred = (y_test_score > self.threshold).astype(int)
        except:
            logger.exception('Cannot use threshold to generate prediction')
            self.y_train_pred = self.model.predict(self.X_train)
            self.y_test_pred = self.model.predict(self.X_test)

        logger.info('Model prediction generated')
        
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

        errs = self.X_train[err_train].tolist()
        errs += self.X_test[err_test].tolist()

        trues = self.y_train[err_train].tolist()
        trues += self.y_test[err_test].tolist()

        preds = self.y_train_pred[err_train].tolist()
        preds += self.y_test_pred[err_test].tolist()

        return pd.DataFrame(zip(errs, trues, preds), columns=[self.x_col, self.y_col, f'{self.y_col}_pred'])

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


    def model_fin(self):
        self.model_compare.append(self.model_sum)

    def models_summary(self):
        return pd.DataFrame(self.model_compare)

def model_pred(X, model, threshold, tokenizer):
    while len(X) > 0:
        try:
            y_proba = model.predict_proba(X)[:,1]
        except ValueError:
            logger.exception('The input paragraph is problematic')
            logger.error(X)
            X = [text for text in X if len(tokenizer(text)) > 0]
        except:
            logger.exception('Cannot call predict_proba')
            try:
                y_pred = model.predict(X)
            except ValueError:
                logger.exception('The input paragraph is problematic')
                logger.error(X)
                X = [text for text in X if len(tokenizer(text)) > 0]
            except:
                logger.exception('Cannot call predict')
                return []
        else:
            y_pred = (y_proba > threshold).astype(int)
            return [(x, proba) for (x, pred, proba) in zip(X, y_pred, y_proba) if pred == 1]
    else:
        return []


def df_model_pred(df, X_col, y_col, model_name, output='whole', **kwargs):
    df.reset_index(drop=True, inplace=True)
    sents = df[X_col].map(nlp_feat.gen_sents)
    pos_all = sents.apply(model_pred, **kwargs)

    if output == 'sent':
        result = pos_all.explode(ignore_index=True).dropna().drop_duplicates()
        result = pd.DataFrame(result.tolist(), columns=['sents', f'{y_col}_{model_name}_proba'])
        logger.info(f'fount {result.shape[0]} sentences with {y_col}')

    elif output == 'whole':
        pos_sents = pos_all.map(lambda t: list(zip(*t))[0] if len(t) > 0 else None)
        pos_proba = pos_all.map(lambda t: list(zip(*t))[1] if len(t) > 0 else None)
        df[f'{y_col}_{model_name}_sents_num'] = pos_all.map(len)
        df[f'{y_col}_{model_name}_sents'] = pos_sents.map(lambda s: " \n ".join(s), na_action='ignore')
        df[f'{y_col}_{model_name}_proba'] = pos_proba.map(np.mean, na_action='ignore')
        df[f'{y_col}_{model_name}_pred'] = pos_all.map(lambda s: 1 if len(s) > 0 else 0)
    
        result = df.loc[df[f'{y_col}_{model_name}_pred'] == 1]
        logger.info(f'found {result.shape[0]} filing out of {df.shape[0]} with {y_col}')

    try:
        return result
    except:
        logger.exception('Cannot generate prediction for DataFrame')
        return pd.DataFrame()

class semi_training(model_eval):
    def __init__(self, labeled, unlabeled, y_col, x_col='sents'):
        super().__init__(labeled, y_col, x_col)
        self.unlabeled = unlabeled

    def prepare_unlabeled_set(self):
        X = self.unlabeled[self.x_col]
        y = self.unlabeled[self.y_col]
        y.fillna(-1, inplace=True)

        self.X_all = pd.concat([self.X_train, X], ignore_index=True)
        self.y_all = pd.concat([self.y_train, y], ignore_index=True)

        return self

    def self_training(self, model, threshold=0.95):
        self.model = model[:-1]
        self.model.steps.append(('clf', SelfTrainingClassifier(model[-1], threshold=threshold)))
        self.model.fit(self.X_all, self.y_all)

        self.threshold = threshold
        self.df = pd.DataFrame({
            self.x_col: self.X_all,
            self.y_col: self.y_all,
            'pseudo_label': self.model[-1].transduction_
        })
        logger.info(f'The number of rounds of self-training is {self.model[-1].n_iter_}')

        return self

    def self_training_result(self):
        results = self.df.loc[(self.df[self.y_col] != -1) | (self.df['pseudo_label'] != -1), [self.x_col, 'pseudo_label']]
        results.rename(columns={'pseudo_label': self.y_col}, inplace=True)
        return results

    def self_training_check(self):
        return self.df.loc[self.df[self.y_col] != self.df['pseudo_label']]
        
    def self_training_noresult(self):
        noresult = self.df.loc[self.df['pseudo_label'] == -1]
        probas = self.model.predict_proba(noresult[self.x_col])
        noresult['proba_neg'] = probas[:,0]
        noresult['proba_pos'] = probas[:,1]

        return noresult
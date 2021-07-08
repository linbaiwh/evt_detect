import logging
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.semi_supervised import SelfTrainingClassifier

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
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

    features = ColumnTransformer(
        [
            ('vect', vect, 'tokens'),
            ('length', trf_length(), 'tokens'),
            ('sentiment', trf_sentiment(), 'sents')
        ],
        remainder='passthrough', n_jobs=-1
    )

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
    def __init__(self, data, y_col, X_col):
        self.data = data
        self.X_col = X_col
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
        X = self.data[self.X_col]
        y = self.data[self.y_col].fillna(0)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y,test_size=test_size,stratify=y, random_state=2021)
        return self

    def params_prepare(self, model, params):
        param_elgible = model.get_params().keys()
        return {k:v for k, v in params.items() if k in param_elgible}

    def model_tuning(self, model, params, refit_score='pr_auc'):
        model_spec = dict((name, type(step).__name__) for name, step in model.steps[1:])

        params = self.params_prepare(model, params)

        gs = GridSearchCV(model, params, n_jobs=-1, scoring=self.scores, cv=3, return_train_score=True, refit=refit_score)
        gs.fit(self.X_train, self.y_train)

        logger.info('GridSearch Finished')
        self.model = gs.best_estimator_

        self.model_sum = {**model_spec, **gs.best_params_}
        self.model_sum[f'valid_{refit_score}'] = gs.best_score_
        self.refit_score = refit_score

        return gs

    def model_fit(self, model, params):
        params = self.params_prepare(model, params)
        model.set_params(params)
        model.fit(self.X_train, self.y_train)
        logger.info('model fit successfully')
        self.model = model
        return self


    def find_best_threshold(self, use_test=False):
        if use_test:
            y_true = self.y_test
            X = self.X_test
        else:
            y_true = self.y_train
            X = self.X_train

        try:
            y_score = self.model.predict_proba(X)[:,1]
        except:
            y_score = self.model.decision_function(X)

        if self.refit_score == 'pr_auc':
            precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
            f1s = [2*p*r/(p+r) for p, r in zip(precisions, recalls)]
            idx = np.argmax(f1s)

            self.threshold_curve = self.plot_best_threshold(recalls, precisions, idx)
            logger.info('PR Curve created')

        elif self.refit_score == 'roc_auc':
            fpr, tpr, thresholds = roc_curve(y_true, y_score)
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

    def model_probas(self, X):
        try:
            return self.model.predict_proba(X)[:,1]
        except:
            logger.exception('Cannot call predict_proba')

            try:
                return self.model.predict(X)
            except:
                logger.exception('Cannot call predict')
                return None

    def model_predict(self, probas):
        if probas is not None:
            return (probas > self.threshold).astype(int)
        logger.error('Cannot generate generate model prediction')
        return None


    def train_test_predict(self):
        self.y_train_proba = self.model_probas(self.X_train)
        self.y_test_proba = self.model_probas(self.X_test)
                
        self.y_train_pred = self.model_predict(self.y_train_proba)
        self.y_test_pred = self.model_predict(self.y_test_proba)

        return self
        

    def predict_error(self):
        err_train = np.not_equal(self.y_train_pred, self.y_train)
        err_test = np.not_equal(self.y_test_pred, self.y_test)

        errs_train = self.X_train.loc[err_train]
        errs_test = self.X_test.loc[err_test]
        err_df = pd.concat([errs_train, errs_test])
        try:
            trues = self.y_train[err_train].tolist()
            trues += self.y_test[err_test].tolist()
        except ValueError:
            trues = self.y_train.loc[err_train].tolist()
            trues += self.y_test.loc[err_test].tolist()

        preds = self.y_train_pred[err_train].tolist()
        preds += self.y_test_pred[err_test].tolist()

        probas = self.y_train_proba[err_train].tolist()
        probas += self.y_test_proba[err_test].tolist()

        err_df[self.y_col] = trues
        err_df[f'{self.y_col}_pred'] = preds
        err_df[f'{self.y_col}_proba'] = probas

        return err_df

    def model_scores(self):
        scores = {
            'train_pr_auc': average_precision_score(self.y_train, self.y_train_proba),
            'test_pr_auc': average_precision_score(self.y_test, self.y_test_proba),
            'train_roc_auc': roc_auc_score(self.y_train, self.y_train_proba),
            'test_roc_auc': roc_auc_score(self.y_test, self.y_test_proba),

            'train_precision': precision_score(self.y_train, self.y_train_pred),
            'test_precision': precision_score(self.y_test, self.y_test_pred),
            'train_recall': recall_score(self.y_train, self.y_train_pred),
            'test_recall': recall_score(self.y_test, self.y_test_pred),
            'train_f1': f1_score(self.y_train, self.y_train_pred),
            'test_f1': f1_score(self.y_test, self.y_test_pred)
        }

        self.model_sum = {**self.model_sum, **scores}
        return self

    def model_save(self, save_file):
        joblib.dump(self.model, save_file)


    def model_fin(self):
        self.model_compare.append(self.model_sum)

    def models_summary(self):
        return pd.DataFrame(self.model_compare)


class semi_training(model_eval):
    def __init__(self, labeled, unlabeled, y_col, X_col):
        super().__init__(labeled, y_col, X_col)
        self.unlabeled = unlabeled

    def prepare_unlabeled_set(self):
        X = self.unlabeled[self.X_col]
        y = self.unlabeled[self.y_col]
        y.fillna(-1, inplace=True)

        self.X_train = pd.concat([self.X_train, X], ignore_index=True)
        self.y_train = pd.concat([self.y_train, y], ignore_index=True)

        return self

    def self_training(self, model, threshold=0.95):
        self.model = model[:-1]
        self.model.steps.append(('clf', SelfTrainingClassifier(model[-1], threshold=threshold)))
        self.model.fit(self.X_train, self.y_train)

        logger.info(f'The number of rounds of self-training is {self.model[-1].n_iter_}')

        return self

    def self_training_result(self):
        X_all = pd.concat([self.X_train, self.X_test], ignore_index=True)
        y_all = pd.concat([self.y_train, self.y_test], ignore_index=True)
        pseudo_label = np.concatenate((self.model[-1].transduction_, self.y_test))
        probas = self.model_probas(X_all)

        self.df = pd.DataFrame(X_all, columns=self.X_col)
        self.df[self.y_col] = y_all
        self.df['pseudo_label'] = pseudo_label
        self.df[f'{self.y_col}_proba'] = probas

        return self

    def self_training_labeled(self):
        results = self.df.loc[(self.df[self.y_col] != -1) | (self.df['pseudo_label'] != -1)]
        results.rename(columns={self.y_col: 'true_label', 'pseudo_label': self.y_col}, inplace=True)
        return results

    def self_training_chg(self):
        results = self.df.loc[self.df[self.y_col] != self.df['pseudo_label']]
        results.rename(columns={self.y_col: 'true_label', 'pseudo_label': self.y_col}, inplace=True)
        return results
        
    def self_training_nolabel(self):
        nolabel = self.df.loc[self.df['pseudo_label'] == -1]
        return nolabel



def parag_pred(df, textcol, tokenizer, y_col, model, threshold, output='whole'):
    sents_dfs = df[textcol].apply(nlp_feat.parag_to_sents, tokenizer=tokenizer).tolist()
    if len(sents_dfs) > 0:
        X_col = sents_dfs[0].columns.tolist()

    sents_dfs = [sents_df.assign(idx=idx) for idx, sents_df in zip(df.index, sents_dfs)]

    sents = pd.concat(sents_dfs, ignore_index=True)

    sents_eval = model_eval(sents, y_col, X_col)
    sents_eval.model = model
    sents_eval.threshold = threshold
    
    X = sents_eval.data[sents_eval.X_col]
    try:
        sents['proba'] = sents_eval.model_probas(X)
    except:
        logger.exception('Cannot predict')
        return pd.DataFrame()

    sents['pred'] = sents_eval.model_predict(sents['proba'])

    pos_sents = sents.loc[sents['pred'] == 1]
    
    pos_parags = pd.DataFrame()
    pos_parags[f'{y_col}_sents'] = pos_sents.groupby('idx')['sents'].apply('\n'.join)
    pos_parags[f'{y_col}_sents_num'] = pos_sents.groupby('idx')['sents'].count()
    pos_parags[f'{y_col}_proba'] = pos_sents.groupby('idx')['proba'].mean()

    logger.info(f'found {pos_parags.shape[0]} filing out of {df.shape[0]} with {y_col}')

    if output == 'whole':
        return df.join(pos_parags, how='inner')
    elif output == 'pos_sents':
        return pos_sents
    elif output == 'all_sents':
        return sents

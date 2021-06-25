#%%
import sys
import os
from pathlib import Path
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd

import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler, MaxAbsScaler, Normalizer
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.feature_selection import SelectKBest, f_classif, chi2, mutual_info_classif

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

from evt_detect.utils.file_io import read_file_df, to_file_df, merge_csv
from evt_detect.models.model_build import model_prep, model_eval
from evt_detect.features import nlp_features as nlp_feat
from evt_detect.utils.visualize import plot_search_results

# * Models
models = [
    {
        'classifier': LogisticRegression,
        'scaler': Normalizer,
        'fselector': NMF
    }, # * Baseline model
    {
        'classifier': XGBClassifier,
        'scaler': Normalizer,
        'fselector': NMF
    }, # * Tree-based model
    {
        'classifier': SVC,
        'scaler': Normalizer,
        'fselector': NMF
    } # * SVC
]

# * Classifier Hyperparameters
clf_params = [
    {
        'classifier__C': [0.4],  
        'classifier__penalty': ['l2'],  
        'classifier__solver': ['lbfgs'],  
        'classifier__class_weight': ['balanced']  
    }, # * Baseline model
    {
        'classifier__n_estimators': [300],  
        'classifier__max_depth': [5],  
        'classifier__min_child_weight': [1],  
        'classifier__gamma': [0],  
        'classifier__subsample': [0.8],  
        'classifier__colsample_bytree': [0.8],  
        'classifier__scale_pos_weight': [1] 
    }, # * Tree-based model
    {
        'classifier__kernel': ['rbf'],  
        'classifier__C': [4.5],  
        'classifier__gamma': ['scale'],  
        'classifier__probability': [True],  
        'classifier__class_weight': ['balanced']  
    } # * SVC
]

model_names = [
    'Baseline',
    'Tree',
    'SVC'
]

def main(form_label, y_col='Incident'):

    # * File path
    data_folder = Path(__file__).resolve().parents[2] / 'data'
    label_folder = data_folder / 'label'
    compare_folder = data_folder / 'compare'
    model_folder = data_folder / 'model'
    tag = 'breach'

    # * Feature Hyperparameters
    if form_label == 'CR':
        tokenizer = nlp_feat.CR_tokenizer

    elif form_label == 'PR':
        tokenizer = nlp_feat.PR_tokenizer

    feat_params = {
        'features__vect__count__lowercase': [False],
        'features__vect__count__tokenizer': [tokenizer],
        'features__vect__count__ngram_range': [(1,2), (1,3)],
        'features__vect__count__max_df': [0.7, 0.9, 1.0],
        'features__vect__count__min_df': [1, 2, 4],
        'features__vect__tfidf__use_idf': [True],
        'features__vect__tfidf__sublinear_tf': [True],
        'features__length__tokenizer': [tokenizer],
        
        'fselector__n_components': [100, 200, 400, 600],
        'fselector__alpha': [0.1],
        'fselector__l1_ratio': [0.5]
        

    }

    # * Prepare traing and test data
    sents_labeled = label_folder / f'{tag}_{form_label}_sents_labeled.xlsx'
    sents_labeled_1 = label_folder / f'{tag}_{form_label}_sents_labeled_1.xlsx'
    data = merge_csv([sents_labeled, sents_labeled_1])
    data.drop('cik', axis=1, inplace=True)
    data.fillna(0, inplace=True)
    data.drop_duplicates(inplace=True)

    data_train = model_eval(data, y_col, x_col='sents')
    data_train.gen_train_test_set()

    # * Model training
    for i in range(len(models)):
        print(f'start training {model_names[i]}')
        model_spec = models[i]
        params = {**feat_params, **clf_params[i]}
        
        pipe = model_prep(**model_spec)
        gs = data_train.model_tuning(pipe, params)

        data_train.find_best_threshold()
        data_train.model_predict()
        data_train.model_val()
        data_train.model_fin()

        # * Save model specific results

        gsplot = plot_search_results(gs)
        gsplot.savefig(compare_folder / f'{form_label}_{y_col}_{model_names[i]}_gsplot.png')


        data_train.pr_curve.savefig(compare_folder / f'{form_label}_{y_col}_{model_names[i]}_pr_curve.png')

        err_pred = data_train.predict_error()
        to_file_df(err_pred, compare_folder / f'{form_label}_{y_col}_{model_names[i]}_pred_err.xlsx')


        data_train.model_save(model_folder / f'{form_label}_{y_col}_{model_names[i]}.joblib')
        print(f'{model_names[i]} saved')
    # * Save results for all models
    models_df = data_train.models_summary()
    models_comp_file = compare_folder / f'{form_label}_{y_col}_compare.xlsx'

    if models_comp_file.exists():
        models_pre = read_file_df(models_comp_file)
        models_df = pd.concat([models_df, models_pre], ignore_index=True)

    to_file_df(models_df, models_comp_file)


if __name__ == "__main__":
    main('CR', 'Incident')
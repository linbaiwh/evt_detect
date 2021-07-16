import logging
import logging.config
import pandas as pd
import warnings
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, Normalizer, RobustScaler
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
from sklearn.feature_selection import SelectKBest, f_classif, chi2, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC

from steps_context import tag, label_folder, model_folder, compare_folder, logger_conf
from evt_detect.utils.file_io import read_file_df, to_file_df, merge_csv
from evt_detect.models.model_build import model_prep, model_eval
from evt_detect.features import nlp_features as nlp_feat
from evt_detect.utils.visualize import plot_search_results
from evt_detect.utils.preprocess import rm_features

warnings.filterwarnings("ignore")

# * Models
models = [
    {
        'classifier': LogisticRegression,
        'scaler': MaxAbsScaler,
        'fselector': TruncatedSVD
    }, # * Baseline model
    {
        'classifier': LogisticRegression,
        'scaler': RobustScaler,
        'fselector': TruncatedSVD
    }, # * Baseline model with Robust scaler
    {
        'classifier': LogisticRegression,
        'scaler': StandardScaler,
        'fselector': TruncatedSVD
    }, # * Baseline model with Standard scaler
    # {
    #     'classifier': XGBClassifier,
    #     'vect_scaler': Normalizer,
    #     'lsa': NMF,
    #     'scaler': MaxAbsScaler
    # }, # * Tree-based model
    {
        'classifier': SVC,
        'scaler': MaxAbsScaler,
        'fselector': TruncatedSVD
    } # * SVC
]

# * Classifier Hyperparameters
clf_params = [
    {
        'classifier__C': [0.1, 0.5, 1, 2, 4, 10],  
        'classifier__penalty': ['l2'],  
        'classifier__solver': ['lbfgs'],  
        'classifier__class_weight': ['balanced'],

        'fselector__n_components': [600, 800, 1200]    
    }, # * Baseline model
    {
        'classifier__C': [0.1, 0.5, 1, 2, 4, 10],  
        'classifier__penalty': ['l2'],  
        'classifier__solver': ['lbfgs'],  
        'classifier__class_weight': ['balanced'],

        'scaler__with_centering': [False],

        'fselector__n_components': [600, 800, 1200]    
    }, # * Baseline model with Robust scaler
    {
        'classifier__C': [0.1, 0.5, 1, 2, 4, 10],  
        'classifier__penalty': ['l2'],  
        'classifier__solver': ['lbfgs'],  
        'classifier__class_weight': ['balanced'],

        'scaler__with_mean': [False],

        'fselector__n_components': [600, 800, 1200]    
    }, # * Baseline model with Standard scaler
    # {
    #     'classifier__n_estimators': [300],  
    #     'classifier__max_depth': [5],  
    #     'classifier__min_child_weight': [1],  
    #     'classifier__gamma': [0],  
    #     'classifier__subsample': [0.8],  
    #     'classifier__colsample_bytree': [0.8],  
    #     'classifier__scale_pos_weight': [1], 
    #     'classifier__use_label_encoder=False': [False], 

    #     'features__vect__count__min_df': [4],

    #     'features__vect__count__max_features': [1000]
    # }, # * Tree-based model
    {
        'classifier__kernel': ['rbf'],  
        'classifier__C': [0.1, 0.5, 1, 4, 10],  
        'classifier__gamma': ['scale'],  
        'classifier__probability': [True],  
        'classifier__class_weight': ['balanced'],
        
        # 'features__vect__count__max_features': [1200],

        'fselector__n_components': [600, 800, 1200],
        # 'fselector__learning_method': ['online'] 


        # 'features__vect__lsa__init': ['nndsvd'],
        # 'features__vect__lsa__alpha': [0.01, 0.1, 1, 4, 10],
        # 'features__vect__lsa__l1_ratio': [0, 0.5, 1],
        # 'features__vect__lsa__max_iter': [10000]
    } # * SVC
]

model_names = [
    'Baseline',
    'Baseline_Robust',
    'Baseline_Std',
    # 'Tree',
    'SVC'
]

def main(form_label, y_col='Incident', propagation=False):

    logging.config.fileConfig(logger_conf)
    logger = logging.getLogger('model_training')

    logger.info(f'Model training for {form_label} {y_col}')

    if form_label == 'CR':
        stop_words_lower = nlp_feat.CR_stopwords_lower
        stop_words_nolower = nlp_feat.CR_stopwords_nolower
    elif form_label == 'PR':
        stop_words_lower = nlp_feat.PR_Related_stopwords_lower
        stop_words_nolower = nlp_feat.PR_Related_stopwords_nolower


    # * Feature Hyperparameters
    feat_params_grid = [
        {
        'features__vect__count__lowercase': [True],
        'features__vect__count__ngram_range': [(1,2)],
        'features__vect__count__max_df': [0.7, 0.8],
        'features__vect__count__min_df': [2],
        'features__vect__count__stop_words': [stop_words_lower],
        'features__vect__tfidf__use_idf': [True],
        'features__vect__tfidf__sublinear_tf': [True, False] 
        },
        {
        'features__vect__count__lowercase': [False],
        'features__vect__count__ngram_range': [(1,2)],
        'features__vect__count__max_df': [0.7, 0.8],
        'features__vect__count__min_df': [2],
        'features__vect__count__stop_words': [stop_words_nolower],
        'features__vect__tfidf__use_idf': [True],
        'features__vect__tfidf__sublinear_tf': [True, False] 
        },
    ]

    # * Prepare traing and test data
    if propagation == False:
        sents_labeled = label_folder / f'{tag}_{form_label}_sents_labeled.xlsx' 
        rm_cols = ['Related', 'Incident', 'Immaterial', 'Cost', 'Litigation', 'Management']
    else:
        sents_labeled = label_folder / f'{form_label}_{y_col}_labeled.xlsx'
        rm_cols = ['true_label', y_col, f'{y_col}_proba']
    
    data = read_file_df(sents_labeled)
    data.drop_duplicates(inplace=True)
    features = rm_features(data, rm_cols)

    data_train = model_eval(data, y_col, X_col=features)
    data_train.gen_train_test_set()

    # * Model training
    for i in range(len(models)):
        logger.info(f'start training {model_names[i]}')
        model_spec = models[i]
        params_grid = [{**feat_params, **clf_params[i]} for feat_params in feat_params_grid]
        
        pipe = model_prep(**model_spec)
        gs = data_train.model_tuning(pipe, params_grid, refit_score='roc_auc')

        data_train.find_best_threshold()
        data_train.train_test_predict()
        data_train.model_scores()
        data_train.model_sum['model_name'] = model_names[i]

        data_train.model_fin()

        logger.info(f'{model_names[i]} finished')
        logger.info(f'{data_train.model_sum}')

        # * Save model specific results
        try:
            data_train.model_save(model_folder / f'{form_label}_{y_col}_{model_names[i]}.joblib')
        except Exception:
            logger.exception('Cannot save best model')
        else:
            logger.info(f'{model_names[i]} saved')

        try:
            gsplot = plot_search_results(gs)
            gsplot.savefig(compare_folder / f'{form_label}_{y_col}_{model_names[i]}_gsplot.png')
        except Exception:
            logger.exception('Cannot save grid search results')
        else:
            logger.info('Grid search results saved')

        try:
            data_train.threshold_curve.savefig(compare_folder / f'{form_label}_{y_col}_{model_names[i]}_threshold_curve.png')
        except Exception:
            logger.exception('Cannot save threshold Curve')
        else:
            logger.info('threshold Curve saved')
        
        try:
            err_pred = data_train.predict_error()
        except Exception:
            logger.exception('Cannot create predict error')
        else:
            to_file_df(err_pred, compare_folder / f'{form_label}_{y_col}_{model_names[i]}_pred_err.xlsx')
            logger.info('Predict error saved')

    
    # * Save results for all models
    models_df = data_train.models_summary()
    models_comp_file = compare_folder / f'{form_label}_{y_col}_compare.xlsx'
    models_sum_file = model_folder / f'{form_label}_{y_col}_spec.xlsx'

    to_file_df(models_df, models_sum_file)

    if models_comp_file.exists():
        models_pre = read_file_df(models_comp_file)
        models_df = pd.concat([models_df, models_pre], ignore_index=True)

    to_file_df(models_df, models_comp_file)


if __name__ == "__main__":
    # main('CR', 'Incident')
    # main('CR', 'Incident', propagation=True)
    # main('CR', 'Related')
    # main('CR', 'Related', propagation=True)
    # main('PR', 'Incident')
    main('PR', 'Related')
    # main('PR', 'Incident', propagation=True)
    # main('PR', 'Immaterial')
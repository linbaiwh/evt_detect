#%%
import sys
import os
from pathlib import Path
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sklearn.preprocessing import StandardScaler, MaxAbsScaler, Normalizer, MinMaxScaler, QuantileTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import SelectKBest, f_classif, chi2, mutual_info_classif

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

from evt_detect.utils.file_io import read_file_df, to_file_df, merge_csv
from evt_detect.models.model_build import model_prep, model_eval
from evt_detect.features import nlp_features as nlp_feat
from evt_detect.utils.visualize import plot_search_results


#%%
data_folder = Path(__file__).resolve().parents[2] / 'data'
label_folder = data_folder / 'label'
compare_folder = data_folder / 'compare'
model_folder = data_folder / 'model'
tag = 'breach'

#%%
form_label = 'CR'
y_col = 'Incident'
tokenizer = nlp_feat.CR_tokenizer

#%%
form_label = 'PR'
y_col = 'Incident'
# y_col = 'Immaterial'
tokenizer = nlp_feat.PR_tokenizer

#%%
sents_notlabeled = label_folder / f'{tag}_{form_label}_sents.xlsx'
sents_labeled = label_folder / f'{tag}_{form_label}_sents_labeled.xlsx'
sents_labeled_1 = label_folder / f'{tag}_{form_label}_sents_labeled_1.xlsx'
sents_col = 'sents'
data = merge_csv([sents_labeled, sents_labeled_1])
data.drop('cik', axis=1, inplace=True)
data.fillna(0, inplace=True)

data_train = model_eval(data, y_col, x_col='sents')
data_train.gen_train_test_set()

# * Baseline model
#%%
model_spec = {
    'classifier': LogisticRegression,
    'scaler': MinMaxScaler,
    'fselector': TruncatedSVD
}

params = {
    'features__vect__count__lowercase': [False],
    'features__vect__count__tokenizer': [tokenizer],
    'features__vect__count__ngram_range': [(1,2), (1,3)],
    'features__vect__count__max_df': [0.9, 1.0],
    'features__vect__count__min_df': [2],
    'features__vect__tfidf__use_idf': [True, False],
    'features__length__tokenizer': [None, tokenizer],
    
    'fselector__n_components': [100, 300, 500, 800],

    'classifier__class_weight': ['balanced']  
}

# * Tree-based model
#%%
model_spec = {
    'classifier': RandomForestClassifier,
    'scaler': MinMaxScaler,
    'fselector': TruncatedSVD
}

params = {
    'features__vect__count__lowercase': [False],
    'features__vect__count__tokenizer': [tokenizer],
    'features__vect__count__ngram_range': [(1,2), (1,3)],
    'features__vect__count__max_df': [0.9],
    'features__vect__count__min_df': [2],
    'features__vect__tfidf__use_idf': [True, False],
    'features__length__tokenizer': [tokenizer],
    
    'fselector__n_components': [100, 300, 500, 800],

    'classifier__criterion': ['gini', 'entropy'],  
    'classifier__n_estimators': [100, 300, 500],  
    'classifier__max_features': ['sqrt', 'log2', 0.33, 0.5, 0.8],  
    'classifier__class_weight': ['balanced', 'balanced_subsample']  
}
# * SVC
#%%
model_spec = {
    'classifier': SVC,
    'scaler': MinMaxScaler,
    'fselector': TruncatedSVD
}

params = {
    'features__vect__count__lowercase': [False],
    'features__vect__count__tokenizer': [tokenizer],
    'features__vect__count__ngram_range': [(1,2), (1,3)],
    'features__vect__count__max_df': [0.9],
    'features__vect__count__min_df': [2],
    'features__vect__tfidf__use_idf': [True, False],
    'features__length__tokenizer': [tokenizer],
    
    'fselector__n_components': [100, 300, 500, 800],

    'classifier__kernel': ['rbf'],  
    'classifier__C': [0.1, 1, 5, 10],  
    'classifier__gamma': ['scale', 'auto'],  
    'classifier__class_weight': ['balanced']  
}

# * Model training
#%%
pipe = model_prep(**model_spec)
gs = data_train.model_tuning(pipe, params)

#%%
print(data_train.model_sum)

#%%
plot_search_results(gs)

#%%
data_train.find_best_threshold()

#%%
data_train.model_predict()
data_train.model_val()
print(data_train.model_sum)
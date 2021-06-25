#%%
import sys
import os
from pathlib import Path
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sklearn.preprocessing import Normalizer, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.decomposition import TruncatedSVD, NMF, LatentDirichletAllocation

#%%
import evt_detect

from evt_detect.utils.file_io import read_file_df, to_file_df, merge_csv
from evt_detect.utils.visualize import compare_features, plot_top_words
from evt_detect.features import nlp_features as nlp_feat

#%%
data_folder = Path(__file__).resolve().parents[2] / 'data'
label_folder = data_folder / 'label'
feature_folder = data_folder / 'feature'
tag = 'breach'

#%%
form_label = 'CR'
keys = ['Incident']
ents = nlp_feat.CR_ents
tokenizer = nlp_feat.CR_tokenizer

#%%
form_label = 'PR'
keys = ['Incident', 'Immaterial']
ents = nlp_feat.PR_ents
tokenizer = nlp_feat.PR_tokenizer

#%%
sents_notlabeled = label_folder / f'{tag}_{form_label}_sents.xlsx'
sents_labeled = label_folder / f'{tag}_{form_label}_sents_labeled.xlsx'
sents_labeled_1 = label_folder / f'{tag}_{form_label}_sents_labeled_1.xlsx'
sents_col = 'sents'
data = merge_csv([sents_labeled, sents_labeled_1])
data.drop('cik', axis=1, inplace=True)
data.fillna(0, inplace=True)
data.drop_duplicates(inplace=True)

#%%
# * Examine the named entities 
named_entities = nlp_feat.find_word_entity(data[sents_col])
entity_save = feature_folder / f'{tag}_{form_label}_named_entity.xlsx'
to_file_df(named_entities, entity_save)

#%%
# * Compare number of named entitie
compare_features(data, sents_col, keys, nlp_feat.entity_feature, ents=ents)

#%%
# * Compare most frequent words
countkwargs1 = {
    'lowercase': False,
    'tokenizer': tokenizer,
    'ngram_range': (1, 1),
}
top_100_1gram = nlp_feat.compare_top_n_words(data, sents_col, keys, n=100, **countkwargs1)
top_1gram_save = feature_folder / f'{tag}_{form_label}_top_1gram.xlsx'
to_file_df(top_100_1gram, top_1gram_save)

#%%
countkwargs2 = {
    'lowercase': False,
    'tokenizer': tokenizer,
    'ngram_range': (2, 2),
}
top_100_2gram = nlp_feat.compare_top_n_words(data, sents_col, keys, n=100, **countkwargs2)
top_2gram_save = feature_folder / f'{tag}_{form_label}_top_2gram.xlsx'
to_file_df(top_100_2gram, top_2gram_save)

#%%
countkwargs3 = {
    'lowercase': False,
    'tokenizer': tokenizer,
    'ngram_range': (3, 3),
}
top_100_3gram = nlp_feat.compare_top_n_words(data, sents_col, keys, n=100, **countkwargs3)
top_3gram_save = feature_folder / f'{tag}_{form_label}_top_3gram.xlsx'
to_file_df(top_100_3gram, top_3gram_save)

#%%
# * Compare length features
compare_features(data, sents_col, keys, nlp_feat.length_feature, tokenizer=tokenizer)

#%%
# * Compare parts-of-speech tags
compare_features(data, sents_col, keys, nlp_feat.pos_feature)

#%%
# * Compare sentiment features
compare_features(data, sents_col, keys, nlp_feat.sentiment_feature)


#%%
# * Examine top 20 words for the top 10 topics
# LSA
vect_params = {
    'lowercase': False,
    'tokenizer': tokenizer,
    'ngram_range': (1, 2),
    'max_df': 0.9,
    'min_df': 2,
    'max_features': 1000
}
svd_params = {
    'n_components': 10
}
svd, svd_words = nlp_feat.topics_lsa(
    data[sents_col], decompose=TruncatedSVD,
    scaler=Normalizer, tfidf=True,
    vect_params=vect_params, dc_params=svd_params
    )
plot_top_words(svd, svd_words, 20, 'Topics in LSA model')

#%%
nmf_params = {
    'n_components': 10,
    'alpha': 0.1,
    'l1_ratio': 0.5
}

nmf, nmf_words = nlp_feat.topics_lsa(
    data[sents_col], decompose=NMF,
    scaler=Normalizer, tfidf=True,
    vect_params=vect_params, dc_params=nmf_params
    )
plot_top_words(nmf, nmf_words, 20, 'Topics in NMF model (Frobenius norm)')

#%%
nmfk_params = {
    'n_components': 10,
    'alpha': 0.1,
    'l1_ratio': 0.5,
    'beta_loss': 'kullback-leibler', 
    'solver':'mu'
}

nmfk, nmfk_words = nlp_feat.topics_lsa(
    data[sents_col], decompose=NMF,
    scaler=Normalizer, tfidf=True,
    vect_params=vect_params, dc_params=nmfk_params
    )
plot_top_words(nmfk, nmfk_words, 20, 'Topics in NMF model (generalized Kullback-Leibler divergence)')

#%%
lda_params = {
    'n_components': 10,
    'learning_method': 'online'
}
lda, lda_words = nlp_feat.topics_lsa(
    data[sents_col], decompose=LatentDirichletAllocation,
    scaler=MaxAbsScaler, tfidf=False,
    vect_params=vect_params, dc_params=lda_params
    )
plot_top_words(lda, lda_words, 20, 'Topics in LDA model')
# %%

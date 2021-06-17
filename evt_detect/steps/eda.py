#%%
import sys
import os
from pathlib import Path
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import importlib

#%%
import evt_detect
importlib.reload(evt_detect)

from evt_detect.utils.file_io import read_file_df, to_file_df
from evt_detect.utils.visualize import compare_features
from evt_detect.features import nlp_features as nlp_feat

#%%
data_folder = Path(__file__).resolve().parents[2] / 'data'
label_folder = data_folder / 'label'
feature_folder = data_folder / 'feature'
tag = 'breach'

#%%
form_label = 'CR'
keys = ['Incident']

#%%
form_label = 'PR'
keys = ['Incident', 'Immaterial']

#%%
sents_notlabeled = label_folder / f'{tag}_{form_label}_sents.xlsx'
sents_labeled = label_folder / f'{tag}_{form_label}_sents_labeled_1.xlsx'
sents_col = 'sents'
data = read_file_df(sents_notlabeled)


#%%
# * Examine the named entities 
named_entities = nlp_feat.find_word_entity(data[sents_col])
entity_save = feature_folder / f'{tag}_{form_label}_named_entity.xlsx'
to_file_df(named_entities, entity_save)

#%%
# * Compare number of named entitie
compare_features(data, sents_col, keys, nlp_feat.entity_feature)

#%%
# * Compare most frequent words
countkwargs1 = {
    'lowercase': False,
    'tokenizer': nlp_feat.tokenizer_ent,
    'ngram_range': (1, 1),
}
top_100_1gram = nlp_feat.compare_top_n_words(data, sents_col, keys, n=100, **countkwargs1)
top_1gram_save = feature_folder / f'{tag}_{form_label}_top_1gram.xlsx'
to_file_df(top_100_1gram, top_1gram_save)

#%%
countkwargs2 = {
    'lowercase': False,
    'tokenizer': nlp_feat.tokenizer_ent,
    'ngram_range': (2, 2),
}
top_100_2gram = nlp_feat.compare_top_n_words(data, sents_col, keys, n=100, **countkwargs2)
top_2gram_save = feature_folder / f'{tag}_{form_label}_top_2gram.xlsx'
to_file_df(top_100_2gram, top_2gram_save)

#%%
countkwargs3 = {
    'lowercase': False,
    'tokenizer': nlp_feat.tokenizer_ent,
    'ngram_range': (3, 3),
}
top_100_3gram = nlp_feat.compare_top_n_words(data, sents_col, keys, n=100, **countkwargs3)
top_3gram_save = feature_folder / f'{tag}_{form_label}_top_3gram.xlsx'
to_file_df(top_100_3gram, top_3gram_save)

#%%
# * Compare length features
compare_features(data, sents_col, keys, nlp_feat.length_feature)

#%%
# * Compare parts-of-speech tags
compare_features(data, sents_col, keys, nlp_feat.pos_feature)

#%%
# * Compare sentiment features
compare_features(data, sents_col, keys, nlp_feat.sentiment_feature)

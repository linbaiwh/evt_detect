#%%
import sys
import os
from pathlib import Path
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import importlib

#%%
import evt_detect
importlib.reload(evt_detect)

from evt_detect.utils import preprocess as prep
from evt_detect.utils.file_io import to_file_df

#%%
# folder environment setup
topfolder = Path(r'E:\SEC filing')
tag = 'breach'
data_folder = Path(__file__).resolve().parents[2] / 'data'
input_folder = data_folder / 'input'
label_folder = data_folder / 'label'

#%%
# * Form types classification
CR_types = ['6-K', '8-K']
PR_types = ['10-Q', '10-K', '20-F', '40-F'] + ['DRS', 'S-1', 'S-3', 'S-4', 'F-1', 'F-10', 'F-3']

#%%
# form_types = CR_types
# form_label = 'CR'

#%%
form_types = PR_types
form_label = 'PR'

#%%
sents_save = label_folder / f'{tag}_{form_label}_sents.xlsx'
csv_ins, csv_outs = prep.find_formtypes(form_types, topfolder)

#%%
# * Decide how many unique cik - text to choose from each file
parags = prep.parags_shuffled(input_folder, csv_ins, csv_outs, nmax=110)

#%%
# * Decide how many unique sentences to choose for each cik - text
sents = prep.sents_shuffled(parags, nsents=4)

#%%
# * Generate file of sentences for mannual check
to_file_df(sents, sents_save)
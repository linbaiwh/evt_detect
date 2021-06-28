#%%
# import steps_context
from steps_context import topfolder, tag, label_folder, input_folder
#%%
from evt_detect.utils import preprocess as prep
from evt_detect.utils.file_io import to_file_df, read_file_df


#%%
# * Form types classification
CR_types = ['6-K', '8-K']
PR_types = ['10-Q', '10-K', '20-F', '40-F'] + ['DRS', 'S-1', 'S-3', 'S-4', 'F-1', 'F-10', 'F-3']

#%%
form_types = CR_types
form_label = 'CR'

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

#%%
# * Prepare unlabeled sentences fron previously labeled file
df = read_file_df(label_folder / 'whole_ALL_2_mc.xlsx')
sents = prep.unlabeled_sents(df, form_types)

#%%
sents_save_unlabel = label_folder / f'{tag}_{form_label}_sents_unlabel.xlsx'
to_file_df(sents, sents_save_unlabel)

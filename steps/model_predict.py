import logging
import logging.config
import warnings
import pandas as pd
import numpy as np
from joblib import load
import multiprocessing
from multiprocessing import Pool
from steps_context import topfolder, tag, label_folder, model_folder, result_folder, logger_conf

from evt_detect.models.model_build import parag_pred, sents_pred
from evt_detect.utils.file_io import read_file_df, to_file_df, merge_csv
from evt_detect.utils.preprocess import find_formtypes
import evt_detect.features.nlp_features as nlp_feat

warnings.filterwarnings("ignore")



def main(form_label, y_col, model_name, threshold=0.99, output='whole', inputs='whole', textcol='filtered_text'):

    logging.config.fileConfig(logger_conf)
    logger = logging.getLogger('model_predict')

    logger.info(f'Model predict for {form_label} using {model_name}')

    # * Preparing model
    model = load(model_folder / f'{form_label}_{y_col}_{model_name}.joblib')
    model_sum_df = read_file_df(model_folder / f'{form_label}_{y_col}_spec.xlsx')
    if threshold is None:
        threshold = model_sum_df.loc[model_sum_df['model_name']==model_name, 'threshold'].squeeze()

    # * Preparing files
    # Form types classification
    CR_types = ['6-K', '8-K']
    PR_types = ['10-Q', '10-K', '20-F', '40-F'] + ['DRS', 'S-1', 'S-3', 'S-4', 'F-1', 'F-10', 'F-3']

    if form_label == 'CR':
        form_types = CR_types
        tokenizer = nlp_feat.CR_tokenizer
    elif form_label == 'PR':
        form_types = PR_types
        tokenizer = nlp_feat.PR_tokenizer

    if inputs == 'sents':
        csv_in = result_folder / f'{form_label}_sents.xlsx'
        df = read_file_df(csv_in)
        sents = nlp_feat.add_tokens_pos(df, tokenizer)
        result_df = sents_pred(sents, y_col, model, threshold)
        logger.info(f'finish predicting {csv_in.name}')
        results_file = result_folder / f'{form_label}_sents_pred.xlsx'

    else:
        if 'sent' in textcol:
            csv_ins, csv_outs = find_formtypes(form_types, topfolder, tag=tag, 
            infolder='temp', outfolder=f'pred_{y_col}')
        else:
            csv_ins, csv_outs = find_formtypes(form_types, topfolder, tag=tag)

        for i in range(len(csv_ins)):
            if not csv_outs[i].exists():
                df = read_file_df(csv_ins[i])
                df.dropna(subset=[textcol], inplace=True)
                logger.info(f'start predicting {csv_ins[i].name}')
                nrows = df.shape[0]

                if form_label == 'CR':
                    sub = nrows // 2000 + 1

                else:
                    sub = nrows // 800 + 1
                
                if sub == 0:
                    continue
                
                dfs = np.array_split(df, sub)
                del df
                result_dfs = []
                for j in range(sub):
                    result_df = parag_pred(dfs[j], textcol=textcol, y_col=y_col, tokenizer=tokenizer,
                        output=output, model=model, threshold=threshold)
                    if result_df is not None:
                        result_dfs.append(result_df)
                        logger.info(f'finish predicting {csv_ins[i].name} chunk {j}')

                if result_dfs:
                    result_df = pd.concat(result_dfs)
                    logger.info(f'finish predicting {csv_ins[i].name}')
                    to_file_df(result_df, csv_outs[i])
                
        result_df = merge_csv(csv_outs)

        results_file = result_folder / f'{form_label}_pred.xlsx'
        if results_file.exists():
            result_pre = read_file_df(results_file)
            result_df = result_df.merge(result_pre, how='outer')


    to_file_df(result_df, results_file)

    logger.info(f'{form_label} {y_col} prediction using {model_name} is saved')

if __name__ == '__main__':
    # main('CR', 'Incident', 'Baseline', threshold=0.6, output='pos_sents')
    # main('CR', 'Incident', 'Baseline_self_train', threshold=0.98)
    # main('PR', 'Incident', 'Baseline', threshold=None, output='pos_sents')
    # main('PR', 'Related', 'Baseline_Robust', threshold=None, output='pos_sents')
    # main('PR', 'Incident', 'Baseline_Std', threshold=None, inputs='sents')
    # main('CR', 'Incident', 'Baseline_Robust', threshold=None, output='pos_sents')
    # main('CR', 'Incident', 'Baseline_self_train', threshold=0.98)
    # main('PR', 'Incident', 'Baseline', threshold=None, output='pos_sents')
    # main('PR', 'Related', 'Baseline_Robust', threshold=None, output='pos_sents')
    # main('PR', 'Related', 'Baseline_Robust', threshold=None, output='whole')

    # main('PR', 'Incident', 'Baseline_Std', threshold=None, textcol='Related_sents')
    main('PR', 'Immaterial', 'SVC_Std', threshold=None, textcol='Related_sents')

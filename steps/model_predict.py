import logging
import logging.config
import warnings
import pandas as pd
import numpy as np
from joblib import load
import multiprocessing
from multiprocessing import Pool
from steps_context import topfolder, tag, label_folder, model_folder, result_folder, logger_conf
from evt_detect.utils.file_io import read_file_df, to_file_df, merge_csv, parallelize_df
from evt_detect.models.model_build import parag_pred
from evt_detect.utils.preprocess import find_formtypes
import evt_detect.features.nlp_features as nlp_feat

warnings.filterwarnings("ignore")



def main(form_label, y_col, model_name, threshold=0.99, output='whole'):

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

    csv_ins, csv_outs = find_formtypes(form_types, topfolder, tag=tag)

    for i in range(len(csv_ins)):
        if not csv_outs[i].exists():
            df = read_file_df(csv_ins[i])
            df.dropna(subset=['filtered_text'], inplace=True)
            logger.info(f'start predicting {csv_ins[i].name}')

            dfs = np.array_split(df, 4)
            del df
            result_dfs = []
            for j in range(4):
                result_df = parag_pred(dfs[j], textcol='filtered_text', y_col=y_col, tokenizer=tokenizer,
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
    main('CR', 'Incident', 'Baseline', threshold=0.6, output='pos_sents')
    # main('CR', 'Incident', 'Baseline_self_train', threshold=0.98)
    # main('PR', 'Incident', 'Baseline', threshold=None, output='pos_sents')
    # main('PR', 'Related', 'Baseline_Robust', threshold=None, output='pos_sents')

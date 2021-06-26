import sys
import os
from pathlib import Path
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
from joblib import load
import multiprocessing
from multiprocessing import Pool
from functools import partial

import logging
import logging.config

import warnings
warnings.filterwarnings("ignore")

from evt_detect.utils.file_io import read_file_df, to_file_df, merge_csv
from evt_detect.features import nlp_features as nlp_feat
from evt_detect.models.model_build import model_pred
from evt_detect.utils.preprocess import find_formtypes

def file_model_pred(csv_in, csv_out, X_col, y_col, model_name, model, threshold):
    df = read_file_df(csv_in)
    sents = df[X_col].map(nlp_feat.gen_sents)
    pos_sents = sents.apply(model_pred, model=model, threshold=threshold)

    df[f'{y_col}_{model_name}_sents_num'] = pos_sents.map(len)
    df[f'{y_col}_{model_name}_sents'] = pos_sents.map(lambda s: " \n ".join(s))
    df[f'{y_col}_{model_name}_pred'] = pos_sents.map(lambda s: 1 if len(s) > 0 else 0)
    
    df = df.loc[df[f'{y_col}_{model_name}_pred'] == 1]
    to_file_df(df, csv_out)

    return csv_out


def main(form_label, y_col, model_name):

    logger_conf = Path(__file__).resolve().parents[2] / 'docs' / 'logging.conf'
    logging.config.fileConfig(logger_conf)
    logger = logging.getLogger('model_predict')

    logger.info(f'Model predict for {form_label} using {model_name}')

    # file location
    topfolder = Path(r'E:\SEC filing')
    tag = 'breach'
    data_folder = Path(__file__).resolve().parents[2] / 'data'
    model_folder = data_folder / 'model'
    result_folder = data_folder / 'result'

    # * Preparing model
    model = load(model_folder / f'{form_label}_{y_col}_{model_name}.joblib')
    model_sum_df = read_file_df(model_folder / f'{form_label}_{y_col}_spec.xlsx')
    threshold = model_sum_df.loc[model_sum_df['model_name']==model_name, 'threshold'][0]

    # * Preparing files
    # Form types classification
    CR_types = ['6-K', '8-K']
    PR_types = ['10-Q', '10-K', '20-F', '40-F'] + ['DRS', 'S-1', 'S-3', 'S-4', 'F-1', 'F-10', 'F-3']

    if form_label == 'CR':
        form_types = CR_types
    elif form_label == 'PR':
        form_types = PR_types

    csv_ins, csv_outs = find_formtypes(form_types, topfolder, tag=tag)

    num_cores = multiprocessing.cpu_count()

    map_file_pred = partial(file_model_pred, 
    X_col='filtered_text', y_col=y_col, 
    model_name=model_name, model=model, threshold=threshold)

    with Pool(num_cores) as pool:
        csv_outs = pool.map(map_file_pred, zip(csv_ins, csv_outs))

    result_df = merge_csv(csv_outs)

    results_file = result_folder / f'{form_label}_pred.xlsx'
    if results_file.exists():
        result_pre = read_file_df(results_file)
        result_df = result_df.merge(result_pre, how='outer')

    to_file_df(result_df, results_file)

    logger.info(f'{form_label} {y_col} prediction using {model_name} is saved')

if __name__ == '__main__':
    main('CR', 'Incident', 'Baseline')

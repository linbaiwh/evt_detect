import logging
import pandas as pd
import numpy as np
from functools import partial
from multiprocessing.pool import Pool as Pool

logger = logging.getLogger(__name__)

def read_file_df(csv_in,**readkwargs):
    if csv_in.suffix == '.csv':
        for encode in ('cp1252','utf-8-sig'):
            try:
                df = pd.read_csv(csv_in,encoding=encode,**readkwargs)
                return df
            except UnicodeDecodeError:
                continue
            except pd.errors.EmptyDataError:
                logger.exception('CSV is empty')
                return None
    elif csv_in.suffix == '.xlsx':
        try:
            df = pd.read_excel(csv_in,**readkwargs)
            return df
        except UnicodeDecodeError:
            logger.exception('cannot read excel')
            return None


def to_file_df(df, csv_out,**savekwargs):
    for encode in ('cp1252','utf-8-sig'):
        if csv_out.suffix == '.csv':
            try:
                df.to_csv(csv_out,encoding=encode,index=False,**savekwargs)
                break
            except UnicodeEncodeError:
                continue
        elif csv_out.suffix == '.xlsx':
            try:
                df.to_excel(csv_out,encoding=encode,index=False,**savekwargs)
                break
            except UnicodeEncodeError:
                continue
    return csv_out

def merge_csv(csv_list, outcsv=False, readkwargs={}, mergekwargs={'ignore_index':True}):
    dfs = [read_file_df(csv_in, **readkwargs) for csv_in in csv_list if csv_in.exists()]
    merged_df = pd.concat(dfs, **mergekwargs)
    
    if outcsv:
        to_file_df(merged_df, outcsv)
    
    return merged_df

def run_apply(df_or_series, func, **kwargs):
    return df_or_series.apply(func, **kwargs)

def parallelize_df(df, func, n_chunks=64, **kwargs):
    if n_chunks == 0:
        return []
    df_split = np.array_split(df, n_chunks)
    func = partial(run_apply, func=func, **kwargs)
    try:
        with Pool(processes=8) as pool:
            return list(pool.imap_unordered(func,df_split,chunksize=2))
    except:
        logger.exception("Uncaught exception for parallelize_df")
        raise

def gen_duo(thelist):
    if len(thelist) < 2:
        return thelist[0]
    length = len(thelist)
    for j in range(0, length, 2):
        if j >= len(thelist):
            return
        if j == len(thelist) - 1:
            yield thelist[-1]
        else:
            duo = thelist[j : j + 2]
            yield duo

def concat(duo):
    if isinstance(duo, list):
        return pd.concat([duo[0], duo[1]])
    else:
        return duo

def fast_concat(dfs):
    while len(dfs) > 1:
        with Pool(processes=8) as pool:
            dfs = list(pool.imap_unordered(concat, gen_duo(dfs)))
    return dfs[0]

def df_concat(df_array):
    return pd.concat(df_array.tolist(), keys=df_array.index, names=['idx']).reset_index(0).reset_index(drop=True)

def fast_df_concat(dfs_all, n_chunks=64):
    if n_chunks == 0:
        return None
    if not isinstance(dfs_all, list):
        dfs_all = np.array_split(dfs_all, n_chunks)
    with Pool(processes=8) as pool:
        dfs = list(pool.imap_unordered(df_concat, dfs_all))
    return fast_concat(dfs)

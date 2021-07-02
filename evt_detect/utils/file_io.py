import logging
import pandas as pd
import numpy as np
from functools import partial
from multiprocessing import Pool

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

def parallelize_df(df, func, n_chunks=64, **kwargs):
    df_split = np.array_split(df, n_chunks)
    mapfunc = partial(func, **kwargs)
    try:
        with Pool(n_chunks // 4) as pool:
            results = pool.imap_unordered(mapfunc,df_split,chunksize=4)
    except:
        logger.exception("Uncaught exception for parallelize_df")
        return None
    else:
        return pd.concat(results, ignore_index=True)
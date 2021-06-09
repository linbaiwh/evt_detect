import logging
import pandas as pd

logger = logging.getLogger(__name__)

def read_file_df(csv_in,**readkwargs):
    if csv_in.suffix == '.csv':
        for encode in ('cp1252','utf-8-sig'):
            try:
                df = pd.read_csv(csv_in,encoding=encode,**readkwargs)
                return df
            except UnicodeDecodeError:
                continue
    elif csv_in.suffix == '.xlsx':
        try:
            df = pd.read_excel(csv_in,**readkwargs)
            return df
        except UnicodeDecodeError:
            logger.exception('cannot read excel')


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

def merge_csv(csv_list, outcsv=False, readkwargs=None, mergekwargs={'ignore_index':True}):
    dfs = [read_file_df(csv_in, **readkwargs) for csv_in in csv_list]
    merged_df = pd.concat(dfs, **mergekwargs)
    
    if outcsv:
        to_file_df(merged_df, outcsv)
    
    return merged_df
    
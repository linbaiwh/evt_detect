import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from file_io import read_file_df, to_file_df, merge_csv
from ..features.nlp_features import gen_sents


logger = logging.getLogger(__name__)

# * randomly select at most nmax files per year per file type
def shuffle_files(csv_in, csv_out, ciks=None, nmax=160, frac=0.5, textcol='filtered_text'):
    df = read_file_df(csv_in)
    df['cik'] = df['cik'].apply(str)
    if ciks is not None:
        df = df.loc[df['cik'].isin(ciks)]

    df = df.loc[df[textcol].notna(), ['cik', textcol]]
    df = df.sample(frac=1).groupby(['cik']).head(1)
    unique_firms = df.shape[0]
    logger.info(f'{unique_firms} unique firms in {csv_in.name[:-4]}')
    
    if unique_firms > nmax / frac:
        df = df.sample(frac=frac, random_state=42).head(nmax)
    else:
        df = df.sample(frac=1, random_state=42).head(nmax)
    to_file_df(df,csv_out)
    
    if unique_firms:
        return unique_firms, df.shape[0]/unique_firms
    return 0, 0

# * select ciks that experience breach media reports
def select_breach_firms(datafolder):
    rpbreached = datafolder / 'rpbreach_cik.csv'
    breached_df = read_file_df(rpbreached, dtype={'cik':str})
    return set(breached_df['cik'].dropna().unique())

# * find original text files for the specified form types
def find_formtypes(form_types, topfolder, tag='breach'):
    tag_folder = topfolder / tag

    tinfo_folder = tag_folder / 'tinfo'
    tempfolder = tag_folder / 'temp'

    tcsvs = []
    for form_type in form_types:
        tcsvs_0 = sorted(tinfo_folder.glob(f'{tag}_{form_type}*.csv'))
        tcsvs = tcsvs + tcsvs_0

    csv_outs = [tempfolder / csv_in.name for csv_in in tcsvs]

    return tcsvs, csv_outs

# * Select random unique cik - text for a list of files and merge the results
def parags_shuffled(cikfolder, csv_ins, csv_outs, nmax=160, frac=0.5):
    uniques = []
    select_pers = []
    ciks = select_breach_firms(cikfolder)
    for i in range(len(csv_ins)):
        unique_firms, select_per = shuffle_files(csv_ins[i],csv_outs[i],ciks=ciks,nmax=nmax,frac=frac)
        uniques.append(unique_firms)
        select_pers.append(select_per)

    _, (ax1, ax2) = plt.subplots(2, 1)
    sns.histplot(x=uniques, ax=ax1)
    sns.histplot(x=select_pers, ax=ax2)

    df = merge_csv(csv_outs, outcsv=False)

    duplicate_firms = df[['cik']].duplicated().sum()
    logger.info(f'{duplicate_firms} duplicated firms')

    return df

# * Randomly select nsents sentences for each cik - text, resulting the file for mannual labeling
def sents_shuffled(df, nsents=2, textcol='filtered_text'):
    df.set_index('cik',inplace=True)
    df['sents'] = df[textcol].map(gen_sents)
    
    # visualize number of sentences for each observation
    num_sents = df['sents'].map(len)
    sns.histplot(x=num_sents)

    sents = df['sents'].explode().reset_index()
    sents = sents.dropna(subset=['sents']).drop_duplicates()

    shuffled = sents.sample(frac=1, random_state=42).groupby('cik').head(nsents)
    
    # add classification outcome variables for mannual fill
    shuffled = shuffled.assign(Incident='', Immaterial='', Cost='', Litigation='', Management='')

    return shuffled
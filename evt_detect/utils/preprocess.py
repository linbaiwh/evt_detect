import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from evt_detect.utils.file_io import read_file_df, to_file_df, merge_csv
from evt_detect.features.nlp_features import gen_sents


logger = logging.getLogger(__name__)

# * randomly select at most nmax files per year per file type
def shuffle_files(csv_in, csv_out, ciks=None, nmax=160, textcol='filtered_text'):
    """Randomly select one observation for at most nmax firms (ciks).

    Args:
        csv_in (pathlib.Path): Original csv file containing cik and filing text
        csv_out (pathlib.Path): Output csv file
        ciks (List of string, optional): List of ciks to filter file. Defaults to None.
        nmax (int, optional): maximum observations (ciks) selected. Defaults to 160.
        textcol (str, optional): column name of the text. Defaults to 'filtered_text'.

    Returns:
        int: number of unique firms in the input file
        float: percentage of selected firms of the input file

    Output:
        file: resulting csv file
    """
    df = read_file_df(csv_in, low_memory=False)
    if df is None:
        return 0, 0
    df['cik'] = df['cik'].apply(str)
    if ciks is not None:
        df = df.loc[df['cik'].isin(ciks)]

    df = df.loc[df[textcol].notna(), ['cik', textcol]]
    df = df.sample(frac=1).groupby(['cik']).head(1)
    unique_firms = df.shape[0]
    logger.info(f'{unique_firms} unique firms in {csv_in.name[:-4]}')
    
    df = df.sample(frac=1, random_state=42).head(nmax)

    to_file_df(df,csv_out)
    
    if unique_firms:
        return unique_firms, df.shape[0]/unique_firms
    return 0, 0

# * select ciks that experience breach media reports
def select_breach_firms(datafolder):
    """Retrieve certain ciks from 'rpbreach_cik.csv'

    Args:
        datafolder (pathlib.Path): Folder that contain the input file

    Returns:
        set: set of ciks (strings)
    """
    rpbreached = datafolder / 'rpbreach_cik.csv'
    breached_df = read_file_df(rpbreached, dtype={'cik':str})
    return set(breached_df['cik'].dropna().unique())

# * find original text files for the specified form types
def find_formtypes(form_types, topfolder, tag='breach'):
    """Find original files accoring to certain form types.

    Args:
        form_types (List of string): List of form types
        topfolder (pathlib.Path): Folder that stores all original files
        tag (str, optional): name of the subfolder for certain project. Defaults to 'breach'.

    Returns:
        List of pathlib.Path: List of original files
        List of pathlib.Path: List of potential output file names
    """
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
def parags_shuffled(cikfolder, csv_ins, csv_outs, nmax=160):
    """For each file in csv_ins, randomly select at most nmax firm observations, then merge

    Args:
        cikfolder (pathlib.Path): Folder containing the needed ciks
        csv_ins (List of Path): list of input csvs
        csv_outs (List of Path): list of files containing selected observations
        nmax (int, optional): maximum number of firm observations to select in each file. Defaults to 160.

    Returns:
        DataFrame: resulting merged DataFrame
    """
    uniques = []
    select_pers = []
    ciks = select_breach_firms(cikfolder)
    for i in range(len(csv_ins)):
        unique_firms, select_per = shuffle_files(csv_ins[i],csv_outs[i],ciks=ciks,nmax=nmax)
        uniques.append(unique_firms)
        select_pers.append(select_per)

    _, (ax1, ax2) = plt.subplots(2, 1)
    sns.histplot(x=uniques, ax=ax1)
    sns.histplot(x=select_pers, ax=ax2)

    df = merge_csv(csv_outs, outcsv=False, readkwargs={'low_memory': False})

    duplicate_firms = df[['cik']].duplicated().sum()
    logger.info(f'{duplicate_firms} duplicated firms')

    return df

# * Randomly select nsents sentences for each cik - text, resulting the file for mannual labeling
def sents_shuffled(df_p, nsents=2, textcol='filtered_text'):
    """Randomly select at most nsents sentences for each firm

    Args:
        df_p (DataFrame): DataFrame containing firm-text observations
        nsents (int, optional): number of sentences to select for each firm. Defaults to 2.
        textcol (str, optional): column name in the df_p for the text. Defaults to 'filtered_text'.

    Returns:
        DataFrame: DataFrame containing firm (cik), sentences (sents), and indicators for mannual
        labeling (Incident, Immaterial, Cost, Litigation, Management) 
    """
    df = df_p.set_index('cik')
    df['sents'] = df[textcol].map(gen_sents)
    
    # visualize number of sentences for each observation
    num_sents = df['sents'].map(len)
    sns.histplot(x=num_sents)

    sents = df['sents'].explode().reset_index()
    sents = sents.dropna(subset=['sents']).drop_duplicates()

    shuffled = sents.sample(frac=1, random_state=42).groupby('cik').head(nsents)
    shuffled = shuffled.sort_values(by='cik')

    # add classification outcome variables for mannual fill
    shuffled = shuffled.assign(Incident='', Immaterial='', Cost='', Litigation='', Management='')

    return shuffled
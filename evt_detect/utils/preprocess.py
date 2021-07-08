import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from evt_detect.utils.file_io import read_file_df, to_file_df, merge_csv
from evt_detect.features.nlp_features import parag_to_sents, CR_tokenizer, PR_tokenizer


logger = logging.getLogger(__name__)

# * randomly select at most nmax files per year per file type
def shuffle_files(df, ciks=None, nmax=160, textcol='filtered_text'):
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
    if df is None:
        return 0, 0
    df['cik'] = df['cik'].apply(str)
    if ciks is not None:
        df = df.loc[df['cik'].isin(ciks)]

    df = df.loc[df[textcol].notna(), ['cik', textcol]]
    df = df.sample(frac=1).groupby(['cik']).head(1)
    unique_firms = df.shape[0]
    
    if nmax is not None:
        df = df.sample(frac=1, random_state=2021).head(nmax)
    
    if unique_firms:
        return df, unique_firms, df.shape[0]/unique_firms
    return df, 0, 0

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
def find_formtypes(form_types, topfolder, tag='breach', year='all'):
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
        if year == 'all':
            tcsvs_0 = sorted(tinfo_folder.glob(f'{tag}_{form_type}*.csv'))
        else:
            tcsvs_0 = sorted(tinfo_folder.glob(f'{tag}_{form_type}_{year}.csv'))
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
    if cikfolder is not None:
        ciks = select_breach_firms(cikfolder)
    else:
        ciks = None
    for i in range(len(csv_ins)):
        df = read_file_df(csv_ins[i], low_memory=False)
        df, unique_firms, select_per = shuffle_files(df,ciks=ciks,nmax=nmax)
        logger.info(f'{unique_firms} unique firms in {csv_ins[i].name[:-4]}')

        to_file_df(df,csv_outs[i])
        uniques.append(unique_firms)
        select_pers.append(select_per)

    if len(csv_ins) > 1:
        _, (ax1, ax2) = plt.subplots(2, 1)
        sns.histplot(x=uniques, ax=ax1)
        sns.histplot(x=select_pers, ax=ax2)

        df = merge_csv(csv_outs, outcsv=False, readkwargs={'low_memory': False})

        duplicate_firms = df[['cik']].duplicated().sum()
        logger.info(f'{duplicate_firms} duplicated firms')

    return df

# * Randomly select nsents sentences for each cik - text, resulting the file for mannual labeling
def sents_shuffled(df_p, nsents=2, textcol='filtered_text', form_label='CR'):
    """Randomly select at most nsents sentences for each firm

    Args:
        df_p (DataFrame): DataFrame containing firm-text observations
        nsents (int, optional): number of sentences to select for each firm. Defaults to 2.
        textcol (str, optional): column name in the df_p for the text. Defaults to 'filtered_text'.

    Returns:
        DataFrame: DataFrame containing firm (cik), sentences (sents), and indicators for mannual
        labeling (Incident, Immaterial, Cost, Litigation, Management) 
    """
    if form_label == 'CR':
        tokenizer = CR_tokenizer
    elif form_label == 'PR':
        tokenizer = PR_tokenizer

    sents_dfs = df_p[textcol].apply(parag_to_sents, tokenizer=tokenizer).tolist()
    sents_dfs = [sents_df.assign(cik=cik) for cik, sents_df in zip(df_p['cik'], sents_dfs)]

    # visualize number of sentences for each observation
    num_sents = [sents_df.shape[0] for sents_df in sents_dfs]
    sns.histplot(x=num_sents)

    sents = pd.concat(sents_dfs, ignore_index=True)
    sents.drop_duplicates(inplace=True)
    
    if nsents is not None:
        shuffled = sents.sample(frac=1, random_state=42).groupby('cik').head(nsents)
    else:
        shuffled = sents

    shuffled = shuffled.sort_values(by='cik')

    # add classification outcome variables for mannual fill
    shuffled = shuffled.assign(Incident='', Immaterial='', Cost='', Litigation='', Management='')

    return shuffled

# add more example senteces from previous labeled file
def unlabeled_sents(df, form_types):
    keep_cols = ['cik', 'filtered_text']
    df = df.loc[df['form_type'].map(lambda form: form in form_types)]
    df = df.loc[df['sents'].notna() == False]
    Incident_df = df.loc[df['Incident'] == 1, keep_cols]
    Immaterial_df = df.loc[df['Immaterial'] == 1, keep_cols]
    df_p = pd.concat([Incident_df, Immaterial_df], ignore_index=True)
    return sents_shuffled(df_p, nsents=None)

def rm_features(df, rm_cols):
    features = df.columns.tolist()
    for rm_col in rm_cols:
        try:
            features.remove(rm_col)
        except ValueError:
            pass
    return features

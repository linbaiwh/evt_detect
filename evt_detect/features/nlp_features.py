import pandas as pd
import re
from pathlib import Path
from functools import partial
from textblob import TextBlob
from sklearn.preprocessing import Normalizer, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from spacy.language import Language
import spacy

import logging
logger = logging.getLogger(__name__)

nlp = spacy.load('en_core_web_md')
doc_folder = Path(__file__).resolve().parents[2] / 'docs'

@Language.component("EDGAR_sentencizer")
def EDGAR_sentencizer(doc):
    """Add EDGAR specific sentencizer to nlp pipeline

    Args:
        doc (spacy.Doc): spacy DOC instance

    Returns:
        spacy.Doc: spacy DOC instance with
    """
    len_doc = len(doc)
    i = 0
    while i < len_doc:
        token = doc[i]
        if i == 0:
            token.is_sent_start = True  # First token is always sentence start
            i += 1
            continue
        elif i == len_doc-1:
            token.is_sent_start = False # Last token can never be sentence start
            i += 1
            continue

        # when token is comma or possesive, this token and the next token cannot be sentence start
        if token.text in (',', "'s", "’s", "v.", "of", "&", "/"):
            token.is_sent_start = False
            doc[i+1].is_sent_start = False
            i += 2
            continue

        # when token and the previous token is consecutive, this token cannot be sentence start
        if token.nbor(-1).text in ("Form", "Case", "File", "No"):
            token.is_sent_start = False
            if doc[i-1].text + doc[i].text == 'No.':
                doc[i+1].is_sent_start = False
                i += 2
                continue
            i += 1
            continue

        # when token is "(" and the previous token suggests end, this token is sentence start
        if token.nbor(-1).text in (";", ":") and token.text in ("(", "and", "or"):
            if token.text in ("and", "or"):
                i += 1
            if i + 3 < len_doc and doc[i+2].text == ")":
                if 1 <= len(doc[i+1]) <= 3 or set(doc[i+1].lower_) <= set('ivx'):
                    doc[i].is_sent_start = True
                    doc[i+1].is_sent_start = False
                    doc[i+2].is_sent_start = False
                    doc[i+3].is_sent_start = False
                    i += 4
                    continue

        # contents between left bracket and right bracket cannot be sentece start
        if token.text == "(":
            i += 1
            while i < len_doc:
                doc[i].is_sent_start = False
                if doc[i].text == ")":
                    break
                i += 1
            i += 1
            continue
        
        i += 1
    return doc

nlp.add_pipe("EDGAR_sentencizer", before="parser")
nlp.add_pipe("entity_ruler", before="ner").from_disk(doc_folder / "ent_pattern.jsonl")
nlp.add_pipe("merge_entities")


def valid_sent(doc):
    """Decide whether a sentence is valid or not.
    Valid sentence is defined as a sentence with more than 2 meaningful words;
    a meaningful word is an alphabetical word that has more than 2 letters.

    Args:
        doc (spacy.Doc): spacy.Doc instance of a sentence

    Returns:
        bool: True if the sentence is valid, and False otherwise
    """
    if len(doc) >=4:
        words = set([token.lower for token in doc if token.is_alpha & (len(token)>2)])
        if len(words) >= 3:
            return True
    return False


def normalize_text(text):
    """remove excessive characters

    Args:
        text (string): The original text to analyze

    Returns:
        string: text after removing excessive characters
    """
    try:
        text = text.replace('\n',' ')
        text = text.replace(r'\99',' ')
        text = ' '.join(text.split())
    except AttributeError:
        logger.error('text is not string')
        return str(text)
    else:
        return text


def gen_sents(text):
    """Generate valid sentences for a paragraph, including removing excessive characters

    Args:
        text (string): The original text containing multiple sentences

    Returns:
        List of strings: list of valid sentences in string format
    """
    while True:
        try:
            sents = [sent for sent in nlp(normalize_text(text)).sents]
            break
        except ValueError:
            nlp.max_length = len(text)

    return [str(sent) for sent in sents if valid_sent(sent)]

# * Each valid sentence is an observation for machine learning observation
# todo: Need to add labels to each sentence mannually

# * Named Entity Analysis
def find_word_entity(sents):
    """Generate table to present named entities, all upper words, and out-of-vector words
    in the corpus, using spaCy.

    Args:
        sents (List of string): the corpus as a list of sentences

    Returns:
        pandas.DataFrame: resulting table
    """
    features = {}

    for sent in sents:
        doc = nlp(sent)
        for ent in doc.ents:
            features[ent.label_] = features.get(ent.label_, []) + [ent.text]
        for token in doc:
            if token.ent_iob_ == 'O' and token.is_alpha:
                if token.is_upper:
                    features['is_upper'] = features.get('is_upper', []) + [token.text]
                if token.is_oov:
                    features['is_oov'] = features.get('is_oov', []) + [token.text]
            elif token.ent_iob_ == 'B' and token.ent_type_ == 'ORG' and not match_not_ent(token):
                features['modified_ORG'] = features.get('modified_ORG', []) + [token.text]
    
    for feature in features:
        features[feature] = sorted(list(set(features[feature])))

    return pd.DataFrame.from_dict(features, orient='index').transpose()

# modify the named entities (remove incorrect named entities)
def match_not_ent(token):
    tokens = token.lower_.split()
    if token.ent_type_ == 'ORG':
        neg_org = ["agreement", "breach", "requirement", "change", "report", "stock", "contract", "shareholder", "policy", 
        "statement", "rate", "notice", "diluted", "government", "document", "commitment", "offer", "award", "termination", "pin",
        "law", " act", " regulation", " time", "market", "asset", "regulatory"]
        for norg in neg_org:
            if norg in tokens:
                return True
    return False

CR_ents = ['DATE', 'TIME', 'TICKER', 'GPE', 'PERSON', 'PERCENT', 'LAW', 'MONEY', 'FILING', 'GOV', 'LOC', 'SEC', 'ORG']
PR_ents = ['PERCENT', 'LAW', 'MONEY', 'FILING', 'GOV', 'SEC']

# * tokenizer that replace certain named entities with the entity label
def tokenizer_ent(text, ents=None):
    """tokenizer that transform text to a list of tokens, can be used with CountVectorizer.
    The tokenizer replaces certain types of named entity with the entity label from spaCy.

    Args:
        text (string): original text to tokenize

    Returns:
        List of string: List of resulting tokens
    """
    ORG_ent = False
    if ents is not None:
        if 'ORG' in ents:
            ORG_ent = True
            ents = [ent for ent in ents if ent != 'ORG']
        
    doc = nlp(text)
    tokens = []
    for token in doc:
        if ents is not None and token.ent_iob_ == 'B' and token.ent_type_ in ents:
            tokens.append(token.ent_type_)
        elif ORG_ent and token.ent_iob_ == 'B' and token.ent_type_ == 'ORG' and not match_not_ent(token):
            tokens.append('ORG')
        elif token.ent_iob_ == 'B':
            tokens += [s for s in token.text.split() if s.isalpha()]
        elif token.ent_iob_ == 'O' and token.pos_ != 'PUNCT':
            if token.like_url:
                tokens.append('URL')
            elif token.like_email:
                tokens.append('EMAIL')
            elif token.like_num:
                tokens.append('NUM')
            elif match_itemNo(token):
                tokens.append('ITEMNUM')
            elif token.is_alpha and len(token) > 1 and not token.is_oov:
                tokens.append(token.text)

    return tokens

# match item No. like (1), (a), (iv)
def match_itemNo(token):
    try:
        if token.nbor(-1).text == '(' and token.nbor(1).text == ')':
            pattern = re.compile(r"^([a-zA-Z]{1}|[ivx]+|\d{1})$")
            if pattern.match(token.text):
                return True
    except IndexError:
        return False
    return False

def entity_feature(sents, ents=None):
    """Count number of specified named entities for each sentence, using spaCy

    Args:
        sents (List of string): List of sentences
        ents (List of string): List of entities to summarize

    Returns:
        DataFrame: DataFrame containing the original sentences and the number of specified named entities
    """
    vocab = ['URL', 'EMAIL', 'NUM', 'ITEMNUM']
    if ents is not None:
        vocab += ents
    
    tokenizer = partial(tokenizer_ent, ents=ents)

    vectorizer = CountVectorizer(lowercase=False, tokenizer=tokenizer, vocabulary=vocab)
    bag_of_words = vectorizer.fit_transform(sents)
    df = pd.DataFrame(bag_of_words.toarray(), columns=vectorizer.get_feature_names())
    df['sents'] = sents
    return df

CR_tokenizer = partial(tokenizer_ent, ents=CR_ents)
PR_tokenizer = partial(tokenizer_ent, ents=PR_ents)

# * Frequency Analysis
def get_top_n_words(sents, n=100, **kwargs):
    """Generate list of the n most frequent words in the corpus. Any transformation
    of the words are applied through tokenizer or other parameters of CounterVectorizer.

    Args:
        sents (List of string): the corpus as a list of string
        n (int, optional): The number of most frequent words to present. Defaults to 100.

    Returns:
        List of string, List of float: 
            List of the n most frequent words, List of corresponding frequency
    """
    vectorizer = CountVectorizer(**kwargs)
    bag_of_words = vectorizer.fit_transform(sents)
    mean_words = bag_of_words.mean(axis=0)
    word_freq = [(word, mean_words[0, idx] * 1000) for word, idx in vectorizer.vocabulary_.items()]
    word_freq.sort(key=lambda x: x[1], reverse=True)
    return list(zip(*word_freq[:n]))

# * Length Analysis
def length_feature(sents, tokenizer=tokenizer_ent):
    """Generate length features for each sentence

    Args:
        sents (List of string): List of sentences
        tokenizer (callable): a tokenizer function that transforms text to list of tokens 

    Returns:
        DataFrame: DataFrame containing original sentences and the length features
    """
    df = pd.DataFrame({'sents': sents})
    if tokenizer is None:
        tokenizer = lambda x: str(x).split(" ")
    tokens = df['sents'].map(tokenizer)
    df['word_count'] = tokens.map(len)
    df['char_count'] = tokens.map(lambda x: sum(len(token) for token in x))
    df['avg_word_length'] = df['char_count'] / df['word_count']
    df['unique_count'] = tokens.map(lambda x: len(set(x)))
    df['unique_vs_words'] = df['unique_count'] / df['word_count']
    return df

# * Linguistic Analysis
def count_pos_tag(doc, tag, pos):
    """Count number of certain tag/pos in a spaCy Doc.

    Args:
        doc (spaCy.Doc): text transformed to doc
        tag (string): name of a specific detailed part-of-speech tag
        pos (string): name of a simple UPOS part-of-speech tag

    Returns:
        int: number of the certain tag/pos
    """
    if tag:
        return sum(token.tag_ == tag for token in doc)
    if pos:
        return sum(token.pos_ == pos for token in doc)


def pos_feature(sents):
    """Generate part-of-speech features for each sentence, using spaCy

    Args:
        sents (List of string): List of sentences

    Returns:
        DataFrame: DataFrame containing original sentences and part-of-speech features
    """
    df = pd.DataFrame({'sents': sents})
    tokens = df['sents'].map(nlp)
    # number of tokens that are not punctuation
    df['token_count'] = tokens.map(lambda t: sum(token.pos_!='PUNCT' for token in t))
    # percentage of verb, past tense
    df['VBD_perc'] = tokens.apply(count_pos_tag, args=('VBD', False,)) / df['token_count']
    # percentage of verb, perfect tense
    df['VBN_perc'] = tokens.apply(count_pos_tag, args=('VBN', False)) / df['token_count']
    # percentage of verb, modal auxiliary
    df['MD_perc'] = tokens.apply(count_pos_tag, args=('MD', False)) / df['token_count']
    # percentage of verb
    df['VERB_perc'] = tokens.apply(count_pos_tag, args=(False, 'VERB')) / df['token_count']
    # percentage of noun
    df['NOUN_perc'] = tokens.apply(count_pos_tag, args=(False, 'NOUN')) / df['token_count']
    return df

# * Sentiment Analysis
def sentiment_feature(sents):
    """Generate sentiment features for each sentence, using TextBlob package.

    Args:
        sents (List of string): List of sentences

    Returns:
        DataFrame: DataFrame containing original sentences and sentiment features
    """
    df = pd.DataFrame({'sents': sents})
    blobs = df['sents'].map(TextBlob)
    df['polarity'] = blobs.map(lambda t: t.sentiment.polarity)
    df['subjectivity'] = blobs.map(lambda t: t.sentiment.subjectivity)
    return df

# * Compare features between classification groups
# Compare n most frequency words
def compare_top_n_words(df, sents_col, keys, n=100, **kwargs):
    """Compare frequencies of the top n words between groups set by keys

    Args:
        df (DataFrame): DataFrame containing sentences and labels (keys) for each sentence
        sents_col (string): column name for the sentences
        keys (List of string): List of column names for the labels (keys)
        n (int, optional): number of the most frequent words for each group. Defaults to 100.

    Returns:
        DataFrame: Each row is a top n word, and each column represents the frequency of
        a specified group.
    """
    corpus_all = df[sents_col]
    top_100_allwords, top_100_allfreq = get_top_n_words(corpus_all, n=100, **kwargs)

    top_100 = pd.DataFrame({'words': top_100_allwords, 'avg_all': top_100_allfreq})

    for key in keys:
        corpus_pos = df.loc[df[key]==1, [sents_col]].squeeze()
        corpus_neg = df.loc[df[key]==0, [sents_col]].squeeze()

        top_100_pos_words, top_100_pos_freq = get_top_n_words(corpus_pos, n=100, **kwargs)
        top_100_neg_words, top_100_neg_freq = get_top_n_words(corpus_neg, n=100, **kwargs)
        top_100_pos = pd.DataFrame({'words': top_100_pos_words, f'{key}_pos': top_100_pos_freq})
        top_100_neg = pd.DataFrame({'words': top_100_neg_words, f'{key}_neg': top_100_neg_freq})
        
        top_100 = top_100.merge(top_100_pos, how='outer', on='words')
        top_100 = top_100.merge(top_100_neg, how='outer', on='words')
        
    return top_100

#   * Latent Semantic Analysis (LSA, LSI) & NMF & Latent Dirichlet Allocation (LDA)
def topics_lsa(X, decompose=TruncatedSVD, scaler=Normalizer, tfidf=True, vect_params={}, dc_params={}):
    steps = [
        ('vect', CountVectorizer(**vect_params))
    ]
    if tfidf:
        steps.append(('tfidf', TfidfTransformer(use_idf=True, sublinear_tf=True)))

    if scaler == StandardScaler:
        steps.append(('scaler', StandardScaler(with_mean=False)))
    elif scaler == RobustScaler:
        steps.append(('scaler', RobustScaler(with_centering=False)))
    else:
        steps.append(('scaler', scaler()))
    
    steps.append(('decompose', decompose(**dc_params)))
    pipe = Pipeline(steps).fit(X)
    feature_names = pipe.named_steps.vect.get_feature_names()

    return pipe.named_steps.decompose, feature_names 

    
import pandas as pd
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from spacy.language import Language
import spacy

nlp = spacy.load('en_core_web_md')

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
            if i + 2 < len_doc and doc[i+2].text == ")":
                if 1 <= len(doc[i+1]) <= 3 or set(list(doc[i+1].lower_)) <= set('i','v','x'):
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
    text = text.replace('\n',' ')
    text = ' '.join(text.split())
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
def word_entity_feature(sents):
    """Generate table to present named entities, all upper words, and out-of-vector words
    in the corpus.

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
    
    for feature in features:
        features[feature] = sorted(list(set(features[feature])))

    return pd.DataFrame.from_dict(features, orient='index').transpose()

# * tokenizer that replace certain named entities with the entity label
def tokenizer_ent(text):
    """tokenizer that transform text to a list of tokens, can be used with CountVectorizer.
    The tokenizer replaces certain types of named entity with the entity label.

    Args:
        text (string): original text to tokenize

    Returns:
        List of string: List of resulting tokens
    """
    doc = nlp(text)
    tokens = []
    for token in doc:
        if token.ent_type_ in ('ORG', 'DATE', 'TIME', 'MONEY', 'GPE', 'LOC'):
            if token.ent_iob_ == 'B':
                tokens.append(token.ent_type_)
        elif token.ent_iob_ == 'O' and token.pos_ != 'PUNCT':
            if token.like_url:
                tokens.append('URL')
            elif token.like_email:
                tokens.append('EMAIL')
            elif token.like_num:
                tokens.append('NUM')
            elif token.is_alpha and len(token) > 1 and not token.is_oov:
                tokens.append(token.text)

    return tokens

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
def length_feature(sents, tokenizer):
    """Generate length features for each sentence

    Args:
        sents (List of string): List of sentences
        tokenizer (callable): a tokenizer function that transforms text to list of tokens 

    Returns:
        DataFrame: DataFrame containing original sentences and the length features
    """
    df = pd.DataFrame({'sents': sents})
    tokens = df['sents'].map(tokenizer)
    df['word_count'] = tokens.map(len)
    df['char_count'] = tokens.map(lambda x: sum(len(token) for token in x))
    df['avg_word_length'] = df['char_count'] / df['word_count']
    df['unique_count'] = tokens.map(lambda x: len(set(x)))
    df['unique_vs_words'] = df['unique_count'] / df['word_count']
    return df

# * Linguistic Analysis
def count_pos_tag(doc, tag, pos):
    if tag:
        return sum(token.tag_ == tag for token in doc)
    if pos:
        return sum(token.pos_ == pos for token in doc)


def pos_feature(sents):
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
    df = pd.DataFrame({'sents': sents})
    blobs = df['sents'].map(TextBlob)
    df['polarity'] = blobs.map(lambda t: t.sentiment.polarity)
    df['subjectivity'] = blobs.map(lambda t: t.sentiment.subjectivity)
    return df


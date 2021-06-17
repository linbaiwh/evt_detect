from ..context import evt_detect
import pytest

import evt_detect.features.nlp_features as nlp_feat


def test_sentencizer():
    assert "EDGAR_sentencizer" in nlp_feat.nlp.pipe_names
    corpus = """(b) "Cause" with respect to an Eligible Employee means a termination of employment based on a finding by the Employer, acting in good faith based upon the information then known to the Employer, of one of the following: (i) Eligible Employee's gross neglect of, or willful failure or refusal to timely perform, not caused by Eligible Employee's physical or mental disability and not solely based on Eligible Employee's failure to accomplish any particular budgeted goal, the material duties of Eligible Employee's employment following written notice and a reasonable opportunity (not to exceed 30 days) to cure, if such neglect, failure or refusal is capable of being cured; (ii) Eligible Employee's material breach of the terms of any offer letter or employment agreement between the Eligible Employee and his or her Employer, or any other agreement (including the confidentiality agreement and/or proprietary information agreement entered into in connection with Eligible Employee's employment) by and between the Eligible Employee and the Employer which causes demonstrable injury to the Employer provided that Eligible Employee has received written notice of the breach and a reasonable opportunity (not to exceed 30 days) to cure, if such breach is capable of being cured; or (iii) Eligible Employee's commission of, or plea of guilty or nolo contender to, a crime involving moral turpitude, dishonesty, fraud or unethical business conduct, or any felony."""
    doc = nlp_feat.nlp(corpus)
    sent_starts = [token for token in doc if token.is_sent_start == True]
    print(sent_starts)
    assert len(sent_starts) == 4
    

def test_ent_ruler():
    assert "entity_ruler" in nlp_feat.nlp.pipe_names
    ticker = """(NASDAQ: DAIO)"""
    doc = nlp_feat.nlp(ticker)
    assert doc.ents[0].label_ == 'TICKER'
    court = """the Bankruptcy Court"""
    doc = nlp_feat.nlp(court)
    assert doc.ents[0].label_ == 'GOV'

def test_valid_sent():
    sents_invalid = [
    "$433 $365 $440 $425 42" ,
    """Allowed” means:""",
    "(b) Termination Events."
    ]
    for sent in sents_invalid:
        doc = nlp_feat.nlp(sent)
        assert nlp_feat.valid_sent(doc) == False

corpus_list =["""SANTA CLARA, Calif., September 20, 2012 Palo Alto Networks (NYSE: PANW), the network security company, today announced the appointment of John Donovan to its Board of Directors. Palo Alto Networks is the network security company. Its innovative platform allows enterprises, service providers, and government entities to secure their networks and safely enable the increasingly complex and rapidly growing number of applications running on their networks. The core of Palo Alto Networks' platform is its Next-Generation Firewall, which delivers application, user, and content visibility and control integrated within the firewall through its proprietary hardware and software architecture. Palo Alto Networks' products and services can address a broad range of network security requirements, from the data center to the network perimeter, as well as the distributed enterprise, which includes branch offices and a growing number of mobile devices. Palo Alto Networks' products are used by more than 9,000 customers in over 100 countries. For more information, visit www.paloaltonetworks.com."""
, """Redmond, WA, Thursday -- October 26, 2017 -- Data I/O Corporation (NASDAQ: DAIO), a leading global provider of advanced data programming and security provisioning solutions for flash-memory, flash based microcontrollers and Secure Elements, today announced financial results for the third quarter ended September 30, 2017. “In addition to the strong current performance driven by the automotive electronics market, we are very pleased in the deployment of our first SentriX Security Provisioning System. This system was placed with a leading European programming center." """
, """Dice Holdings, Inc. (NYSE: DHX) is a leading provider of specialized career websites for professional communities, including technology and engineering, financial services, energy, healthcare, and security clearance. Our mission is to help our customers source and hire the most qualified professionals in select and highly skilled occupations, and to help those professionals find the best job opportunities in their respective fields and further their careers. For more than 19 years, we have built our company by providing our customers with quick and easy access to high-quality, unique professional communities and offering those communities access to highly relevant career opportunities and information. Today, we serve multiple markets primarily in North America, Europe, the Middle East, Asia and Australia."""
, """Upon (i) such execution of such Commitment Transfer Supplement, (ii) delivery of an executed copy thereof to the Company and (iii) payment by such Purchasing Bank, such Purchasing Bank shall for all purposes be a Bank party to this Agreement and shall have all the rights and obligations of a Bank under this Agreement, to the same extent as if it were an original party hereto with the Commitment Percentage of the Commitments set forth in such Commitment Transfer Supplement."""
, """Any termination by the Company for Cause, or by the Executive for Good Reason, shall be communicated by Notice of Termination to the other party hereto given in accordance with Section 12(b) of this Agreement."""
]

def test_gen_sents():
    sents_num = [7, 3, 4, 1, 1]
    for i in range(4):
        sents = nlp_feat.gen_sents(corpus_list[i])
        print(sents)
        assert len(sents) == sents_num[i]


@pytest.fixture
def sents_list():
    sents = []
    for corpus in corpus_list:
        sents += nlp_feat.gen_sents(corpus)
    return sents
    

def test_find_entity(sents_list):
    df = nlp_feat.find_word_entity(sents_list)
    print(df.shape)
    print(df.columns)
    assert df.shape[0] > 1
    assert df.shape[1] > 1

def test_match_itemNo():
    itemNo = ['(a)', '(iv)', '(3)', '(xxxi)']
    for item in itemNo:
        doc = nlp_feat.nlp(item)
        assert nlp_feat.match_itemNo(doc[1]) == True
    
    noitems = ['a)', '(NYSE: PWAN), (34)']
    for item in noitems:
        doc = nlp_feat.nlp(item)
        assert nlp_feat.match_itemNo(doc[0]) == False
        assert nlp_feat.match_itemNo(doc[1]) == False

def test_tokenizer_ent(sents_list):
    tokens_0 = nlp_feat.tokenizer_ent(sents_list[0])
    print(tokens_0)
    assert 'DATE' in tokens_0

    tokens_6 = nlp_feat.tokenizer_ent(sents_list[6])
    print(tokens_6)
    assert 'URL' in tokens_6

    tokens_14 = nlp_feat.tokenizer_ent(sents_list[14])
    print(tokens_14)
    assert 'ITEMNUM' in tokens_14
    
    tokens_15 = nlp_feat.tokenizer_ent(sents_list[15])
    print(tokens_15)
    assert 'ITEMNUM' not in tokens_15

def test_entity_feature(sents_list):
    df = nlp_feat.entity_feature(sents_list)
    print(df.iloc[0])
    assert df.shape[0] == len(sents_list)
    assert df.shape[1] == 10

def test_get_top_words(sents_list):
    top_words, words_freq = nlp_feat.get_top_n_words(sents_list, n=5)
    print(top_words)
    print(words_freq)
    assert len(top_words) == 5

    top_words, words_freq = nlp_feat.get_top_n_words(sents_list, n=5, 
    lowercase=False, tokenizer=nlp_feat.tokenizer_ent, ngram_range=(1,2))
    print(top_words)
    print(words_freq)
    assert len(top_words) == 5

def test_length_feature(sents_list):
    df = nlp_feat.length_feature(sents_list, nlp_feat.tokenizer_ent)
    print(df.iloc[0])
    assert df.shape[0] == len(sents_list)
    assert df.shape[1] == 6


def test_count_pos_tag(sents_list):
    doc = nlp_feat.nlp(sents_list[0])
    count_verb = nlp_feat.count_pos_tag(doc, False, 'VERB')
    count_vbd = nlp_feat.count_pos_tag(doc, 'VBD', False)
    assert count_verb == 1
    assert count_vbd == 1

def test_pos_feature(sents_list):
    df = nlp_feat.pos_feature(sents_list)
    print(df.iloc[0])
    assert df.shape[0] == len(sents_list)
    assert df.shape[1] == 7

def test_sentiment_feature(sents_list):
    df = nlp_feat.sentiment_feature(sents_list)
    print(df.iloc[11])
    assert df.shape[0] == len(sents_list)
    assert df.shape[1] == 3
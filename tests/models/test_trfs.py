from ..context import evt_detect
import pytest

import evt_detect.models.trfs as trfs
import evt_detect.features.nlp_features as nlp_feat


corpus_list =["""SANTA CLARA, Calif., September 20, 2012 Palo Alto Networks (NYSE: PANW), the network security company, today announced the appointment of John Donovan to its Board of Directors. Palo Alto Networks is the network security company. Its innovative platform allows enterprises, service providers, and government entities to secure their networks and safely enable the increasingly complex and rapidly growing number of applications running on their networks. The core of Palo Alto Networks' platform is its Next-Generation Firewall, which delivers application, user, and content visibility and control integrated within the firewall through its proprietary hardware and software architecture. Palo Alto Networks' products and services can address a broad range of network security requirements, from the data center to the network perimeter, as well as the distributed enterprise, which includes branch offices and a growing number of mobile devices. Palo Alto Networks' products are used by more than 9,000 customers in over 100 countries. For more information, visit www.paloaltonetworks.com."""
, """Redmond, WA, Thursday -- October 26, 2017 -- Data I/O Corporation (NASDAQ: DAIO), a leading global provider of advanced data programming and security provisioning solutions for flash-memory, flash based microcontrollers and Secure Elements, today announced financial results for the third quarter ended September 30, 2017. “In addition to the strong current performance driven by the automotive electronics market, we are very pleased in the deployment of our first SentriX Security Provisioning System. This system was placed with a leading European programming center." """
, """Dice Holdings, Inc. (NYSE: DHX) is a leading provider of specialized career websites for professional communities, including technology and engineering, financial services, energy, healthcare, and security clearance. Our mission is to help our customers source and hire the most qualified professionals in select and highly skilled occupations, and to help those professionals find the best job opportunities in their respective fields and further their careers. For more than 19 years, we have built our company by providing our customers with quick and easy access to high-quality, unique professional communities and offering those communities access to highly relevant career opportunities and information. Today, we serve multiple markets primarily in North America, Europe, the Middle East, Asia and Australia."""
, """Upon (i) such execution of such Commitment Transfer Supplement, (ii) delivery of an executed copy thereof to the Company and (iii) payment by such Purchasing Bank, such Purchasing Bank shall for all purposes be a Bank party to this Agreement and shall have all the rights and obligations of a Bank under this Agreement, to the same extent as if it were an original party hereto with the Commitment Percentage of the Commitments set forth in such Commitment Transfer Supplement."""
, """Any termination by the Company for Cause, or by the Executive for Good Reason, shall be communicated by Notice of Termination to the other party hereto given in accordance with Section 12(b) of this Agreement."""
]

@pytest.fixture
def sents_list():
    sents = []
    for corpus in corpus_list:
        sents += nlp_feat.gen_sents(corpus)
    return sents


def test_trf_length(sents_list):
    trf = trfs.trf_length(tokenizer=nlp_feat.CR_tokenizer)
    df = trf.fit_transform(sents_list)
    features = trf.get_feature_names()
    print(df.iloc[0])
    print(features)
    assert df.shape[0] == len(sents_list)
    assert df.shape[1] == len(features)


def test_trf_pos(sents_list):
    trf = trfs.trf_pos()
    df = trf.fit_transform(sents_list)
    features = trf.get_feature_names()
    print(df.iloc[0])
    print(features)
    assert df.shape[0] == len(sents_list)
    assert df.shape[1] == len(features)


def test_trf_sentiment(sents_list):
    trf = trfs.trf_sentiment()
    df = trf.fit_transform(sents_list)
    features = trf.get_feature_names()
    print(df.iloc[0])
    print(features)
    assert df.shape[0] == len(sents_list)
    assert df.shape[1] == len(features)
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
    

def test_valid_sent():
    sents_invalid = [
    "$433 $365 $440 $425 42" ,
    """Allowed” means:""",
    "(b) Termination Events."
    ]
    for sent in sents_invalid:
        doc = nlp_feat.nlp(sent)
        assert nlp_feat.valid_sent(doc) == False

def test_gen_sents():
    corpus = """Redmond, WA, Thursday -- October 26, 2017 -- Data I/O Corporation (NASDAQ: DAIO), a leading global provider of advanced data programming and security provisioning solutions for flash-memory, flash based microcontrollers and Secure Elements, today announced financial results for the third quarter ended September 30, 2017. “In addition to the strong current performance driven by the automotive electronics market, we are very pleased in the deployment of our first SentriX Security Provisioning System. This system was placed with a leading European programming center." """
    sents = nlp_feat.gen_sents(corpus)
    print(sents)
import pytest
from joblib import load
import pandas as pd

from ..context import evt_detect, model_folder
import evt_detect.models.model_build as model_build
import evt_detect.features.nlp_features as nlp_feat

filtered_text = [
    """Seagate has issued a Supplemental Financial Information document. The Supplemental Financial Information is available on Seagate's Investors website at www.seagate.com/investors. Seagate management will hold a public webcast today at 6:00 a.m. Pacific Time that can be accessed on its Investors website at www.seagate.com/investors. During today's webcast, the Company will provide an outlook for its fourth fiscal quarter of 2016 including key underlying assumptions. A replay will be available beginning today at approximately 9:00 a.m. Pacific Time at www.seagate.com/investors. Investors and others should note that the Company routinely uses the Investors section of its corporate website to announce material information to investors and the marketplace. While not all of the information that the Company posts on its corporate website is of a material nature, some information could be deemed to be material. Accordingly, the Company encourages investors, the media, and others interested in the Company to review the information that it shares on www.seagate.com. The inclusion of Seagate's website address in this report is intended to be an inactive textual reference only and not an active hyperlink. The information contained in, or that can be accessed through, Seagate's website is not part of this report. The inclusion of Seagate's website address in this press release is intended to be an inactive textual reference only and not an active hyperlink. The information contained in, or that can be accessed through, Seagate's website is not part of this press release.""",
    """Facebook will host a conference call to discuss the results at 2 p.m. PT / 5 p.m. ET today. The live webcast of Facebook's earnings conference call can be accessed at investor.fb.com, along with the earnings press release, financial tables, and slide presentation. Facebook uses the investor.fb.com and newsroom.fb.com websites as well as Mark Zuckerberg's Facebook Page (https://www.facebook.com/zuck) as means of disclosing material non-public information and for complying with its disclosure obligations under Regulation FD.""",
    """On September 29, 2014, Essex Property Trust, Inc. (“Essex” or “the Company”) reported that certain of its computer networks containing personal and proprietary information have been compromised by a cyber-intrusion. Essex has confirmed that evidence exists of exfiltration of data on company systems. The precise nature of the data has not yet been identified and the Company does not presently have any evidence that data belonging to the Company has been misused. After detecting unusual activity, the Company took immediate steps to assess and contain the intrusion and secure its systems. Essex has retained independent forensic computer experts to analyze the impacted data systems and is consulting with law enforcement. The investigation into this cyber-intrusion is ongoing, and the team is working as quickly as possible to identify whether any employee or tenant data may be at risk. When the analysis is complete, Essex will notify any affected parties promptly, as appropriate."""
]

model = load(model_folder / 'CR_Incident_Baseline.joblib')
threshold = 0.640327721789483

def test_model_pred():
    X = nlp_feat.gen_sents(filtered_text[0])
    pos_sents = model_build.model_pred(X, model, threshold)
    assert len(pos_sents) == 0


def test_df_model_pred():
    df = pd.DataFrame({'filtered_text': filtered_text})
    results = model_build.df_model_pred(df, X_col='filtered_text', y_col='Incident',
    model_name='Baseline', model=model, threshold=threshold)
    print(results)
    assert results.shape[1] == 4
    assert results.shape[0] == 1

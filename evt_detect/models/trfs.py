from sklearn.base import BaseEstimator, TransformerMixin

import evt_detect.features.nlp_features as nlp_feat

class trf_length(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.features = None

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = nlp_feat.length_feature(X).drop('sents', axis=1)
        self.features = df.columns
        return df

    def get_feature_names(self):
        return self.features.tolist()


class trf_pos(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.features = None

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = nlp_feat.pos_feature(X).drop('sents', axis=1)
        self.features = df.columns
        return df

    def get_feature_names(self):
        return self.features.tolist()


class trf_sentiment(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.features = None

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = nlp_feat.sentiment_feature(X).drop('sents', axis=1)
        self.features = df.columns
        return df

    def get_feature_names(self):
        return self.features.tolist()
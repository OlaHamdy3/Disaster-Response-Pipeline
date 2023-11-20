import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Text Length Extractor for feature unions
class TextLengthExtractor(BaseEstimator, TransformerMixin):
    '''
    Add the length of the text message as a feature to dataset
    '''
    
    def fit(self, x, y=None):
        return self

    def transform(self, X):
        return pd.DataFrame(X).map(len)
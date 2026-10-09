# ml_utils.py
# Shared code for the breast & prostate cancer models.
# Kept in a real .py file (not inside the notebooks) so that any model saved
# with joblib can be loaded again from anywhere -- including the Streamlit app.

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif, chi2, f_classif


class MultiMethodFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Ranks features by averaging ranks from Mutual Information, Chi-Squared,
    and ANOVA F-test, then keeps the top-k.

    Being a proper sklearn transformer (fit/transform), this can sit inside
    a Pipeline: `fit` only ever sees the training fold, so ranking never
    peeks at validation/test data.
    """
    def __init__(self, k=6):
        self.k = k

    def fit(self, X, y):
        X = pd.DataFrame(X)
        feature_names = list(X.columns)

        mi_scores = mutual_info_classif(X, y, random_state=42)
        mi_rank = pd.Series(mi_scores, index=feature_names).rank(ascending=False)

        chi2_scores, _ = chi2(X, y)
        chi2_rank = pd.Series(chi2_scores, index=feature_names).rank(ascending=False)

        f_scores, _ = f_classif(X, y)
        f_rank = pd.Series(f_scores, index=feature_names).rank(ascending=False)

        avg_rank = (mi_rank + chi2_rank + f_rank) / 3
        self.feature_importance_ = avg_rank.sort_values()
        self.top_features_ = self.feature_importance_.head(self.k).index.tolist()
        self.feature_names_in_ = feature_names
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.feature_names_in_) if not hasattr(X, 'columns') else X
        return X[self.top_features_]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.top_features_)
# encoding.py
# ------------------------------------------------------------------
# TRANSLATOR: converts a person's plain questionnaire answers into the
# exact numeric features each model was trained on.
#
#  - BREAST model was trained on ALREADY-preprocessed data (MinMax-scaled
#    + label-encoded). So we reproduce that SAME preprocessing here at
#    prediction time. This is called "train/serve parity". The scaling
#    constants below come straight from the training data's min/max.
#  - PROSTATE model has a MinMaxScaler INSIDE its pipeline, so it scales
#    inputs itself -> we just pass the raw clinical values.
# ------------------------------------------------------------------
import pandas as pd

# Exact features each model's selector expects
BREAST_FEATURES = ['tobacco', 'alcohol', 'hyperplasia', 'breastfeeding',
                   'children', 'age', 'nrelbc', 'menopause', 'agefirst', 'menarche']
PROSTATE_FEATURES = ['PSA_level', 'alkaline_phosphatase', 'testosterone_level',
                     'free_PSA_ratio', 'age', 'BMI']


def encode_breast(a: dict) -> pd.DataFrame:
    """Plain breast answers -> one row of the 10 preprocessed numbers."""
    row = {
        # yes/no -> 1/0
        'tobacco':      1 if a['tobacco'] else 0,
        'alcohol':      1 if a['alcohol'] else 0,
        'hyperplasia':  1 if a['hyperplasia'] else 0,
        'nrelbc':       1 if a['family_history'] else 0,   # any relative w/ breast cancer
        # count 0-5 (5 means "5 or more"), passed through
        'children':     int(a['children']),
        # MinMax-scaled continuous  (constants = training-data min/max)
        'age':          (a['age'] - 20) / 70,
        'menarche':     (a['menarche'] - 8) / 9,
        # a value, or 0 when it doesn't apply
        'menopause':     (a.get('menopause_age', 0)        / 60) if a.get('is_menopausal') else 0.0,
        'agefirst':      (a.get('age_first_birth', 0)      / 46) if a.get('children', 0) > 0 else 0.0,
        'breastfeeding': (a.get('breastfeeding_months', 0) / 72) if a.get('children', 0) > 0 else 0.0,
    }
    return pd.DataFrame([row])[BREAST_FEATURES]


def encode_prostate(a: dict) -> pd.DataFrame:
    """Prostate model scales internally -> assemble the 6 raw values."""
    row = {
        'PSA_level':            a['psa'],
        'alkaline_phosphatase': a['alp'],
        'testosterone_level':   a['testosterone'],
        'free_PSA_ratio':       a['free_psa_ratio'],
        'age':                  a['age'],
        'BMI':                  a['bmi'],
    }
    return pd.DataFrame([row])[PROSTATE_FEATURES]
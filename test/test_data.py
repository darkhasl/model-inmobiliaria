import pandas as pd
import pytest
import sys
import os
import joblib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.make_dataset import prepare_data


def test_data_processing():
    prepare_data()
    
    # Verificar consistencia entre features y preprocesador
    preprocessor = joblib.load('../models/preprocessor.joblib')
    X_test = pd.read_csv('../data/processed/X_test.csv')
    
    # Validar número de features
    expected_features = len(preprocessor.get_feature_names_out())
    assert X_test.shape[1] == expected_features
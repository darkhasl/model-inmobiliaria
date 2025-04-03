import numpy as np
import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.predict import predict
@pytest.mark.usefixtures("run_pipeline")
def test_prediction():
    sample_data = pd.DataFrame({
        'area_m2': [85],
        'habitaciones': [3],
        'banos': [2],
        'ubicacion': ['Centro'],
        'antiguedad': [10],
        'piscina': [1],
        'garaje': [1]
    })
    
    pred = predict(sample_data)
    assert 100_000 < pred[0] < 5_000_000  # Rango realista

'''
def test_prediction():
    sample_data = pd.DataFrame({
        'area_m2': [100],
        'habitaciones': [3],
        'banos': [2],
        'ubicacion': ['Centro'],
        'antiguedad': [5],
        'piscina': [1],
        'garaje': [1]
    })
        
    prediction = predict(sample_data)
    assert isinstance(prediction[0], float)
    assert prediction[0] > 100_000  # Valor mínimo razonable
'''
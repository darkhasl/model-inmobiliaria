import pandas as pd
import pytest
from src.data.make_dataset import prepare_data

def test_data_processing():
    # Ejecutar el pipeline
    prepare_data()
    
    # Verificar que se crearon los archivos
    assert pd.read_csv('data/processed/X_train.csv').shape[0] > 0
    assert pd.read_csv('data/processed/X_test.csv').shape[1] == 8  # N features esperados
    assert 'precio' not in pd.read_csv('data/processed/X_train.csv').columns
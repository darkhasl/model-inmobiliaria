import joblib
import os
import pytest
import pandas as pd
from unittest.mock import call
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.train import train_model

def test_model_training():
    # Ejecutar el entrenamiento real
    train_model()
    
    # Verificar que el modelo se guardó correctamente
    assert os.path.exists('../models/model.joblib')
    
    # Verificar métricas básicas
    model = joblib.load('../models/model.joblib')
    X_train = pd.read_csv('../data/processed/X_train.csv')
    y_train = pd.read_csv('../data/processed/y_train.csv').squeeze()
    
    predictions = model.predict(X_train)
    assert len(predictions) == len(y_train)
    assert predictions.dtype == 'float64'

def test_model_output():
    # Cargar y validar el modelo
    model = joblib.load('../models/model.joblib')
    
    assert hasattr(model, 'predict')
    assert hasattr(model, 'feature_importances_')
    assert len(model.feature_importances_) > 0
    
    # Verificar predicción básica
    sample_data = pd.DataFrame({
        'area_m2': [0.5],
        'habitaciones': [-1.2],
        'banos': [0.3],
        'antiguedad': [1.1],
        'ubicacion_Centro': [1],
        'ubicacion_Este': [0],
        'ubicacion_Norte': [0],
        'ubicacion_Oeste': [0],
        'ubicacion_Sur': [0],
        'piscina': [1],
        'garaje': [1]
    })
    prediction = model.predict(sample_data)
    assert 100_000 < prediction[0] < 5_000_000
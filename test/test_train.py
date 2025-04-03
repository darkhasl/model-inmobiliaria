import joblib
import os
import pytest
from src.models.train import train_model

def test_model_training(mocker):
    # Mock para evitar entrenamiento real
    mock_fit = mocker.patch('sklearn.ensemble.RandomForestRegressor.fit')
    
    train_model()
    
    # Verificar que se guardó el modelo
    assert os.path.exists('models/model.joblib')
    mock_fit.assert_called_once()

def test_model_output():
    model = joblib.load('models/model.joblib')
    assert hasattr(model, 'predict')
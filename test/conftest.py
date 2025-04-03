# tests/conftest.py
import pytest
import sys
import os

# Agregar el directorio raíz al PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from src.make_dataset import prepare_data
from src.train import train_model


            
@pytest.fixture(scope="session", autouse=True)
def run_pipeline():
    # Ejecutar pipeline completo una vez antes de todos los tests
    prepare_data()
    train_model()
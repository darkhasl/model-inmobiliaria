# Código de Evaluación - Modelo de Riesgo de Default en un Banco de Corea
############################################################################

import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import json

def evaluate_model():
    # Cargar datos y modelo
    X_test = pd.read_csv('../data/processed/X_test.csv')
    y_test = pd.read_csv('../data/processed/y_test.csv').squeeze()
    model = joblib.load('../models/model.joblib')
    
    # Predecir
    y_pred = model.predict(X_test)
    
    # Calcular métricas
    metrics = {
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': mean_squared_error(y_test, y_pred, squared=False),
        'r2': r2_score(y_test, y_pred)
    }
    
    # Guardar métricas
    with open('../data/scores/metrics.json', 'w') as f:
        json.dump(metrics, f)
    
    # Mostrar resultados
    print("Métricas de evaluación:")
    for k, v in metrics.items():
        print(f"{k.upper()}: {v:.2f}")

if __name__ == '__main__':
    evaluate_model()
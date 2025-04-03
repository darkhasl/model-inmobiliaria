# Código de Entrenamiento
############################################################################


import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

def train_model():
    # 1. Cargar datos procesados
    X_train = pd.read_csv('../data/processed/X_train.csv')
    y_train = pd.read_csv('../data/processed/y_train.csv').squeeze()
    
    # 2. Asegurar orden correcto de columnas
    preprocessor = joblib.load('../models/preprocessor.joblib')
    ordered_features = preprocessor.get_feature_names_out()
    X_train = X_train[ordered_features]
    
    # 3. Entrenar modelo
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1  # Usar todos los cores
    )
    model.fit(X_train, y_train)
    
    # 4. Guardar modelo
    os.makedirs('../models', exist_ok=True)
    joblib.dump(model, '../models/model.joblib')
    
    # 5. Evaluación en entrenamiento
    y_pred = model.predict(X_train)
    
    metrics = {
        'mae': mean_absolute_error(y_train, y_pred),
        'r2': r2_score(y_train, y_pred)
    }
    
    print(f"MAE en entrenamiento: {metrics['mae']:.2f}")
    print(f"R2 en entrenamiento: {metrics['r2']:.2f}")
    
    return metrics

if __name__ == '__main__':
    train_model()
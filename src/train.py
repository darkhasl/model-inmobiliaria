# Código de Entrenamiento
############################################################################


import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

def train_model():
    # Cargar datos procesados
    X_train = pd.read_csv('../data/processed/X_train.csv')
    y_train = pd.read_csv('../data/processed/y_train.csv').squeeze()
    
    # Entrenar modelo
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Guardar modelo
    joblib.dump(model, '../models/model.joblib')
    
    # Evaluación rápida en entrenamiento
    y_pred = model.predict(X_train)
    mae = mean_absolute_error(y_train, y_pred)
    r2 = r2_score(y_train, y_pred)
    
    print(f"MAE en entrenamiento: {mae:.2f}")
    print(f"R2 en entrenamiento: {r2:.2f}")

if __name__ == '__main__':
    train_model()
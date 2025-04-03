# Código de Scoring - Modelo de Riesgo de Default en un Banco de Corea
############################################################################

import pandas as pd
import joblib

def predict(new_data):
    # Cargar modelo y preprocesador
    model = joblib.load('../models/model.joblib')
    preprocessor = joblib.load('../models/preprocessor.joblib')
    
    # Procesar nuevos datos
    processed_data = preprocessor.transform(new_data)
    
    # Hacer predicción
    prediction = model.predict(processed_data)
    
    return prediction

if __name__ == '__main__':
    # Ejemplo de uso
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
    print(f"Precio predicho: ${pred[0]:,.2f}")
# Código de Scoring -
############################################################################

import pandas as pd
import joblib

def predict(new_data):
    # Cargar componentes
    model = joblib.load('../models/model.joblib')
    preprocessor = joblib.load('../models/preprocessor.joblib')
    
    # Transformar manteniendo la estructura original
    processed_df = preprocessor.transform(new_data)  # Ya es DataFrame con nombres
    
    # Asegurar orden de columnas
    ordered_columns = preprocessor.get_feature_names_out()
    return model.predict(processed_df[ordered_columns])

if __name__ == '__main__':
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

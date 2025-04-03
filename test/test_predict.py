import numpy as np
from src.models.predict import predict

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
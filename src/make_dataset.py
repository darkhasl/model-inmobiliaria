
# Script de Preparación de Datos
###################################

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import joblib

def prepare_data():
    # Cargar datos
    df = pd.read_csv('../data/raw/inmuebles.csv')
    
    # Preprocesamiento
    X = df.drop('precio', axis=1)
    y = df['precio']
    
    # Definir transformadores
    numeric_features = ['area_m2', 'habitaciones', 'banos', 'antiguedad']
    categorical_features = ['ubicacion']
    binary_features = ['piscina', 'garaje']
    '''
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(), categorical_features),
            ('binary', 'passthrough', binary_features)
        ])
    '''
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(sparse_output=False), categorical_features),
            ('binary', 'passthrough', binary_features)
        ],
        verbose_feature_names_out=False
    ).set_output(transform="pandas")  # Mantener como DataFrame
   
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    # Aplicar transformaciones
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Guardar datos procesados y transformador
    joblib.dump(preprocessor, '../models/preprocessor.joblib')
    pd.DataFrame(X_train_processed).to_csv('../data/processed/X_train.csv', index=False)
    pd.DataFrame(X_test_processed).to_csv('../data/processed/X_test.csv', index=False)
    y_train.to_csv('../data/processed/y_train.csv', index=False)
    y_test.to_csv('../data/processed/y_test.csv', index=False)

if __name__ == '__main__':
    prepare_data()
import streamlit as st
import pandas as pd
import re
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import RobustScaler
import joblib  

# Título de la aplicación
st.title("Prediction of Brain Tumor Diagnosis")

st.write("""
This application allows uploading data of metabolite concentrations to predict brain tumor type.
""")

# Cargar el modelo guardado
@st.cache_resource
def load_saved_model():
    model = load_model('mejor_modelo.h5')
    return model

# Cargar el escalador guardado
@st.cache_resource
def load_scaler():
    scaler = joblib.load('robust_scaler.joblib')
    return scaler

model = load_saved_model()
scaler = load_scaler()

# Cargar el archivo CSV de entrada
uploaded_file = st.file_uploader("Upload a CSV file with metabolite concentrations.", type=["csv"])

if uploaded_file is not None:
    # Leer el archivo CSV
    input_data = pd.read_csv(uploaded_file, sep = ";", usecols=range(1,16))

    metabolite_columns = input_data.columns[input_data.columns != 'TYPE']
    
    def clean_number_format(value):
        # Eliminar todas las comas y cualquier punto existente
        cleaned_value = re.sub(r'[,.]', '', value)
        
        # Insertar un punto después del primer dígito
        cleaned_value = f"{cleaned_value[0]}.{cleaned_value[1:]}"
        
        return cleaned_value

    # Aplicar la función de limpieza y luego convertir a float
    input_data[metabolite_columns] = input_data[metabolite_columns].apply(
        lambda col: col.str.replace(',', '.').apply(clean_number_format).astype(float)
    )    

    st.write("Data loaded:")
    st.write(input_data.head())

    # Verificar si las columnas necesarias están presentes
    required_columns = [f"METABOLITE {i}" for i in range(15)]  # Lista de nombres de columnas de metabolitos esperados
    if all(col in input_data.columns for col in required_columns):
        # Preprocesar los datos
        X_input = input_data[required_columns]
        X_scaled = scaler.transform(X_input)
        
        # Realizar las predicciones
        predictions = model.predict(X_scaled)
        
        # Obtener las probabilidades y el diagnóstico recomendado
        diagnosis_classes = ['ASTROCYTOMA', 'GLIOBLASTOMA', 'MENINGIOMA']
        predicted_classes = np.argmax(predictions, axis=1)
        predicted_labels = [diagnosis_classes[i] for i in predicted_classes]
        
        # Crear un DataFrame con los resultados
        results = input_data.copy()
        for idx, diagnosis in enumerate(diagnosis_classes):
            results[f'Probability_{diagnosis}'] = predictions[:, idx]
        results['Diagnosis_Prediction'] = predicted_labels
        
        st.write("Prediction results:")
        st.write(results.head())
        
        # Botón para descargar los resultados
        @st.cache_data
        def convert_df(df):
            return df.to_csv(index=False).encode('utf-8')
        
        csv = convert_df(results)
        
        st.download_button(
            label="Download results as CSV",
            data=csv,
            file_name='prediction_results.csv',
            mime='text/csv',
        )
    else:
        st.error("The CSV file does not contain the required columns. Please check the file format.")

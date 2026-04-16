import streamlit as st
import joblib
import pickle
import numpy as np
import pandas as pd
import psycopg2

# Fetch variables
USER = "postgres.hksmisbyizxycuduheak" #os.getenv("user")
PASSWORD = "Z4r0Pinguino1808"# os.getenv("password")
HOST = "aws-1-us-west-2.pooler.supabase.com" #os.getenv("host")
PORT = "6543" #os.getenv("port")
DBNAME = "postgres" #os.getenv("dbname")

#Cosita nueva del act
def insertar_prediccion(sepal_length, sepal_width, petal_length, petal_width, especie, confianza):
    try:
        conn = psycopg2.connect(
            user=USER,
            password=PASSWORD,
            host=HOST,
            port=PORT,
            dbname=DBNAME
        )
        cur = conn.cursor()

        query = """
        INSERT INTO historial_predicciones
        (sepal_length, sepal_width, petal_length, petal_width, especie_predicha, confianza)
        VALUES (%s, %s, %s, %s, %s, %s);
        """

        cur.execute(query, (
            sepal_length,
            sepal_width,
            petal_length,
            petal_width,
            especie,
            float(confianza)
        ))

        conn.commit()
        cur.close()
        conn.close()

    except Exception as e:
        st.error(f"Error al insertar: {e}")

#Historial
def obtener_historial():
    conn = psycopg2.connect(
        user=USER,
        password=PASSWORD,
        host=HOST,
        port=PORT,
        dbname=DBNAME
    )

    query = """
    SELECT 
        sepal_length,
        sepal_width,
        petal_length,
        petal_width,
        especie_predicha,
        confianza,
        fecha_creacion
    FROM historial_predicciones
    ORDER BY fecha_creacion DESC;
    """

    df = pd.read_sql(query, conn)
    conn.close()
    return df



# Configuración de la página
st.set_page_config(page_title="Predictor de Iris", page_icon="🌸")
# Connect to the database
try:
    connection = psycopg2.connect(
        user=USER,
        password=PASSWORD,
        host=HOST,
        port=PORT,
        dbname=DBNAME
    )
    print("Connection successful!")
    
    # Create a cursor to execute SQL queries
    cursor = connection.cursor()
    
    # Example query
    cursor.execute("SELECT NOW();")
    result = cursor.fetchone()
    print("Current Time:", result)
    # Close the cursor and connection
    cursor.close()
    connection.close()
    print("Connection closed.")

except Exception as e:
    st.write(str(e))



# Función para cargar los modelos
@st.cache_resource
def load_models():
    try:
        model = joblib.load('components/iris_model.pkl')
        scaler = joblib.load('components/iris_scaler.pkl')
        with open('components/model_info.pkl', 'rb') as f:
            model_info = pickle.load(f)
        return model, scaler, model_info
    except FileNotFoundError:
        st.error("No se encontraron los archivos del modelo en la carpeta 'models/'")
        return None, None, None

# Título
st.title("🌸 Predictor de Especies de Iris")

# Cargar modelos
model, scaler, model_info = load_models()

if model is not None:
    # Inputs
    st.header("Ingresa las características de la flor:")
    st.write(result)
    
    sepal_length = st.number_input("Longitud del Sépalo (cm)", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
    sepal_width = st.number_input("Ancho del Sépalo (cm)", min_value=0.0, max_value=10.0, value=3.0, step=0.1)
    petal_length = st.number_input("Longitud del Pétalo (cm)", min_value=0.0, max_value=10.0, value=4.0, step=0.1)
    petal_width = st.number_input("Ancho del Pétalo (cm)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    
    # Botón de predicción
    if st.button("Predecir Especie"):
        # Preparar datos
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        
        # Estandarizar
        features_scaled = scaler.transform(features)
        
        # Predecir
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        
        # Mostrar resultado
        target_names = model_info['target_names']
        predicted_species = target_names[prediction]
        
        st.success(f"Especie predicha: **{predicted_species}**")
        st.write(f"Confianza: **{max(probabilities):.1%}**")
        
        # Mostrar todas las probabilidades
        st.write("Probabilidades:")
        for species, prob in zip(target_names, probabilities):
            st.write(f"- {species}: {prob:.1%}")
            
        insertar_prediccion(
            sepal_length,
            sepal_width,
            petal_length,
            petal_width,
            predicted_species,
            max(probabilities)
        )

st.header("📊 Histórico de Predicciones")

historial = obtener_historial()

if not historial.empty:
    st.dataframe(historial, use_container_width=True)
else:
    st.info("No hay predicciones aún.")

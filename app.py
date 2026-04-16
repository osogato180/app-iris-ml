import streamlit as st
import joblib
import pickle
import numpy as np
import pandas as pd
import psycopg2

# =========================
# CONFIGURACIÓN DE LA PÁGINA
# =========================
st.set_page_config(page_title="Predictor de Iris", page_icon="🌸")

# =========================
# CREDENCIALES SUPABASE
# (por ahora aquí, luego van a Secrets)
# =========================
USER = "postgres.hksmisbyizxycuduheak"
PASSWORD = "Z4r0Pinguino1808"
HOST = "aws-1-us-west-2.pooler.supabase.com"
PORT = "6543"
DBNAME = "postgres"

# =========================
# FUNCIONES BASE DE DATOS
# =========================
def get_connection():
    return psycopg2.connect(
        user=USER,
        password=PASSWORD,
        host=HOST,
        port=PORT,
        dbname=DBNAME
    )

def insertar_prediccion(sepal_length, sepal_width, petal_length, petal_width, especie, confianza):
    try:
        conn = get_connection()
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
        st.error(f"Error al insertar en la BD: {e}")

def obtener_historial():
    try:
        conn = get_connection()
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
    except Exception as e:
        st.error(f"Error al obtener histórico: {e}")
        return pd.DataFrame()

# =========================
# CARGAR MODELOS
# =========================
@st.cache_resource
def load_models():
    model = joblib.load('components/iris_model.pkl')
    scaler = joblib.load('components/iris_scaler.pkl')
    with open('components/model_info.pkl', 'rb') as f:
        model_info = pickle.load(f)
    return model, scaler, model_info

# =========================
# INTERFAZ STREAMLIT
# =========================
st.title("🌸 Predictor de Especies de Iris")

model, scaler, model_info = load_models()

st.header("Ingresa las características de la flor")

sepal_length = st.number_input("Longitud del Sépalo (cm)", 0.0, 10.0, 5.0, 0.1)
sepal_width = st.number_input("Ancho del Sépalo (cm)", 0.0, 10.0, 3.0, 0.1)
petal_length = st.number_input("Longitud del Pétalo (cm)", 0.0, 10.0, 4.0, 0.1)
petal_width = st.number_input("Ancho del Pétalo (cm)", 0.0, 10.0, 1.0, 0.1)

# =========================
# PREDICCIÓN
# =========================
if st.button("Predecir Especie"):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]

    target_names = model_info['target_names']
    predicted_species = target_names[prediction]
    confianza = max(probabilities)

    st.success(f"🌼 Especie predicha: **{predicted_species}**")
    st.write(f"Confianza: **{confianza:.1%}**")

    st.subheader("Probabilidades")
    for species, prob in zip(target_names, probabilities):
        st.write(f"- {species}: {prob:.1%}")

    # GUARDAR EN SUPABASE
    insertar_prediccion(
        sepal_length,
        sepal_width,
        petal_length,
        petal_width,
        predicted_species,
        confianza
    )

# =========================
# HISTÓRICO
# =========================
st.header("📊 Histórico de Predicciones")

historial = obtener_historial()

if not historial.empty:
    st.dataframe(historial, use_container_width=True)
else:
    st.info("No hay predicciones registradas aún.")

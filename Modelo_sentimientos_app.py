from click import prompt
import numpy as np
import pickle
import re
import requests
import streamlit as st
## import tensorflow as tf
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

#
# Configuración de la página, titulo, icono y layout (panel centrado)
st.set_page_config(
    page_title="Análisis de Sentimientos IA - Omar Velez ",
    page_icon="💌",
    layout="centered",
)


#
# Funciones de utilidad para cargar modelo los tokenizadores y preprocesadores una sola vez y almacenarlos en caché
@st.cache_resource
def cargar_modelo_y_preprocesadores():
    try:
        # Cargar modelo
        modelo = load_model("modelo_sentimientos.keras")

        # Cargar tokenizer
        with open("tokenizer_sentimientos.pkl", "rb") as f:
            tokenizer = pickle.load(f)

        # Cargar label encoder
        with open("label_encoder_sentimientos.pkl", "rb") as f:
            label_encoder = pickle.load(f)

        return modelo, tokenizer, label_encoder
    except Exception as e:
        st.error(
            f"No se pudo cargar el modelo, por favor procesar nuevamente Sentimiento.ipynb: {str(e)}"
        )
        return None, None, None


#
# Función para limpiar y preprocesar el texto
# Elimina caracteres especiales, convierte a minúsculas y pone espacios simples
def limpiar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r"[^a-záéíóúüñ\s]", "", texto)
    texto = " ".join(texto.split())
    return texto


#
# Función para predecir el sentimiento
# Devuelve el sentimiento, la confianza y las probabilidades de cada clase
def predecir_sentimiento(texto, modelo, tokenizer, label_encoder, max_length=50):
    # Limpiar el texto
    texto_limpio = limpiar_texto(texto)

    # Convertir a secuencia
    secuencia = tokenizer.texts_to_sequences([texto_limpio])
    secuencia_padded = pad_sequences(secuencia, maxlen=max_length, padding="post")

    # Hacer predicción
    prediccion = modelo.predict(secuencia_padded, verbose=0)

    # Obtener la clase con mayor probabilidad
    clase_predicha = np.argmax(prediccion[0])
    confianza = np.max(prediccion[0])

    # Decodificar la etiqueta
    sentimiento = label_encoder.inverse_transform([clase_predicha])[0]

    return sentimiento, confianza, prediccion[0]

##
## Función para llamar a la API de DeepSeek
# Devuelve la respuesta del modelo
def analizar_sentimiento_deepseek(textoPrompt):
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "deepseek-chat",  # O "deepseek-reasoner" para el modelo de razonamiento
        "messages": [
            {"role": "system", "content": "Eres un asistente útil."},
            {"role": "user", "content": textoPrompt},
        ],
        "stream": False,  # Establece a True si quieres streaming de respuesta
    }

    try:
        response = requests.post(
            f"{BASE_URL}/chat/completions", json=payload, headers=headers
        )
        response.raise_for_status()  # Lanza una excepción para códigos de estado de error

        data = response.json()
        # La respuesta del asistente suele estar en data['choices'][0]['message']['content']
        return data["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        print(f"Error al llamar a la API: {e}")
        return None


# Carga la clave API desde la variable de entorno
API_KEY = "sk-e13a9000354e4f4daea601a368958b3c"
BASE_URL = "https://api.deepseek.com"  # O la URL de tu región si es diferente

#
# Cargar modelo primera vez
modelo, tokenizer, label_encoder = cargar_modelo_y_preprocesadores()

#
# Interfaz principal de la pagina
st.title("💌 Análisis de Sentimientos con IA")
st.write("Analiza el sentimiento de parrafos, desarrollado por Omar Velez")

#
# Si el modelo no se cargo
if modelo is None:
    st.error("❌ Error al cargar el modelo previamente procesado.")
    st.write("Archivos Pendientes:")
    st.write("- `modelo_sentimientos.keras`")
    st.write("- `tokenizer_sentimientos.pkl`")
    st.write("- `label_encoder_sentimientos.pkl`")
else:
    # Input del usuario
    col1, col2 = st.columns([2, 1])

    with col1:
        user_input = st.text_area(
            "Ingresa tu texto aquí:", height=150, placeholder="Que frase eres hoy?"
        )

    # Análisis
    if st.button("Analizar", type="primary") and user_input:
        with st.spinner("Analizando sentimiento..."):
            start_time = time.time()

            # Predecir
            sentimiento, confianza, probabilidades = predecir_sentimiento(
                user_input, modelo, tokenizer, label_encoder
            )

            processing_time = time.time() - start_time

            promptDeepseek = "analiza la siguiente frase, si es positiva, negativa o neutral, informando ademas una estadística de la precisión de tu análisis, respondiendo solo una palabra y el porcentaje de la precision. La frase es: '"+ user_input + "'"    
            respuestaDeepseek = analizar_sentimiento_deepseek(promptDeepseek)

            # Resultados
            st.subheader("🇨🇳 Resultados del Análisis Deepseek", divider=True)

            st.text_area(
                "Deepseek tiene el siguiente analisis:",
                height=10,
                value=respuestaDeepseek,
            )

            st.subheader("📊 Resultados del Análisis Modelo Personalizado", divider=True)

            col1, col2, col3 = st.columns(3)

            with col1:
                # Determinar emoji y color según sentimiento
                if sentimiento == "positivo":
                    emoji = "😊"
                    color = "green"
                elif sentimiento == "negativo":
                    emoji = "😢"
                    color = "red"
                else:
                    emoji = "😐"
                    color = "gray"

                st.metric(
                    label="Sentimiento Detectado",
                    value=f"{emoji} {sentimiento.title()}",
                )

            with col2:
                st.metric(label="Confianza", value=f"{confianza * 100:.1f}%")

            with col3:
                st.metric(
                    label="Tiempo de Procesamiento", value=f"{processing_time:.3f}s"
                )

            # Gráfico de probabilidades
            st.subheader("📈 Grafico de las Probabilidades")
            
            col1, col2 = st.columns([2, 1])

            with col1:
                import matplotlib.pyplot as plt

                labels = ["Negativo", "Neutro", "Positivo"]
                colors = ["#FF6B6B", "#95A5A6", "#2ECC71"]

                fig, ax = plt.subplots(figsize=(8, 4))
                bars = ax.bar(labels, probabilidades, color=colors, alpha=0.7)
                ax.set_ylabel("Probabilidad")
                ax.set_title("Distribución de Probabilidades por Sentimiento")
                ax.set_ylim(0, 1)

                # Añadir valores en las barras
                for bar, prob in zip(bars, probabilidades):
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height + 0.01,
                        f"{prob:.3f}",
                        ha="center",
                        va="bottom",
                    )

                st.pyplot(fig)

            with col2:
                st.write("**Detalles estadistico :**")
                st.write(f"• **Negativo:** {probabilidades[0]:.3f}")
                st.write(f"• **Neutro:** {probabilidades[1]:.3f}")
                st.write(f"• **Positivo:** {probabilidades[2]:.3f}")

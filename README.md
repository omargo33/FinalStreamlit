# 💌 Análisis de Sentimientos con Redes Neuronales

**Proyecto Final - Módulo: Fundamentos de Inteligencia Artificial**  
**Maestría en Ciencia de Datos**  
**Autor:** Omar Vélez  
**Fecha:** Septiembre 2025

## 📋 Descripción del Proyecto

Este proyecto implementa un sistema de análisis de sentimientos utilizando redes neuronales con arquitectura LSTM (Long Short-Term Memory). El sistema clasifica texto en tres categorías: **Positivo**, **Negativo** y **Neutro**, y está diseñado para analizar el clima social en redes sociales y comentarios de publicaciones.

### 🎯 Motivación

La motivación surge de la necesidad de analizar el clima social expresado en redes sociales, inspirado en servicios como [Golden Social Suite](https://www.gosocialsuite.com/es) que brindan análisis de sentimiento para empresas y organizaciones.

## 🚀 Características Principales

- **Red Neuronal LSTM**: Arquitectura especializada en procesamiento de secuencias de texto
- **Comparación con IA Comercial**: Contraste con API de DeepSeek para validación
- **Interfaz Web Interactiva**: Aplicación Streamlit para análisis en tiempo real
- **Visualización de Resultados**: Gráficos de probabilidades y métricas detalladas

## 📁 Estructura del Proyecto

```
FinalStreamlit/
├── Sentimiento.ipynb              # Notebook principal de entrenamiento
├── Modelo_sentimientos_app.py     # Aplicación Streamlit
├── modelo_sentimientos.keras      # Modelo entrenado
├── tokenizer_sentimientos.pkl     # Tokenizador guardado
├── label_encoder_sentimientos.pkl # Codificador de etiquetas
└── README.md                      # Este archivo
```

## 🛠️ Tecnologías Utilizadas

- **Python 3.12**
- **TensorFlow/Keras**: Para la red neuronal
- **Streamlit**: Interfaz web interactiva
- **NLTK**: Procesamiento de lenguaje natural
- **Scikit-learn**: Preprocesamiento y métricas
- **Pandas & NumPy**: Manipulación de datos
- **Matplotlib**: Visualización

## 📊 Arquitectura del Modelo

La red neuronal implementa la siguiente arquitectura:

1. **Capa Embedding**: Representación vectorial de palabras (50 dimensiones)
2. **Capa LSTM**: 32 unidades para capturar patrones secuenciales
3. **Capas Dense**: 
   - Primera capa: 64 neuronas (ReLU + Dropout 0.3)
   - Segunda capa: 32 neuronas (ReLU + Dropout 0.3)
4. **Capa de Salida**: 3 neuronas (Softmax) para clasificación multiclase

## 🎯 Resultados del Modelo

- **Precisión en Entrenamiento**: 43.75%
- **Precisión en Prueba**: 50.00%
- **Diferencia (Overfitting)**: -6.25%

### 📈 Análisis de Resultados

El modelo muestra resultados modestos debido a:
- **Dataset limitado**: Conjunto de datos pequeño para entrenamiento
- **Datos desbalanceados**: Distribución irregular entre las clases
- **Necesidad de más datos**: Requiere ampliación del conjunto de entrenamiento

## 🚀 Instalación y Uso

### Requisitos Previos

```bash
pip install streamlit tensorflow keras scikit-learn nltk pandas numpy matplotlib requests
```

### Ejecución de la Aplicación

```bash
streamlit run Modelo_sentimientos_app.py
```

### Uso del Notebook

Abre [`Sentimiento.ipynb`](FinalStreamlit/Sentimiento.ipynb) en Jupyter para:
- Entrenar el modelo desde cero
- Experimentar con hiperparámetros
- Generar nuevos archivos del modelo

## 📱 Funcionalidades de la Aplicación

### Interfaz Principal
- **Entrada de Texto**: Campo para ingresar frases a analizar
- **Análisis Dual**: Comparación entre modelo propio y DeepSeek AI
- **Métricas en Tiempo Real**: Confianza y tiempo de procesamiento

### Visualizaciones
- **Gráfico de Probabilidades**: Distribución por sentimiento
- **Estadísticas Detalladas**: Valores numéricos de cada clase
- **Indicadores Visuales**: Emojis y colores según el sentimiento

## 🔮 Funciones Principales

### Preprocesamiento
```python
def limpiar_texto(texto):
    # Convierte a minúsculas y elimina caracteres especiales
    # Normaliza espacios en blanco
```

### Predicción
```python
def predecir_sentimiento(texto, modelo, tokenizer, label_encoder):
    # Procesa el texto y genera predicciones
    # Retorna sentimiento, confianza y probabilidades
```

## 🎯 Propuesta de Valor

### Aplicaciones Comerciales
- **Análisis de Redes Sociales**: Monitoreo de comentarios y publicaciones
- **Gestión de Reputación**: Seguimiento de percepción pública
- **Marketing Digital**: Evaluación de campañas publicitarias
- **Atención al Cliente**: Análisis de feedback de usuarios

### Escalabilidad
- **Web Scraping**: Integración con herramientas de recopilación automática
- **Tiempo Real**: Procesamiento instantáneo de grandes volúmenes
- **Multiplataforma**: Adaptable a diferentes redes sociales

## 📈 Mejoras Futuras

### Datos
- [ ] Ampliar el dataset con más ejemplos
- [ ] Balancear las clases de sentimientos
- [ ] Incluir expresiones coloquiales y modismos

### Modelo
- [ ] Optimizar hiperparámetros con Grid Search
- [ ] Implementar técnicas de aumento de datos
- [ ] Explorar modelos pre-entrenados (BERT, RoBERTa)

### Funcionalidades
- [ ] Análisis de emociones específicas
- [ ] Detección de sarcasmo e ironía
- [ ] Soporte multiidioma

## 📖 Referencias y Recursos

- [NLTK Documentation](https://www.nltk.org/)
- [TensorFlow/Keras](https://www.tensorflow.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Golden Social Suite](https://www.gosocialsuite.com/es)

## 📧 Contacto

**Omar Vélez**  
Estudiante de Maestría en Ciencia de Datos  
Proyecto Final - Fundamentos de Inteligencia Artificial

---

## 📄 Licencia

Este proyecto fue desarrollado con fines académicos como parte del programa de Maestría en Ciencia de Datos.
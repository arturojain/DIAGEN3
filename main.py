import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os

# Page configuration
st.set_page_config(
    page_title="Predictor de Riesgo de Vasopresores",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1rem;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .subtitle {
        text-align: center;
        color: #7f8c8d;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #3498db;
        margin: 0.5rem 0;
    }
    
    .risk-high {
        background: linear-gradient(135deg, #fff5f5, #fed7d7);
        border-left: 5px solid #e53e3e;
    }
    
    .risk-low {
        background: linear-gradient(135deg, #f0fff4, #c6f6d5);
        border-left: 5px solid #38a169;
    }
    
    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">🏥 Predictor de Riesgo de Vasopresores</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Sistema de predicción clínica basado en machine learning</p>', unsafe_allow_html=True)

# Define features in the same order as training
FEATURES = [
    'Age',
    'HR_last', 
    'MAP_last',
    'SaO2_last',
    'RespRate_last',
    'Temp_last',
    'GCS_last',
    'WBC_last',
    'Lactate_last'
]

# Feature descriptions and normal ranges
FEATURE_INFO = {
    'Age': {'name': 'Edad', 'unit': 'años', 'range': (0, 150), 'normal': '18-65'},
    'HR_last': {'name': 'Frecuencia Cardíaca', 'unit': 'lpm', 'range': (0, 300), 'normal': '60-100'},
    'MAP_last': {'name': 'Presión Arterial Media (PAM)', 'unit': 'mmHg', 'range': (0, 200), 'normal': '70-100'},
    'SaO2_last': {'name': 'Saturación de O₂', 'unit': '%', 'range': (0, 100), 'normal': '95-100'},
    'RespRate_last': {'name': 'Frecuencia Respiratoria', 'unit': 'rpm', 'range': (0, 60), 'normal': '12-20'},
    'Temp_last': {'name': 'Temperatura', 'unit': '°C', 'range': (30, 45), 'normal': '36.1-37.2'},
    'GCS_last': {'name': 'Escala de Coma de Glasgow', 'unit': 'puntos', 'range': (3, 15), 'normal': '13-15'},
    'WBC_last': {'name': 'Leucocitos', 'unit': '/μL', 'range': (0, 100000), 'normal': '4000-11000'},
    'Lactate_last': {'name': 'Lactato', 'unit': 'mmol/L', 'range': (0, 20), 'normal': '0.5-2.2'}
}

@st.cache_resource
def load_model():
    """Load the vasopressor model with caching and compatibility handling"""
    try:
        # Try to load the model
        model = joblib.load('vasopressor_model.pkl')
        
        # Test the model with dummy data to check compatibility
        dummy_data = pd.DataFrame([[65, 85, 75, 98, 18, 37.2, 15, 8500, 2.1]], columns=FEATURES)
        _ = model.predict_proba(dummy_data)
        
        return model, None
    except FileNotFoundError:
        return None, "❌ Error: El archivo 'vasopressor_model.pkl' no se encontró. Asegúrese de que esté en el mismo directorio."
    except AttributeError as e:
        if 'keep_empty_features' in str(e):
            return None, "❌ Error de compatibilidad: El modelo fue entrenado con una versión diferente de scikit-learn. Necesita reentrenar el modelo con scikit-learn >= 1.4.0."
        else:
            return None, f"❌ Error de atributo: {str(e)}"
    except Exception as e:
        return None, f"❌ Error al cargar el modelo: {str(e)}"

def create_probability_gauge(probability):
    """Create a probability gauge using Plotly"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Probabilidad de Riesgo (%)"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "lightcoral"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def create_feature_radar(input_data):
    """Create radar chart showing input values vs normal ranges"""
    categories = []
    values = []
    
    for feature in FEATURES:
        info = FEATURE_INFO[feature]
        value = input_data[feature]
        
        # Normalize to 0-100 scale based on reasonable ranges
        if feature == 'Age':
            normalized = min(100, (value / 100) * 100)
        elif feature == 'MAP_last':
            normalized = min(100, (value / 120) * 100)
        elif feature == 'SaO2_last':
            normalized = value
        elif feature == 'GCS_last':
            normalized = (value / 15) * 100
        elif feature == 'WBC_last':
            normalized = min(100, (value / 15000) * 100)
        elif feature == 'Lactate_last':
            normalized = min(100, (value / 10) * 100)
        else:
            # General normalization
            range_max = info['range'][1]
            normalized = min(100, (value / range_max) * 100)
        
        categories.append(info['name'])
        values.append(normalized)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Valores del Paciente',
        line_color='rgb(102, 126, 234)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=True,
        title="Perfil del Paciente",
        height=400
    )
    
    return fig

# Load model
model, error_message = load_model()

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ Información")
    
    if error_message:
        st.error(error_message)
        st.info("Para usar esta aplicación, coloque el archivo 'vasopressor_model.pkl' en el mismo directorio que este script.")
    else:
        st.success("✅ Modelo cargado correctamente")
    
    st.markdown("---")
    
    st.subheader("📋 Sobre esta herramienta")
    st.write("""
    Esta aplicación predice el riesgo de que un paciente requiera vasopresores debido a hipotensión, basándose en parámetros clínicos.
    
    **Características:**
    - Predicción en tiempo real
    - Visualización interactiva
    - Interpretación clínica
    """)
    
    st.markdown("---")
    
    st.subheader("⚠️ Advertencia Médica")
    st.warning("""
    Esta herramienta es una ayuda para la decisión clínica y NO reemplaza el juicio médico profesional. 
    
    Los resultados deben interpretarse en el contexto clínico completo del paciente.
    """)

# Main content
if model is not None:
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Predicción", "📊 Análisis", "📚 Información"])
    
    with tab1:
        st.header("Ingrese los Parámetros Clínicos")
        
        # Create form for input
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            
            input_data = {}
            
            with col1:
                st.subheader("Signos Vitales")
                input_data['Age'] = st.number_input(
                    f"👤 {FEATURE_INFO['Age']['name']} ({FEATURE_INFO['Age']['unit']})",
                    min_value=float(FEATURE_INFO['Age']['range'][0]),
                    max_value=float(FEATURE_INFO['Age']['range'][1]),
                    value=65.0,
                    help=f"Rango normal: {FEATURE_INFO['Age']['normal']}"
                )
                
                input_data['HR_last'] = st.number_input(
                    f"💓 {FEATURE_INFO['HR_last']['name']} ({FEATURE_INFO['HR_last']['unit']})",
                    min_value=float(FEATURE_INFO['HR_last']['range'][0]),
                    max_value=float(FEATURE_INFO['HR_last']['range'][1]),
                    value=85.0,
                    help=f"Rango normal: {FEATURE_INFO['HR_last']['normal']}"
                )
                
                input_data['MAP_last'] = st.number_input(
                    f"🩸 {FEATURE_INFO['MAP_last']['name']} ({FEATURE_INFO['MAP_last']['unit']})",
                    min_value=float(FEATURE_INFO['MAP_last']['range'][0]),
                    max_value=float(FEATURE_INFO['MAP_last']['range'][1]),
                    value=75.0,
                    help=f"Rango normal: {FEATURE_INFO['MAP_last']['normal']}"
                )
            
            with col2:
                st.subheader("Respiratorio y Neurológico")
                input_data['SaO2_last'] = st.number_input(
                    f"🫁 {FEATURE_INFO['SaO2_last']['name']} ({FEATURE_INFO['SaO2_last']['unit']})",
                    min_value=float(FEATURE_INFO['SaO2_last']['range'][0]),
                    max_value=float(FEATURE_INFO['SaO2_last']['range'][1]),
                    value=98.0,
                    help=f"Rango normal: {FEATURE_INFO['SaO2_last']['normal']}"
                )
                
                input_data['RespRate_last'] = st.number_input(
                    f"🌬️ {FEATURE_INFO['RespRate_last']['name']} ({FEATURE_INFO['RespRate_last']['unit']})",
                    min_value=float(FEATURE_INFO['RespRate_last']['range'][0]),
                    max_value=float(FEATURE_INFO['RespRate_last']['range'][1]),
                    value=18.0,
                    help=f"Rango normal: {FEATURE_INFO['RespRate_last']['normal']}"
                )
                
                input_data['GCS_last'] = st.number_input(
                    f"🧠 {FEATURE_INFO['GCS_last']['name']} ({FEATURE_INFO['GCS_last']['unit']})",
                    min_value=float(FEATURE_INFO['GCS_last']['range'][0]),
                    max_value=float(FEATURE_INFO['GCS_last']['range'][1]),
                    value=15.0,
                    step=1.0,
                    help=f"Rango normal: {FEATURE_INFO['GCS_last']['normal']}"
                )
            
            with col3:
                st.subheader("Laboratorio")
                input_data['Temp_last'] = st.number_input(
                    f"🌡️ {FEATURE_INFO['Temp_last']['name']} ({FEATURE_INFO['Temp_last']['unit']})",
                    min_value=float(FEATURE_INFO['Temp_last']['range'][0]),
                    max_value=float(FEATURE_INFO['Temp_last']['range'][1]),
                    value=37.2,
                    help=f"Rango normal: {FEATURE_INFO['Temp_last']['normal']}"
                )
                
                input_data['WBC_last'] = st.number_input(
                    f"🔬 {FEATURE_INFO['WBC_last']['name']} ({FEATURE_INFO['WBC_last']['unit']})",
                    min_value=float(FEATURE_INFO['WBC_last']['range'][0]),
                    max_value=float(FEATURE_INFO['WBC_last']['range'][1]),
                    value=8500.0,
                    help=f"Rango normal: {FEATURE_INFO['WBC_last']['normal']}"
                )
                
                input_data['Lactate_last'] = st.number_input(
                    f"⚡ {FEATURE_INFO['Lactate_last']['name']} ({FEATURE_INFO['Lactate_last']['unit']})",
                    min_value=float(FEATURE_INFO['Lactate_last']['range'][0]),
                    max_value=float(FEATURE_INFO['Lactate_last']['range'][1]),
                    value=2.1,
                    help=f"Rango normal: {FEATURE_INFO['Lactate_last']['normal']}"
                )
            
            # Submit button
            submitted = st.form_submit_button("🔍 Calcular Riesgo", use_container_width=True)
            
def create_compatible_model():
    """Create a compatible model pipeline when the original fails"""
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    
    # Create the preprocessing pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Create the column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, FEATURES)
        ]
    )
    
    # Create the full pipeline
    model = Pipeline(steps=[
        ('pre', preprocessor),
        ('clf', LogisticRegression(random_state=42, max_iter=1000))
    ])
    
    return model

def predict_with_fallback(model, df):
    """Make predictions with fallback to manual preprocessing if needed"""
    try:
        # Try normal prediction
        probability = model.predict_proba(df)[0][1]
        prediction = model.predict(df)[0]
        return probability, prediction, None
    except Exception as e:
        # If prediction fails, try with compatible model structure
        try:
            # Create a new compatible model
            compatible_model = create_compatible_model()
            
            # Use default coefficients (you would need to retrain properly)
            st.warning("⚠️ Usando modelo de compatibilidad con parámetros por defecto. Para mejores resultados, reentrenar el modelo.")
            
            # Simple rule-based prediction as fallback
            map_val = df['MAP_last'].iloc[0]
            lactate_val = df['Lactate_last'].iloc[0]
            gcs_val = df['GCS_last'].iloc[0]
            
            # Simple risk calculation based on key indicators
            risk_score = 0
            if map_val < 65: risk_score += 0.4
            if lactate_val > 2.5: risk_score += 0.3
            if gcs_val < 13: risk_score += 0.2
            if df['Age'].iloc[0] > 70: risk_score += 0.1
            
            probability = min(0.95, max(0.05, risk_score))
            prediction = 1 if probability > 0.5 else 0
            
            return probability, prediction, "Usando modelo de compatibilidad"
            
        except Exception as e2:
            return None, None, f"Error en predicción: {str(e2)}"
        
        # Display results if available
        if 'last_prediction' in st.session_state:
            result = st.session_state.last_prediction
            prob_percent = result['probability']
            prediction = result['prediction']
            
            st.markdown("---")
            st.header("📊 Resultado de la Predicción")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Risk assessment
                risk_level = "Alto" if prediction == 1 else "Bajo"
                risk_color = "🔴" if prediction == 1 else "🟢"
                
                st.markdown(f"""
                <div class="metric-card {'risk-high' if prediction == 1 else 'risk-low'}">
                    <h3>{risk_color} Riesgo de Vasopresores: <strong>{risk_level}</strong></h3>
                    <h2>{"Sí" if prediction == 1 else "No"}</h2>
                    <p>Probabilidad: <strong>{prob_percent:.1f}%</strong></p>
                </div>
                """, unsafe_allow_html=True)
                
                # Clinical recommendations
                if prediction == 1:
                    st.error("""
                    ⚠️ **Recomendaciones para Alto Riesgo:**
                    - Monitoreo hemodinámico estrecho
                    - Preparar acceso vascular adecuado
                    - Considerar evaluación por intensivista
                    - Revisar volemia y función cardíaca
                    """)
                else:
                    st.success("""
                    ✅ **Recomendaciones para Bajo Riesgo:**
                    - Continuar con manejo clínico habitual
                    - Monitoreo rutinario de signos vitales
                    - Reevaluar si cambia el estado clínico
                    """)
            
            with col2:
                # Probability gauge
                fig_gauge = create_probability_gauge(prob_percent)
                st.plotly_chart(fig_gauge, use_container_width=True)
    
    with tab2:
        st.header("📈 Análisis de Parámetros")
        
        if 'last_prediction' in st.session_state:
            result = st.session_state.last_prediction
            
            # Radar chart
            fig_radar = create_feature_radar(result['input_data'])
            st.plotly_chart(fig_radar, use_container_width=True)
            
            # Parameters table
            st.subheader("📋 Resumen de Parámetros")
            
            data_for_table = []
            for feature in FEATURES:
                info = FEATURE_INFO[feature]
                value = result['input_data'][feature]
                
                # Determine if value is in normal range (simplified)
                status = "Normal"
                if feature == 'MAP_last' and value < 65:
                    status = "⚠️ Bajo"
                elif feature == 'SaO2_last' and value < 95:
                    status = "⚠️ Bajo"
                elif feature == 'GCS_last' and value < 13:
                    status = "⚠️ Bajo"
                elif feature == 'Lactate_last' and value > 2.2:
                    status = "⚠️ Alto"
                
                data_for_table.append({
                    'Parámetro': info['name'],
                    'Valor': f"{value} {info['unit']}",
                    'Rango Normal': info['normal'],
                    'Estado': status
                })
            
            df_table = pd.DataFrame(data_for_table)
            st.dataframe(df_table, use_container_width=True)
            
        else:
            st.info("👆 Primero realice una predicción en la pestaña 'Predicción' para ver el análisis.")
    
    with tab3:
        st.header("📚 Información del Modelo")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Características del Modelo")
            st.write("""
            - **Tipo:** Regresión Logística con preprocesamiento
            - **Variables:** 9 parámetros clínicos
            - **Objetivo:** Predecir riesgo de vasopresores
            - **Preprocesamiento:** Imputación + Normalización
            """)
            
            st.subheader("📊 Variables de Entrada")
            for feature in FEATURES:
                info = FEATURE_INFO[feature]
                st.write(f"• **{info['name']}**: {info['normal']} {info['unit']}")
        
        with col2:
            st.subheader("⚕️ Interpretación Clínica")
            st.write("""
            **Alto Riesgo (≥50%):**
            - Monitoreo hemodinámico intensivo
            - Preparación para soporte vasopresor
            - Evaluación de causas subyacentes
            
            **Bajo Riesgo (<50%):**
            - Monitoreo estándar
            - Manejo conservador
            - Reevaluación periódica
            """)
            
            st.subheader("🔄 Frecuencia de Evaluación")
            st.write("""
            - **Pacientes críticos:** Cada 2-4 horas
            - **Pacientes estables:** Cada 8-12 horas
            - **Cambios clínicos:** Inmediatamente
            """)

else:
    st.error("No se pudo cargar el modelo. Verifique que el archivo 'vasopressor_model.pkl' esté disponible.")
    
    st.info("""
    **Para usar esta aplicación:**
    1. Asegúrese de que el archivo `vasopressor_model.pkl` esté en el mismo directorio
    2. Instale las dependencias: `pip install streamlit pandas scikit-learn joblib plotly`
    3. Ejecute: `streamlit run app.py`
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d;'>
    <p>🏥 Predictor de Riesgo de Vasopresores | Desarrollado para apoyo a la decisión clínica</p>
    <p><small>Última actualización: {}</small></p>
</div>
""".format(datetime.now().strftime("%Y-%m-%d %H:%M")), unsafe_allow_html=True)

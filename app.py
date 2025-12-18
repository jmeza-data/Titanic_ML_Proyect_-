import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with dark background
st.markdown("""
<style>
    /* Main background gradient - DARK */
    .stApp {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #7e22ce 100%);
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #ff6b6b 0%, #ee5a6f 100%);
        color: white;
        font-weight: bold;
        font-size: 18px;
        padding: 15px;
        border-radius: 12px;
        border: none;
        box-shadow: 0 4px 15px rgba(238, 90, 111, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(238, 90, 111, 0.6);
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: bold;
        color: #FFD700;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: white !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3c72 0%, #2a5298 50%, #7e22ce 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.2);
        border-radius: 8px;
        color: white;
        font-weight: bold;
        padding: 10px 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #ff6b6b 0%, #ee5a6f 100%);
    }
    
    /* Cards with glass effect */
    div[data-testid="column"] {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 20px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Text color */
    p, span, label {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    try:
        with open('ensemble_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('preprocessor.pkl', 'rb') as f:
            prep = pickle.load(f)
        return model, prep
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

model, preprocessor = load_models()

# ==================== LANGUAGE SELECTOR ====================
lang = st.sidebar.selectbox("🌐 Language / Idioma", ["English", "Español"])

# ==================== TRANSLATIONS ====================
if lang == "English":
    txt = {
        "title": "🚢 Titanic Survival Prediction",
        "subtitle": "Advanced Machine Learning System",
        "tab1": "🔮 Prediction",
        "tab2": "📊 Model Metrics",
        "tab3": "📋 About",
        "profile": "👤 Profile",
        "family": "👨‍👩‍👧‍👦 Family",
        "trip": "💰 Trip",
        "ticket_class": "Ticket Class",
        "sex": "Sex",
        "age": "Age",
        "siblings": "Siblings/Spouse",
        "parents": "Parents/Children",
        "family_size": "Family Size",
        "traveling_alone": "Traveling Alone",
        "yes": "Yes",
        "no": "No",
        "fare": "Fare ($)",
        "port": "Port of Embarkation",
        "predict_btn": "🔮 PREDICT SURVIVAL",
        "would_survive": "✅ WOULD SURVIVE",
        "would_not_survive": "❌ WOULD NOT SURVIVE",
        "probability": "Probability",
        "survival_prob": "Survival Probability",
        "metrics_title": "📊 Model Performance Metrics",
        "model_comparison": "🏆 Model Comparison",
        "feature_importance": "📈 Feature Importance (Risk Drivers)",
        "about_title": "📋 About This Project",
        "objective": "Objective",
        "objective_text": "Predict passenger survival on the Titanic using Machine Learning with a risk analysis approach.",
        "dataset": "Dataset",
        "models_used": "Models Used",
        "best_model": "Best Model Performance",
        "key_insights": "Key Insights",
        "author": "Author",
        "footer": "Developed by Jhoan Meza | Ensemble Model (F1: 0.7752) | Bosonit 2024"
    }
else:
    txt = {
        "title": "🚢 Predicción de Supervivencia del Titanic",
        "subtitle": "Sistema Avanzado de Machine Learning",
        "tab1": "🔮 Predicción",
        "tab2": "📊 Métricas del Modelo",
        "tab3": "📋 Acerca de",
        "profile": "👤 Perfil",
        "family": "👨‍👩‍👧‍👦 Familia",
        "trip": "💰 Viaje",
        "ticket_class": "Clase del Ticket",
        "sex": "Sexo",
        "age": "Edad",
        "siblings": "Hermanos/Cónyuge",
        "parents": "Padres/Hijos",
        "family_size": "Tamaño Familiar",
        "traveling_alone": "Viaja Solo",
        "yes": "Sí",
        "no": "No",
        "fare": "Tarifa ($)",
        "port": "Puerto de Embarque",
        "predict_btn": "🔮 PREDECIR SUPERVIVENCIA",
        "would_survive": "✅ SOBREVIVIRÍA",
        "would_not_survive": "❌ NO SOBREVIVIRÍA",
        "probability": "Probabilidad",
        "survival_prob": "Probabilidad de Supervivencia",
        "metrics_title": "📊 Métricas de Rendimiento del Modelo",
        "model_comparison": "🏆 Comparación de Modelos",
        "feature_importance": "📈 Importancia de Variables (Drivers de Riesgo)",
        "about_title": "📋 Acerca de Este Proyecto",
        "objective": "Objetivo",
        "objective_text": "Predecir la supervivencia de pasajeros del Titanic usando Machine Learning con enfoque de análisis de riesgo.",
        "dataset": "Dataset",
        "models_used": "Modelos Utilizados",
        "best_model": "Mejor Modelo",
        "key_insights": "Insights Clave",
        "author": "Autor",
        "footer": "Desarrollado por Jhoan Meza | Modelo Ensemble (F1: 0.7752) | Bosonit 2024"
    }

# Header with animation
st.title(txt["title"])
st.markdown(f"### {txt['subtitle']}")
st.markdown("---")

# TABS
tab1, tab2, tab3 = st.tabs([txt["tab1"], txt["tab2"], txt["tab3"]])

# ==================== TAB 1: PREDICTION ====================
with tab1:
    st.markdown("## 🎯 Enter Passenger Information")
    st.markdown("")
    
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"### {txt['profile']}")
        pclass = st.selectbox(txt["ticket_class"], [1, 2, 3], help="1 = First Class, 2 = Second Class, 3 = Third Class")
        sex = st.selectbox(txt["sex"], ["male", "female"])
        age = st.slider(txt["age"], 0, 80, 30)

    with col2:
        st.markdown(f"### {txt['family']}")
        sibsp = st.number_input(txt["siblings"], 0, 10, 0)
        parch = st.number_input(txt["parents"], 0, 10, 0)
        family_size = sibsp + parch + 1
        is_alone = 1 if family_size == 1 else 0
        
        st.metric(txt["family_size"], family_size)
        st.metric(txt["traveling_alone"], txt["yes"] if is_alone else txt["no"])

    with col3:
        st.markdown(f"### {txt['trip']}")
        fare = st.number_input(txt["fare"], 0.0, 600.0, 50.0, step=10.0)
        embarked = st.selectbox(txt["port"], ["S", "C", "Q"], 
                                format_func=lambda x: {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}[x])

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button(txt["predict_btn"]):
        # ==================== BUG FIX: Correct DataFrame creation ====================
        # Create DataFrame with only the columns the preprocessor expects
        input_df = pd.DataFrame({
            'Pclass': [pclass],
            'Sex': [sex],
            'Age': [age],
            'SibSp': [sibsp],
            'Parch': [parch],
            'Fare': [fare],
            'Embarked': [embarked]
        })
        
        # Ensure correct data types
        input_df['Pclass'] = input_df['Pclass'].astype(int)
        input_df['Age'] = input_df['Age'].astype(float)
        input_df['SibSp'] = input_df['SibSp'].astype(int)
        input_df['Parch'] = input_df['Parch'].astype(int)
        input_df['Fare'] = input_df['Fare'].astype(float)
        
        try:
            # Transform and predict
            X = preprocessor.transform(input_df)
            prediction = model.predict(X)[0]
            probability = model.predict_proba(X)[0][1]
            
            st.markdown("---")
            st.markdown("## 🎯 Prediction Results")
            
            col_res1, col_res2 = st.columns(2)
            
            with col_res1:
                if prediction == 1:
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, #00b09b, #96c93d); 
                                padding: 30px; border-radius: 15px; text-align: center;
                                box-shadow: 0 8px 32px rgba(0, 176, 155, 0.4);'>
                        <h2 style='color: white; margin: 0;'>{txt['would_survive']}</h2>
                        <h1 style='color: white; font-size: 48px; margin: 10px 0;'>{probability:.1%}</h1>
                        <p style='color: white; margin: 0;'>{txt['probability']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, #ff6b6b, #ee5a6f); 
                                padding: 30px; border-radius: 15px; text-align: center;
                                box-shadow: 0 8px 32px rgba(238, 90, 111, 0.4);'>
                        <h2 style='color: white; margin: 0;'>{txt['would_not_survive']}</h2>
                        <h1 style='color: white; font-size: 48px; margin: 10px 0;'>{(1-probability):.1%}</h1>
                        <p style='color: white; margin: 0;'>{txt['probability']}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col_res2:
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = probability * 100,
                    title = {'text': txt["survival_prob"], 'font': {'size': 24, 'color': 'white'}},
                    delta = {'reference': 50, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
                    gauge = {
                        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
                        'bar': {'color': "green" if prediction == 1 else "red", 'thickness': 0.75},
                        'bgcolor': "rgba(255,255,255,0.2)",
                        'borderwidth': 2,
                        'bordercolor': "white",
                        'steps': [
                            {'range': [0, 30], 'color': 'rgba(255, 107, 107, 0.3)'},
                            {'range': [30, 70], 'color': 'rgba(255, 193, 7, 0.3)'},
                            {'range': [70, 100], 'color': 'rgba(0, 176, 155, 0.3)'}
                        ],
                        'threshold': {
                            'line': {'color': "white", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig.update_layout(
                    height=350,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font={'color': 'white', 'size': 16}
                )
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.info("Please check that your model files are compatible with the input data.")

# ==================== TAB 2: METRICS ====================
with tab2:
    st.markdown(f"## {txt['metrics_title']}")
    st.markdown("---")
    
    # Metrics with better styling
    col1, col2, col3, col4, col5 = st.columns(5)
    
    metrics_data = [
        ("🎯 Accuracy", "0.838", col1),
        ("🔍 Precision", "0.833", col2),
        ("📊 Recall", "0.725", col3),
        ("⚖️ F1 Score", "0.775", col4),
        ("📈 ROC AUC", "0.862", col5)
    ]
    
    for label, value, col in metrics_data:
        with col:
            st.markdown(f"""
            <div style='background: rgba(255, 255, 255, 0.1); 
                        padding: 20px; border-radius: 12px; text-align: center;
                        border: 2px solid rgba(255, 255, 255, 0.2);'>
                <p style='color: white; margin: 0; font-size: 14px;'>{label}</p>
                <h2 style='color: #FFD700; margin: 5px 0; font-size: 32px;'>{value}</h2>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(f"## {txt['model_comparison']}")
    
    comparison_data = {
        'Model': ['🏆 Ensemble', 'Random Forest', 'SVM', 'KNN', 'Logistic Regression', 'Gradient Boosting'],
        'Accuracy': [0.838, 0.832, 0.821, 0.810, 0.804, 0.810],
        'F1 Score': [0.775, 0.766, 0.746, 0.742, 0.733, 0.721],
        'ROC AUC': [0.862, 0.836, 0.840, 0.853, 0.857, 0.822]
    }
    
    df_comparison = pd.DataFrame(comparison_data)
    
    # Create bar chart for model comparison
    fig_comparison = go.Figure()
    
    fig_comparison.add_trace(go.Bar(
        name='Accuracy',
        x=df_comparison['Model'],
        y=df_comparison['Accuracy'],
        marker_color='rgb(102, 126, 234)'
    ))
    
    fig_comparison.add_trace(go.Bar(
        name='F1 Score',
        x=df_comparison['Model'],
        y=df_comparison['F1 Score'],
        marker_color='rgb(238, 90, 111)'
    ))
    
    fig_comparison.add_trace(go.Bar(
        name='ROC AUC',
        x=df_comparison['Model'],
        y=df_comparison['ROC AUC'],
        marker_color='rgb(0, 176, 155)'
    ))
    
    fig_comparison.update_layout(
        barmode='group',
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(255,255,255,0.1)',
        font={'color': 'white', 'size': 12},
        legend={'bgcolor': 'rgba(255,255,255,0.1)', 'font': {'color': 'white'}},
        xaxis={'gridcolor': 'rgba(255,255,255,0.2)', 'color': 'white'},
        yaxis={'gridcolor': 'rgba(255,255,255,0.2)', 'color': 'white'}
    )
    
    st.plotly_chart(fig_comparison, use_container_width=True)
    
    st.markdown("---")
    st.markdown(f"## {txt['feature_importance']}")
    
    # ==================== BUG FIX: Simplified colorbar configuration ====================
    importance_data = {
        'Feature': ['💰 Fare', '👤 Age', '👨 Sex_male', '👩 Sex_female', '🎫 Pclass', '👨‍👩‍👧 FamilySize', '👫 SibSp', '👶 Parch'],
        'Importance': [0.25, 0.22, 0.18, 0.15, 0.10, 0.05, 0.03, 0.02]
    }
    
    df_importance = pd.DataFrame(importance_data)
    
    # Simplified approach - no colorbar to avoid the error
    fig_importance = go.Figure(go.Bar(
        x=df_importance['Importance'],
        y=df_importance['Feature'],
        orientation='h',
        marker=dict(
            color=df_importance['Importance'],
            colorscale='Viridis'
        ),
        text=[f"{val:.2f}" for val in df_importance['Importance']],
        textposition='outside',
        textfont=dict(color='white', size=14)
    ))
    
    fig_importance.update_layout(
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(255,255,255,0.1)',
        font={'color': 'white', 'size': 14},
        xaxis={'title': 'Importance', 'gridcolor': 'rgba(255,255,255,0.2)', 'color': 'white'},
        yaxis={'title': '', 'gridcolor': 'rgba(255,255,255,0.2)', 'color': 'white'}
    )
    
    st.plotly_chart(fig_importance, use_container_width=True)

# ==================== TAB 3: ABOUT ====================
with tab3:
    st.markdown(f"## {txt['about_title']}")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div style='background: rgba(255, 255, 255, 0.1); 
                    padding: 25px; border-radius: 15px;
                    border: 2px solid rgba(255, 255, 255, 0.2);'>
            <h3 style='color: #FFD700;'>🎯 {txt['objective']}</h3>
            <p style='color: white; font-size: 16px;'>{txt['objective_text']}</p>
            
            <h3 style='color: #FFD700; margin-top: 20px;'>📊 {txt['dataset']}</h3>
            <p style='color: white; font-size: 16px;'>
                <strong>Training:</strong> 891 passengers<br>
                <strong>Test:</strong> 418 passengers<br>
                <strong>Features:</strong> 7 main variables
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style='background: rgba(255, 255, 255, 0.1); 
                    padding: 25px; border-radius: 15px;
                    border: 2px solid rgba(255, 255, 255, 0.2);'>
            <h3 style='color: #FFD700;'>🤖 {txt['models_used']}</h3>
            <ul style='color: white; font-size: 16px;'>
                <li>Logistic Regression</li>
                <li>Random Forest</li>
                <li>Gradient Boosting</li>
                <li>Support Vector Machine (SVM)</li>
                <li>K-Nearest Neighbors (KNN)</li>
                <li><strong>⭐ Ensemble Model</strong></li>
            </ul>
            
            <h3 style='color: #FFD700; margin-top: 20px;'>🏆 {txt['best_model']}</h3>
            <p style='color: white; font-size: 16px;'>
                <strong>Ensemble Model</strong><br>
                F1 Score: 0.7752 | ROC AUC: 0.8620
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #1e3c72 0%, #7e22ce 100%); 
                padding: 30px; border-radius: 15px; text-align: center;
                border: 2px solid rgba(255, 255, 255, 0.3);'>
        <h3 style='color: white; margin: 0;'>👤 {txt['author']}</h3>
        <h2 style='color: #FFD700; margin: 10px 0;'>Jhoan Meza</h2>
        <p style='color: white; font-size: 18px; margin: 0;'>Data Scientist | Machine Learning Engineer</p>
        <p style='color: white; font-size: 16px; margin-top: 10px;'>Bosonit 2024</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; padding: 20px;'>
    <p style='color: white; font-size: 14px;'>{txt['footer']}</p>
</div>
""", unsafe_allow_html=True)


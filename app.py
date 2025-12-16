import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide"
)

# CSS
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #ff6b6b 0%, #ee5a6f 100%);
        color: white;
        font-weight: bold;
        font-size: 18px;
        padding: 15px;
        border-radius: 10px;
        border: none;
    }
</style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    with open('ensemble_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('preprocessor.pkl', 'rb') as f:
        prep = pickle.load(f)
    return model, prep

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
        "port": "Port",
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
        "port": "Puerto",
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

# Header
st.title(txt["title"])
st.markdown(f"### {txt['subtitle']}")
st.markdown("---")

# TABS
tab1, tab2, tab3 = st.tabs([txt["tab1"], txt["tab2"], txt["tab3"]])

# ==================== TAB 1: PREDICTION ====================
with tab1:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"### {txt['profile']}")
        pclass = st.selectbox(txt["ticket_class"], [1, 2, 3])
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
        fare = st.number_input(txt["fare"], 0.0, 600.0, 50.0)
        embarked = st.selectbox(txt["port"], ["S", "C", "Q"])

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button(txt["predict_btn"]):
        input_df = pd.DataFrame({
            'Pclass': [pclass],
            'Sex': [sex],
            'Age': [age],
            'SibSp': [sibsp],
            'Parch': [parch],
            'Fare': [fare],
            'Embarked': [embarked],
            'FamilySize': [family_size],
            'IsAlone': [is_alone]
        })
        
        X = preprocessor.transform(input_df)
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0][1]
        
        col_res1, col_res2 = st.columns(2)
        
        with col_res1:
            if prediction == 1:
                st.success(f"### {txt['would_survive']}")
                st.markdown(f"**{txt['probability']}: {probability:.1%}**")
            else:
                st.error(f"### {txt['would_not_survive']}")
                st.markdown(f"**{txt['probability']}: {(1-probability):.1%}**")
        
        with col_res2:
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = probability * 100,
                title = {'text': txt["survival_prob"]},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "green" if prediction == 1 else "red"},
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

# ==================== TAB 2: METRICS ====================
with tab2:
    st.markdown(f"### {txt['metrics_title']}")
    st.markdown("---")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Accuracy", "0.838")
    with col2:
        st.metric("Precision", "0.833")
    with col3:
        st.metric("Recall", "0.725")
    with col4:
        st.metric("F1 Score", "0.775")
    with col5:
        st.metric("ROC AUC", "0.862")
    
    st.markdown("---")
    st.markdown(f"### {txt['model_comparison']}")
    
    comparison_data = {
        'Model': ['Ensemble', 'Random Forest', 'SVM', 'KNN', 'Logistic Regression', 'Gradient Boosting'],
        'Accuracy': [0.838, 0.832, 0.821, 0.810, 0.804, 0.810],
        'F1 Score': [0.775, 0.766, 0.746, 0.742, 0.733, 0.721],
        'ROC AUC': [0.862, 0.836, 0.840, 0.853, 0.857, 0.822]
    }
    
    st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.markdown(f"### {txt['feature_importance']}")
    
    importance_data = {
        'Feature': ['Fare', 'Age', 'Sex_male', 'Sex_female', 'Pclass', 'FamilySize', 'SibSp', 'Parch'],
        'Importance': [0.25, 0.22, 0.18, 0.15, 0.10, 0.05, 0.03, 0.02]
    }
    
    fig_importance = go.Figure(go.Bar(
        x=pd.DataFrame(importance_data)['Importance'],
        y=pd.DataFrame(importance_data)['Feature'],
        orientation='h',
        marker_color='steelblue'
    ))
    fig_importance.update_layout(height=400)
    st.plotly_chart(fig_importance, use_container_width=True)

# ==================== TAB 3: ABOUT ====================
with tab3:
    st.markdown(f"### {txt['about_title']}")
    st.markdown("---")
    
    st.markdown(f"#### 🎯 {txt['objective']}")
    st.markdown(txt['objective_text'])
    
    st.markdown(f"#### 📊 {txt['dataset']}")
    st.markdown("- **Training:** 891 | **Test:** 418")
    
    st.markdown(f"#### 🤖 {txt['models_used']}")
    st.markdown("Logistic Regression, Random Forest, Gradient Boosting, SVM, KNN, **Ensemble**")
    
    st.markdown(f"#### 🏆 {txt['best_model']}")
    st.markdown("**Ensemble** - F1: 0.7752 | ROC AUC: 0.8620")
    
    st.markdown(f"#### 👤 {txt['author']}")
    st.markdown("**Jhoan Meza** - Bosonit 2024")

# Footer
st.markdown("---")
st.caption(txt["footer"])
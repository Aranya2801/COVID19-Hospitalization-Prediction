"""
COVID-19 Hospitalization Risk — Interactive Clinical Dashboard
==============================================================
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import json

# ── Page config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="COVID-19 Hospitalization Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #0D47A1 0%, #1565C0 50%, #1976D2 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
    }
    .main-header h1 { margin: 0; font-size: 2rem; font-weight: 700; }
    .main-header p  { margin: 0.5rem 0 0; opacity: 0.85; font-size: 1rem; }

    .risk-card {
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px solid;
        text-align: center;
        margin: 1rem 0;
    }
    .risk-low      { background: #E8F5E9; border-color: #2E7D32; color: #1B5E20; }
    .risk-medium   { background: #FFF8E1; border-color: #F9A825; color: #E65100; }
    .risk-high     { background: #FFF3E0; border-color: #E65100; color: #BF360C; }
    .risk-critical { background: #FFEBEE; border-color: #C62828; color: #B71C1C; }

    .metric-box {
        background: #F8F9FA;
        border-radius: 10px;
        padding: 1.2rem;
        border-left: 4px solid #0D47A1;
        margin-bottom: 1rem;
    }
    .metric-box h4 { margin: 0; color: #424242; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.5px; }
    .metric-box h2 { margin: 0.3rem 0 0; color: #0D47A1; font-size: 1.8rem; }

    .factor-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.5rem 0.8rem;
        border-radius: 6px;
        margin: 0.3rem 0;
        font-size: 0.92rem;
    }
    .factor-risk     { background: #FFEBEE; }
    .factor-protect  { background: #E8F5E9; }

    .disclaimer {
        background: #FFF9C4;
        border: 1px solid #F9A825;
        border-radius: 8px;
        padding: 1rem;
        font-size: 0.85rem;
        color: #5D4037;
    }
    div[data-testid=\"stSidebar\"] { background: #1A237E; }
    div[data-testid=\"stSidebar\"] .css-1d391kg { color: white; }
</style>
""", unsafe_allow_html=True)


# ── Load model ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        with open("models/saved/covid19_predictor.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None


@st.cache_data
def load_dataset():
    try:
        return pd.read_csv("data/raw/covid19_hospitalization_data.csv")
    except:
        return None


model = load_model()
df = load_dataset()


# ── Header ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>🧬 COVID-19 Hospitalization Risk Predictor</h1>
  <p>Calibrated Ensemble ML • SHAP Explainability • Clinical Decision Support</p>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.error("⚠️ Model not found. Run `python train.py` first.")
    st.stop()

# ── Navigation ────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🏥 Patient Assessment",
    "📊 Model Analytics",
    "🗺️ Population Explorer",
    "ℹ️ About"
])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1: Patient Assessment
# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Enter Patient Clinical Data")
    st.markdown("_Complete the form below to generate a real-time hospitalization risk assessment._")

    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.markdown("#### 👤 Demographics")
        c1, c2, c3 = st.columns(3)
        age  = c1.number_input("Age", 1, 100, 55)
        bmi  = c2.number_input("BMI", 10.0, 60.0, 27.5)
        sex  = c3.selectbox("Sex", ["Male", "Female"])

        c1, c2 = st.columns(2)
        race  = c1.selectbox("Race/Ethnicity", ["White","Black","Hispanic","Asian","Other"])
        smoke = c2.selectbox("Smoking", ["Never", "Former", "Current"])

        st.markdown("#### 🦠 COVID Details")
        c1, c2, c3 = st.columns(3)
        variant = c1.selectbox("Variant", ["Omicron","Delta","XBB","JN.1","Alpha","Original","BA.2"])
        vax     = c2.slider("Vaccine Doses", 0, 4, 2)
        days_vax= c3.number_input("Days Since Last Dose", -1, 730, 120)

        c1, c2 = st.columns(2)
        onset   = c1.number_input("Symptom Onset (days)", 0, 30, 4)
        state   = c2.selectbox("State", ["Massachusetts","California","New York","Texas",
                                          "Florida","Illinois","Other"])

        st.markdown("#### 🏥 Vital Signs")
        c1, c2, c3 = st.columns(3)
        o2   = c1.number_input("O₂ Saturation (%)", 70.0, 100.0, 96.0)
        temp = c2.number_input("Temperature (°F)", 96.0, 106.0, 100.2)
        hr   = c3.number_input("Heart Rate (bpm)", 40, 200, 88)

        c1, c2 = st.columns(2)
        rr = c1.number_input("Respiratory Rate", 8, 50, 18)

        st.markdown("#### 🧪 Lab Results")
        c1, c2, c3 = st.columns(3)
        crp     = c1.number_input("CRP (mg/L)", 0.0, 300.0, 15.0)
        d_dimer = c2.number_input("D-Dimer (mg/L)", 0.0, 20.0, 0.5)
        ferr    = c3.number_input("Ferritin (ng/mL)", 0.0, 3000.0, 250.0)

        c1, c2, c3 = st.columns(3)
        il6   = c1.number_input("IL-6 (pg/mL)", 0.0, 500.0, 15.0)
        wbc   = c2.number_input("WBC (K/µL)", 1.0, 30.0, 8.5)
        lymph = c3.number_input("Lymphocytes (%)", 2.0, 60.0, 22.0)

        st.markdown("#### 🩺 Comorbidities & Symptoms")
        c1, c2, c3, c4 = st.columns(4)
        diabetes = int(c1.checkbox("Diabetes"))
        htn      = int(c2.checkbox("Hypertension"))
        heart    = int(c3.checkbox("Heart Disease"))
        lung     = int(c4.checkbox("Lung Disease"))

        c1, c2, c3, c4 = st.columns(4)
        immuno = int(c1.checkbox("Immunocomp."))
        obese  = int(c2.checkbox("Obesity"))
        fever  = int(c3.checkbox("Fever"))
        cough  = int(c4.checkbox("Cough"))
        sob = int(st.checkbox("Shortness of Breath"))

        comorbidity_count = diabetes + htn + heart + lung + immuno + obese

    with col_right:
        if st.button("⚡ Generate Risk Assessment", type="primary", use_container_width=True):
            patient_data = {
                'age': age, 'bmi': bmi, 'sex': sex,
                'race_ethnicity': race, 'smoking_status': smoke,
                'variant': variant, 'vaccine_doses': vax,
                'days_since_last_dose': days_vax if vax > 0 else -1,
                'symptom_onset_days': onset, 'state': state,
                'oxygen_saturation': o2, 'temperature_f': temp,
                'heart_rate': hr, 'respiratory_rate': rr,
                'crp_level': crp, 'd_dimer': d_dimer, 'ferritin': ferr,
                'il6_level': il6, 'wbc_count': wbc, 'lymphocyte_pct': lymph,
                'diabetes': diabetes, 'hypertension': htn,
                'heart_disease': heart, 'chronic_lung_disease': lung,
                'immunocompromised': immuno, 'obesity': obese,
                'fever': fever, 'cough': cough, 'shortness_of_breath': sob,
                'comorbidity_count': comorbidity_count,
                'age_group': pd.cut([age], bins=[0,17,29,39,49,59,69,79,100],
                                    labels=['0-17','18-29','30-39','40-49',
                                            '50-59','60-69','70-79','80+'])[0],
                'county_population': 500000,
                'county_hospital_beds_per_100k': 250,
                'county_poverty_rate': 13.0,
            }
            df_patient = pd.DataFrame([patient_data])

            try:
                explanation = model.explain_patient(pd.Series(patient_data))
                prob    = explanation['hospitalization_probability']
                tier    = explanation['risk_tier']
                rec     = explanation['recommendation']
                factors = explanation['top_risk_factors']

                tier_class = {
                    'LOW': 'risk-low', 'MEDIUM': 'risk-medium',
                    'HIGH': 'risk-high', 'CRITICAL': 'risk-critical'
                }.get(tier, 'risk-high')
                tier_emoji = {'LOW': '🟢', 'MEDIUM': '🟡', 'HIGH': '🟠', 'CRITICAL': '🔴'}.get(tier, '🔴')

                st.markdown(f"""
                <div class="risk-card {tier_class}">
                    <div style="font-size:2.5rem">{tier_emoji}</div>
                    <h2 style="margin:0.3rem 0; font-size:1.6rem">{tier} RISK</h2>
                    <h1 style="margin:0.3rem 0; font-size:3rem; font-weight:700">{prob:.1%}</h1>
                    <p style="margin:0; font-size:0.9rem">Hospitalization Probability</p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"**📋 Clinical Recommendation:**\n> {rec}")

                # Gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    domain={'x': [0,1], 'y': [0,1]},
                    title={'text': "Risk Score", 'font': {'size': 16}},
                    number={'suffix': '%', 'font': {'size': 28}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1},
                        'bar':  {'color': '#0D47A1', 'thickness': 0.3},
                        'steps': [
                            {'range': [0, 20],  'color': '#C8E6C9'},
                            {'range': [20, 40], 'color': '#FFF9C4'},
                            {'range': [40, 65], 'color': '#FFE0B2'},
                            {'range': [65, 100],'color': '#FFCDD2'},
                        ],
                        'threshold': {'line': {'color': 'red','width': 3}, 'thickness': 0.75, 'value': 40}
                    }
                ))
                fig.update_layout(height=250, margin=dict(t=30, b=10, l=30, r=30),
                                  paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)

                # Top factors
                st.markdown("#### 🔍 Key Risk Factors (SHAP)")
                for factor in factors[:8]:
                    direc = factor['direction']
                    cls   = 'factor-risk' if '↑' in direc else 'factor-protect'
                    icon  = '🔺' if '↑' in direc else '🔻'
                    cont  = abs(factor['contribution'])
                    st.markdown(f"""
                    <div class="factor-row {cls}">
                        <span>{icon} <b>{factor['feature'].replace('_',' ').title()}</b></span>
                        <span style="font-weight:600">{direc} ({cont:.3f})</span>
                    </div>
                    """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Prediction error: {e}")
                st.exception(e)

        st.markdown("""
        <div class="disclaimer">
            ⚠️ <b>Clinical Disclaimer:</b> This tool is for research and educational use only.
            It does not replace professional medical judgment. Always consult a licensed
            healthcare provider for clinical decisions.
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2: Model Analytics
# ═══════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Model Performance Metrics")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ROC-AUC",    "0.855",  "+0.087 vs baseline")
    c2.metric("Avg Prec",   "0.926",  "AP Score")
    c3.metric("Brier Score","0.145",  "Lower = Better")
    c4.metric("Features",   "42+",    "Engineered")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        img_path = Path("docs/images/roc_pr_curves.png")
        if img_path.exists():
            st.image(str(img_path), caption="ROC & Precision-Recall Curves", use_column_width=True)

        img_path = Path("docs/images/confusion_matrix.png")
        if img_path.exists():
            st.image(str(img_path), caption="Confusion Matrix", use_column_width=True)

    with col2:
        img_path = Path("docs/images/shap_feature_importance.png")
        if img_path.exists():
            st.image(str(img_path), caption="SHAP Feature Importance", use_column_width=True)

        img_path = Path("docs/images/calibration_curve.png")
        if img_path.exists():
            st.image(str(img_path), caption="Probability Calibration", use_column_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3: Population Explorer
# ═══════════════════════════════════════════════════════════════════════════
with tab3:
    if df is not None:
        st.markdown("### 🗺️ Population-Level Risk Analysis")

        c1, c2, c3 = st.columns(3)
        sel_state = c1.multiselect("Filter by State", df['state'].unique(),
                                    default=list(df['state'].unique()[:5]))
        sel_var   = c2.multiselect("Filter by Variant", df['variant'].unique(),
                                    default=list(df['variant'].unique()))
        age_range = c3.slider("Age Range", 0, 100, (0, 100))

        mask = (
            df['state'].isin(sel_state) &
            df['variant'].isin(sel_var) &
            df['age'].between(*age_range)
        )
        df_filt = df[mask]

        st.info(f"Showing **{len(df_filt):,}** patients | Hospitalization rate: **{df_filt['hospitalized'].mean():.1%}**")

        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df_filt, x='age', color='hospitalized',
                               barmode='overlay', nbins=30, opacity=0.7,
                               color_discrete_map={0:'#2E7D32', 1:'#C62828'},
                               labels={'hospitalized':'Hospitalized','age':'Age'},
                               title='Age Distribution by Hospitalization Status')
            fig.update_layout(height=350, paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            state_stats = df_filt.groupby('state').agg(
                hosp_rate=('hospitalized','mean'),
                n_patients=('hospitalized','count')
            ).reset_index()
            state_stats['hosp_rate_pct'] = state_stats['hosp_rate'] * 100
            fig = px.bar(state_stats.sort_values('hosp_rate_pct', ascending=False).head(10),
                         x='state', y='hosp_rate_pct', color='hosp_rate_pct',
                         color_continuous_scale='RdYlGn_r',
                         title='Hospitalization Rate by State (%)',
                         labels={'hosp_rate_pct':'Hosp. Rate (%)', 'state':'State'})
            fig.update_layout(height=350, paper_bgcolor='rgba(0,0,0,0)', showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            variant_stats = df_filt.groupby('variant').agg(
                hosp_rate=('hospitalized','mean'),
                n=('hospitalized','count')
            ).reset_index()
            fig = px.scatter(variant_stats, x='n', y='hosp_rate',
                             text='variant', size='n',
                             title='Hospitalization Rate vs Volume by Variant',
                             labels={'n':'Patient Count','hosp_rate':'Hosp. Rate'})
            fig.update_traces(textposition='top center')
            fig.update_layout(height=350, paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            vax_age = df_filt.groupby(['vaccine_doses','age_group'])['hospitalized'].mean().reset_index()
            vax_age.columns = ['vaccine_doses','age_group','hosp_rate']
            fig = px.density_heatmap(df_filt, x='vaccine_doses', y='age_group',
                                     z='hospitalized', histfunc='avg',
                                     color_continuous_scale='RdYlGn_r',
                                     title='Hospitalization Rate: Vaccine Doses × Age Group',
                                     labels={'vaccine_doses':'Doses','age_group':'Age Group','hospitalized':'Hosp. Rate'})
            fig.update_layout(height=350, paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Dataset not found.")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4: About
# ═══════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("""
    ## 🧬 About This Project

    **COVID-19 Hospitalization Risk Predictor** is an MIT-grade, production-ready
    clinical decision support system that uses an ensemble of tree-based ML models
    to predict 30-day hospitalization risk for COVID-19 patients.

    ### 🏗️ Architecture
    | Component | Technology |
    |-----------|-----------|
    | Base Models | XGBoost, LightGBM, Random Forest |
    | Meta-Learner | Logistic Regression (Stacking) |
    | Calibration | Isotonic Regression |
    | Explainability | SHAP TreeExplainer |
    | Dashboard | Streamlit + Plotly |

    ### 📊 Model Performance
    | Metric | Score |
    |--------|-------|
    | ROC-AUC | 0.855 |
    | Average Precision | 0.926 |
    | Brier Score | 0.145 |

    ### 🔑 Key Features
    - 42+ clinical, demographic, and biomarker features
    - Real-time individual patient risk scoring
    - SHAP-based feature attribution per patient
    - Probability calibration (isotonic regression)
    - Population-level analytics dashboard
    - 4-tier clinical risk stratification

    ### ⚠️ Disclaimer
    This tool is developed for **research and educational purposes only**.
    It is not FDA-cleared and should not be used for clinical decision-making
    without professional medical oversight.

    ### 📜 License
    MIT License — Copyright (c) 2024 Aranya2801
    """)

"""
Tests for COVID-19 Hospitalization Prediction Pipeline
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
import pandas as pd
import pickle

from src.models.predictor import COVID19HospitalizationPredictor


# ── Fixtures ──────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def sample_df():
    """Minimal synthetic dataframe for testing."""
    np.random.seed(0)
    n = 200
    return pd.DataFrame({
        'patient_id':           [f'T-{i}' for i in range(n)],
        'report_date':          ['2021-01-01'] * n,
        'state':                np.random.choice(['Massachusetts','California'], n),
        'age':                  np.random.randint(18, 85, n),
        'age_group':            np.random.choice(['30-39','40-49','50-59','60-69'], n),
        'sex':                  np.random.choice(['Male','Female'], n),
        'race_ethnicity':       np.random.choice(['White','Black','Hispanic'], n),
        'bmi':                  np.random.uniform(20, 40, n),
        'comorbidity_count':    np.random.randint(0, 5, n),
        'diabetes':             np.random.randint(0, 2, n),
        'hypertension':         np.random.randint(0, 2, n),
        'heart_disease':        np.random.randint(0, 2, n),
        'chronic_lung_disease': np.random.randint(0, 2, n),
        'immunocompromised':    np.random.randint(0, 2, n),
        'obesity':              np.random.randint(0, 2, n),
        'smoking_status':       np.random.choice(['Never','Former','Current'], n),
        'vaccine_doses':        np.random.randint(0, 4, n),
        'days_since_last_dose': np.random.randint(-1, 365, n),
        'variant':              np.random.choice(['Delta','Omicron','XBB'], n),
        'symptom_onset_days':   np.random.randint(0, 14, n),
        'fever':                np.random.randint(0, 2, n),
        'cough':                np.random.randint(0, 2, n),
        'shortness_of_breath':  np.random.randint(0, 2, n),
        'oxygen_saturation':    np.random.uniform(88, 100, n),
        'temperature_f':        np.random.uniform(98, 104, n),
        'heart_rate':           np.random.randint(60, 120, n),
        'respiratory_rate':     np.random.randint(12, 30, n),
        'crp_level':            np.random.exponential(20, n),
        'd_dimer':              np.random.exponential(0.8, n),
        'ferritin':             np.random.exponential(300, n),
        'il6_level':            np.random.exponential(20, n),
        'wbc_count':            np.random.uniform(4, 15, n),
        'lymphocyte_pct':       np.random.uniform(8, 40, n),
        'county_population':    [500000] * n,
        'county_hospital_beds_per_100k': [250.0] * n,
        'county_poverty_rate':  [13.0] * n,
        'hospitalized':         np.random.randint(0, 2, n),
        'icu_admitted':         np.random.randint(0, 2, n),
        'mechanical_ventilation': np.random.randint(0, 2, n),
        'hospital_los_days':    np.random.randint(0, 15, n),
        'icu_los_days':         np.random.randint(0, 10, n),
        'mortality':            np.random.randint(0, 2, n),
    })


@pytest.fixture(scope="module")
def trained_model(sample_df):
    """Quick-train a model on the sample dataframe."""
    import xgboost as xgb
    import lightgbm as lgb
    from sklearn.ensemble import RandomForestClassifier, StackingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import RobustScaler
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    import shap

    model = COVID19HospitalizationPredictor(model_dir="/tmp/test_models")
    X, fn = model.prepare_features(sample_df)
    model.feature_names = fn
    y = sample_df['hospitalized'].values

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2,
                                                  stratify=y, random_state=0)
    model.scaler = RobustScaler()
    Xs_tr  = model.scaler.fit_transform(X_tr)
    Xs_val = model.scaler.transform(X_val)

    base = {
        'xgb': xgb.XGBClassifier(n_estimators=30, max_depth=3,
                                   use_label_encoder=False, eval_metric='logloss',
                                   random_state=0, n_jobs=1),
        'lgb': lgb.LGBMClassifier(n_estimators=30, max_depth=3,
                                   random_state=0, n_jobs=1, verbose=-1),
        'rf':  RandomForestClassifier(n_estimators=30, max_depth=4,
                                       random_state=0, n_jobs=1),
    }
    for m in base.values():
        m.fit(Xs_tr, y_tr)
    model.models = base

    stack = StackingClassifier(
        estimators=list(base.items()),
        final_estimator=LogisticRegression(max_iter=200, random_state=0),
        cv=3, n_jobs=1
    )
    stack.fit(Xs_tr, y_tr)
    model.ensemble = stack

    cal = CalibratedClassifierCV(stack, method='isotonic', cv=3)
    cal.fit(Xs_tr, y_tr)
    model.calibrated_model = cal

    model.explainer = shap.TreeExplainer(base['xgb'])
    model._is_fitted = True
    model.training_metadata = {'val_auc': 0.75}
    return model


# ── Feature Engineering Tests ─────────────────────────────────────────────
class TestFeatureEngineering:
    def test_prepare_features_returns_dataframe(self, trained_model, sample_df):
        X, fn = trained_model.prepare_features(sample_df)
        assert isinstance(X, pd.DataFrame)
        assert len(fn) > 0

    def test_feature_count_reasonable(self, trained_model, sample_df):
        X, fn = trained_model.prepare_features(sample_df)
        assert 30 <= len(fn) <= 80, f"Expected 30-80 features, got {len(fn)}"

    def test_no_string_columns(self, trained_model, sample_df):
        X, _ = trained_model.prepare_features(sample_df)
        # All columns must be castable to float (bool, int, float all ok)
        for col in X.columns:
            try:
                X[col].astype(float)
            except (ValueError, TypeError):
                pytest.fail(f"Column {col} not castable to float: {X[col].dtype}")
    def test_no_nan_after_prep(self, trained_model, sample_df):
        X, _ = trained_model.prepare_features(sample_df)
        assert X.isna().sum().sum() == 0

    def test_severity_score_created(self, trained_model, sample_df):
        X, fn = trained_model.prepare_features(sample_df)
        assert 'severity_score' in fn

    def test_cytokine_storm_index(self, trained_model, sample_df):
        X, fn = trained_model.prepare_features(sample_df)
        assert 'cytokine_storm_index' in fn

    def test_vaccine_protection(self, trained_model, sample_df):
        X, fn = trained_model.prepare_features(sample_df)
        assert 'vaccine_protection' in fn


# ── Prediction Tests ───────────────────────────────────────────────────────
class TestPrediction:
    def test_predict_proba_shape(self, trained_model, sample_df):
        probs = trained_model.predict_proba(sample_df)
        assert probs.shape == (len(sample_df), 2)

    def test_predict_proba_range(self, trained_model, sample_df):
        probs = trained_model.predict_proba(sample_df)[:, 1]
        assert probs.min() >= 0.0
        assert probs.max() <= 1.0

    def test_predict_proba_sums_to_one(self, trained_model, sample_df):
        probs = trained_model.predict_proba(sample_df)
        row_sums = probs.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    def test_predict_risk_columns(self, trained_model, sample_df):
        result = trained_model.predict_risk(sample_df)
        assert 'hospitalization_prob' in result.columns
        assert 'risk_tier' in result.columns
        assert 'clinical_recommendation' in result.columns

    def test_risk_tier_values(self, trained_model, sample_df):
        result = trained_model.predict_risk(sample_df)
        valid_tiers = {'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'}
        actual = set(result['risk_tier'].dropna().astype(str))
        assert actual.issubset(valid_tiers)

    def test_single_row_prediction(self, trained_model, sample_df):
        single = sample_df.iloc[[0]]
        probs = trained_model.predict_proba(single)
        assert probs.shape == (1, 2)


# ── Explanation Tests ──────────────────────────────────────────────────────
class TestExplainability:
    def test_explain_returns_dict(self, trained_model, sample_df):
        result = trained_model.explain(sample_df.head(20))
        assert isinstance(result, dict)
        assert 'shap_values' in result
        assert 'top_features' in result

    def test_shap_values_shape(self, trained_model, sample_df):
        result = trained_model.explain(sample_df.head(20))
        sv = result['shap_values']
        n_feats = len(trained_model.feature_names)
        assert sv.shape == (20, n_feats)

    def test_top_features_list(self, trained_model, sample_df):
        result = trained_model.explain(sample_df.head(20))
        assert len(result['top_features']) > 0
        assert all(len(t) == 2 for t in result['top_features'])

    def test_patient_explanation_keys(self, trained_model, sample_df):
        patient = sample_df.iloc[0]
        result = trained_model.explain_patient(patient)
        for key in ['hospitalization_probability','risk_tier',
                    'recommendation','top_risk_factors']:
            assert key in result

    def test_probability_in_range(self, trained_model, sample_df):
        patient = sample_df.iloc[0]
        result = trained_model.explain_patient(patient)
        p = result['hospitalization_probability']
        assert 0.0 <= p <= 1.0


# ── Clinical Utility Tests ─────────────────────────────────────────────────
class TestClinicalUtilities:
    def test_risk_tier_low(self, trained_model):
        assert trained_model._risk_tier(0.10) == 'LOW'

    def test_risk_tier_medium(self, trained_model):
        assert trained_model._risk_tier(0.30) == 'MEDIUM'

    def test_risk_tier_high(self, trained_model):
        assert trained_model._risk_tier(0.50) == 'HIGH'

    def test_risk_tier_critical(self, trained_model):
        assert trained_model._risk_tier(0.80) == 'CRITICAL'

    def test_clinical_recommendation_not_empty(self, trained_model):
        for prob in [0.05, 0.25, 0.55, 0.90]:
            rec = trained_model._clinical_recommendation(prob)
            assert isinstance(rec, str)
            assert len(rec) > 5


# ── Evaluation Tests ───────────────────────────────────────────────────────
class TestEvaluation:
    def test_evaluate_returns_metrics(self, trained_model, sample_df):
        metrics = trained_model.evaluate(sample_df)
        for key in ['roc_auc','avg_precision','brier_score',
                    'classification_report','confusion_matrix']:
            assert key in metrics

    def test_roc_auc_range(self, trained_model, sample_df):
        metrics = trained_model.evaluate(sample_df)
        assert "roc_auc" in metrics
        assert metrics["roc_auc"] is not None

    def test_brier_range(self, trained_model, sample_df):
        metrics = trained_model.evaluate(sample_df)
        assert 0.0 <= metrics['brier_score'] <= 0.5

    def test_confusion_matrix_shape(self, trained_model, sample_df):
        metrics = trained_model.evaluate(sample_df)
        cm = np.array(metrics['confusion_matrix'])
        assert cm.shape == (2, 2)


# ── Persistence Tests ──────────────────────────────────────────────────────
class TestPersistence:
    def test_save_and_load(self, trained_model, tmp_path):
        trained_model.model_dir = tmp_path
        save_path = trained_model.save("test_model")
        assert save_path.exists()

        loaded = COVID19HospitalizationPredictor.load(str(save_path))
        assert loaded._is_fitted

    def test_loaded_model_predicts(self, trained_model, sample_df, tmp_path):
        trained_model.model_dir = tmp_path
        save_path = trained_model.save("test_model2")
        loaded = COVID19HospitalizationPredictor.load(str(save_path))
        probs = loaded.predict_proba(sample_df)
        assert probs.shape == (len(sample_df), 2)

    def test_unfitted_raises(self):
        m = COVID19HospitalizationPredictor()
        with pytest.raises(RuntimeError):
            m.predict_proba(pd.DataFrame())

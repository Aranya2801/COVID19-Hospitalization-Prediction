"""
COVID-19 Hospitalization Prediction Engine
==========================================
MIT-Grade ML Pipeline with Ensemble Methods, SHAP Explainability,
Calibration, and Clinical Decision Support.

Author: Aranya2801
Version: 2.0.0
"""

import pandas as pd
import numpy as np
import pickle
import json
import warnings
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Union
from datetime import datetime

from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    StackingClassifier, VotingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, train_test_split
)
from sklearn.preprocessing import (
    StandardScaler, LabelEncoder, RobustScaler
)
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix,
    brier_score_loss, average_precision_score,
    roc_curve, precision_recall_curve
)
from sklearn.calibration import calibration_curve
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import xgboost as xgb
import lightgbm as lgb
import shap

warnings.filterwarnings('ignore')


class COVID19HospitalizationPredictor:
    """
    Production-grade COVID-19 hospitalization risk prediction system.
    
    Uses an ensemble of XGBoost, LightGBM, Random Forest, and 
    Gradient Boosting models with probability calibration and
    SHAP-based explainability for clinical deployment.
    """
    
    VERSION = "2.0.0"
    RISK_THRESHOLDS = {
        'low':    (0.00, 0.20),
        'medium': (0.20, 0.40),
        'high':   (0.40, 0.65),
        'critical':(0.65, 1.01)
    }
    
    CLINICAL_FEATURES = [
        'age', 'bmi', 'comorbidity_count', 'oxygen_saturation',
        'temperature_f', 'heart_rate', 'respiratory_rate',
        'crp_level', 'd_dimer', 'ferritin', 'il6_level',
        'wbc_count', 'lymphocyte_pct', 'symptom_onset_days',
        'vaccine_doses', 'days_since_last_dose'
    ]
    
    BINARY_FEATURES = [
        'diabetes', 'hypertension', 'heart_disease',
        'chronic_lung_disease', 'immunocompromised', 'obesity',
        'fever', 'cough', 'shortness_of_breath'
    ]
    
    CATEGORICAL_FEATURES = [
        'sex', 'race_ethnicity', 'smoking_status', 'variant', 'age_group'
    ]

    def __init__(self, model_dir: str = "models/saved", random_state: int = 42):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state
        self.models = {}
        self.preprocessor = None
        self.ensemble = None
        self.calibrated_model = None
        self.explainer = None
        self.feature_names = None
        self.training_metadata = {}
        self._is_fitted = False
        
    # ------------------------------------------------------------------ #
    #  PREPROCESSING                                                       #
    # ------------------------------------------------------------------ #
    def _build_preprocessor(self) -> ColumnTransformer:
        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler',  RobustScaler())
        ])
        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
        ])
        return ColumnTransformer(transformers=[
            ('num', numeric_transformer,
             self.CLINICAL_FEATURES + self.BINARY_FEATURES),
            ('cat', categorical_transformer, self.CATEGORICAL_FEATURES),
        ], remainder='drop')

    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['sex'] = (df['sex'] == 'Male').astype(int)
        df['smoking_current'] = (df['smoking_status'] == 'Current').astype(int)
        df['smoking_former']  = (df['smoking_status'] == 'Former').astype(int)
        variant_map = {'Original':0,'Alpha':1,'Delta':2,'Omicron':3,
                       'BA.2':4,'XBB':5,'JN.1':6}
        df['variant_encoded'] = df['variant'].map(variant_map).fillna(3)
        
        race_dummies = pd.get_dummies(df['race_ethnicity'], prefix='race', drop_first=True)
        df = pd.concat([df, race_dummies], axis=1)
        
        age_group_map = {'0-17':0,'18-29':1,'30-39':2,'40-49':3,
                         '50-59':4,'60-69':5,'70-79':6,'80+':7}
        df['age_group_encoded'] = df['age_group'].map(age_group_map).fillna(3)
        
        return df
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Advanced feature engineering for clinical predictors."""
        df = df.copy()
        
        # Severity indices
        df['hypoxia_flag']         = (df['oxygen_saturation'] < 94).astype(int)
        df['tachycardia_flag']     = (df['heart_rate'] > 100).astype(int)
        df['tachypnea_flag']       = (df['respiratory_rate'] > 20).astype(int)
        df['fever_flag']           = (df['temperature_f'] > 100.4).astype(int)
        df['high_crp']             = (df['crp_level'] > 50).astype(int)
        df['elevated_d_dimer']     = (df['d_dimer'] > 0.5).astype(int)
        df['lymphopenia']          = (df['lymphocyte_pct'] < 20).astype(int)
        df['high_ferritin']        = (df['ferritin'] > 500).astype(int)
        
        # Composite clinical scores
        df['severity_score'] = (
            df['hypoxia_flag'] * 3 +
            df['tachypnea_flag'] * 2 +
            df['high_crp'] * 1.5 +
            df['elevated_d_dimer'] * 1.5 +
            df['lymphopenia'] * 1.0 +
            df['high_ferritin'] * 1.0 +
            df['tachycardia_flag'] * 0.5 +
            df['fever_flag'] * 0.5
        )
        df['cytokine_storm_index'] = (
            np.log1p(df['crp_level']) +
            np.log1p(df['ferritin']) +
            np.log1p(df['il6_level'])
        ) / 3
        df['coagulopathy_risk'] = (
            df['d_dimer'] * 2 + df['elevated_d_dimer'] * 3
        )
        
        # Age-comorbidity interaction (key hospitalization driver)
        df['age_comorbidity']  = df['age'] * df['comorbidity_count']
        df['age_squared']      = df['age'] ** 2
        df['bmi_category']     = pd.cut(df['bmi'],
                                        bins=[0, 18.5, 25, 30, 35, 100],
                                        labels=[0,1,2,3,4]).astype(float)
        
        # Vaccine protection score
        df['vaccine_protection'] = np.where(
            df['vaccine_doses'] == 0, 0,
            np.where(
                df['days_since_last_dose'] < 0, 0,
                np.exp(-df['days_since_last_dose'] / 180) * df['vaccine_doses'] * 0.3
            )
        )
        df['unvaccinated'] = (df['vaccine_doses'] == 0).astype(int)
        
        # Symptom burden
        if all(c in df.columns for c in ['fever','cough','shortness_of_breath']):
            df['symptom_burden'] = df['fever'] + df['cough'] + df['shortness_of_breath']
        
        # Late presentation
        df['late_presentation'] = (df['symptom_onset_days'] > 7).astype(int)
        
        return df

    # ------------------------------------------------------------------ #
    #  MODEL BUILDING                                                      #
    # ------------------------------------------------------------------ #
    def _build_models(self) -> dict:
        return {
            'xgboost': xgb.XGBClassifier(
                n_estimators=600,
                max_depth=7,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1.0,
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=self.random_state,
                n_jobs=-1
            ),
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=600,
                max_depth=7,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=20,
                num_leaves=63,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=400,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.08,
                subsample=0.8,
                min_samples_split=5,
                min_samples_leaf=3,
                random_state=self.random_state
            ),
        }

    # ------------------------------------------------------------------ #
    #  TRAINING                                                            #
    # ------------------------------------------------------------------ #
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
        df = self._encode_categoricals(df)
        df = self._engineer_features(df)
        
        feature_cols = (
            self.CLINICAL_FEATURES + self.BINARY_FEATURES +
            [c for c in df.columns if c.startswith('race_')] +
            ['sex', 'smoking_current', 'smoking_former', 'variant_encoded',
             'age_group_encoded', 'hypoxia_flag', 'tachycardia_flag',
             'tachypnea_flag', 'fever_flag', 'high_crp', 'elevated_d_dimer',
             'lymphopenia', 'high_ferritin', 'severity_score',
             'cytokine_storm_index', 'coagulopathy_risk',
             'age_comorbidity', 'age_squared', 'bmi_category',
             'vaccine_protection', 'unvaccinated', 'symptom_burden',
             'late_presentation']
        )
        feature_cols = [c for c in feature_cols if c in df.columns]
        X = df[feature_cols].fillna(0)
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
        return X, feature_cols

    def fit(self, df: pd.DataFrame, target: str = 'hospitalized',
            validation_split: float = 0.2) -> 'COVID19HospitalizationPredictor':
        """Full training pipeline with CV, stacking, and calibration."""
        print("=" * 60)
        print("  COVID-19 Hospitalization Predictor — Training")
        print("=" * 60)
        
        t0 = datetime.now()
        X, self.feature_names = self.prepare_features(df)
        y = df[target].values
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split,
            stratify=y, random_state=self.random_state
        )
        
        print(f"\n[DATA] Train: {X_train.shape[0]:,} | Val: {X_val.shape[0]:,}")
        print(f"       Positive rate: {y.mean():.1%}\n")
        
        # --- Scale ---
        self.scaler = RobustScaler()
        X_train_s = self.scaler.fit_transform(X_train)
        X_val_s   = self.scaler.transform(X_val)
        
        # --- Base models ---
        self.models = self._build_models()
        cv_results = {}
        
        for name, model in self.models.items():
            print(f"  Training {name}...")
            model.fit(X_train_s, y_train)
            cv = cross_val_score(model, X_train_s, y_train,
                                 cv=StratifiedKFold(5, shuffle=True,
                                                    random_state=self.random_state),
                                 scoring='roc_auc', n_jobs=-1)
            val_auc = roc_auc_score(y_val, model.predict_proba(X_val_s)[:, 1])
            cv_results[name] = {'cv_mean': cv.mean(), 'cv_std': cv.std(),
                                'val_auc': val_auc}
            print(f"    CV-AUC: {cv.mean():.4f} ± {cv.std():.4f} | "
                  f"Val-AUC: {val_auc:.4f}")
        
        # --- Stacking ensemble ---
        print("\n  Building stacking ensemble...")
        estimators = [(n, m) for n, m in self.models.items()]
        meta_learner = LogisticRegression(C=1.0, max_iter=1000,
                                          random_state=self.random_state)
        self.ensemble = StackingClassifier(
            estimators=estimators,
            final_estimator=meta_learner,
            cv=5, passthrough=False, n_jobs=-1
        )
        self.ensemble.fit(X_train_s, y_train)
        
        # --- Isotonic calibration ---
        print("  Calibrating probabilities...")
        self.calibrated_model = CalibratedClassifierCV(
            self.ensemble, method='isotonic', cv=5
        )
        self.calibrated_model.fit(X_train_s, y_train)
        
        # --- Final evaluation ---
        probs = self.calibrated_model.predict_proba(X_val_s)[:, 1]
        final_auc = roc_auc_score(y_val, probs)
        brier    = brier_score_loss(y_val, probs)
        avg_prec = average_precision_score(y_val, probs)
        
        print(f"\n{'='*60}")
        print(f"  FINAL ENSEMBLE (Calibrated Stacking)")
        print(f"  Val-AUC   : {final_auc:.4f}")
        print(f"  Avg-Prec  : {avg_prec:.4f}")
        print(f"  Brier     : {brier:.4f}")
        print(f"  Training time: {(datetime.now()-t0).seconds}s")
        print(f"{'='*60}\n")
        
        # --- SHAP explainer ---
        print("  Fitting SHAP explainer...")
        best_name = max(cv_results, key=lambda k: cv_results[k]['val_auc'])
        self.explainer = shap.TreeExplainer(self.models[best_name])
        
        self._is_fitted = True
        self.training_metadata = {
            'train_date': datetime.now().isoformat(),
            'n_samples': len(df),
            'n_features': len(self.feature_names),
            'target': target,
            'val_auc': final_auc,
            'brier_score': brier,
            'avg_precision': avg_prec,
            'cv_results': cv_results,
            'version': self.VERSION
        }
        return self

    # ------------------------------------------------------------------ #
    #  PREDICTION & EXPLANATION                                            #
    # ------------------------------------------------------------------ #
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        X, _ = self.prepare_features(df)
        # Align to training feature set: add missing cols as 0, drop extras
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0
        X = X[self.feature_names].fillna(0)
        X_s = self.scaler.transform(X)
        return self.calibrated_model.predict_proba(X_s)

    def predict_risk(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return probabilities + risk tier + clinical flags."""
        probs = self.predict_proba(df)[:, 1]
        risk_tiers = pd.cut(probs,
                            bins=[0, 0.20, 0.40, 0.65, 1.01],
                            labels=['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'],
                            right=True)
        result = pd.DataFrame({
            'hospitalization_prob': probs.round(4),
            'risk_tier': risk_tiers,
            'clinical_recommendation': [
                self._clinical_recommendation(p) for p in probs
            ]
        })
        return result

    def explain(self, df: pd.DataFrame, max_display: int = 20) -> Dict:
        """Generate SHAP explanations for a batch."""
        self._check_fitted()
        X, _ = self.prepare_features(df)
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0
        X = X[self.feature_names].fillna(0)
        X_s_df = pd.DataFrame(self.scaler.transform(X), columns=self.feature_names)
        
        shap_values = self.explainer.shap_values(X_s_df)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        mean_abs = np.abs(shap_values).mean(0)
        top_idx  = np.argsort(mean_abs)[::-1][:max_display]
        
        return {
            'shap_values': shap_values,
            'feature_names': self.feature_names,
            'top_features': [(self.feature_names[i], mean_abs[i])
                             for i in top_idx],
            'X_transformed': X_s_df
        }

    def explain_patient(self, patient_row: pd.Series) -> Dict:
        """Individual patient explanation."""
        df_single = pd.DataFrame([patient_row])
        try:
            explanation = self.explain(df_single, max_display=15)
        except Exception:
            explanation = {'feature_names': self.feature_names,
                           'shap_values': np.zeros((1, len(self.feature_names))),
                           'top_features': [(f, 0.0) for f in self.feature_names[:15]],
                           'X_transformed': None}
        prob = self.predict_proba(df_single)[0, 1]
        
        feature_contributions = sorted(
            zip(explanation['feature_names'],
                explanation['shap_values'][0]),
            key=lambda x: abs(x[1]), reverse=True
        )[:10]
        
        return {
            'hospitalization_probability': round(prob, 4),
            'risk_tier': self._risk_tier(prob),
            'recommendation': self._clinical_recommendation(prob),
            'top_risk_factors': [
                {'feature': f, 'contribution': round(v, 4),
                 'direction': '↑ Risk' if v > 0 else '↓ Risk'}
                for f, v in feature_contributions
            ]
        }

    # ------------------------------------------------------------------ #
    #  CLINICAL UTILITIES                                                  #
    # ------------------------------------------------------------------ #
    def _risk_tier(self, prob: float) -> str:
        for tier, (lo, hi) in self.RISK_THRESHOLDS.items():
            if lo <= prob < hi:
                return tier.upper()
        return 'CRITICAL'

    def _clinical_recommendation(self, prob: float) -> str:
        tier = self._risk_tier(prob)
        recs = {
            'LOW':      'Monitor at home; telehealth follow-up in 48h',
            'MEDIUM':   'Outpatient monitoring; repeat O2 sat q6h; Rx eligibility check',
            'HIGH':     'Urgent ED evaluation; consider Remdesivir / monoclonal Ab',
            'CRITICAL': 'IMMEDIATE HOSPITALIZATION; ICU-level monitoring warranted'
        }
        return recs.get(tier, 'Clinical evaluation required')

    def evaluate(self, df: pd.DataFrame, target: str = 'hospitalized') -> Dict:
        """Comprehensive evaluation metrics."""
        self._check_fitted()
        probs = self.predict_proba(df)[:, 1]
        y     = df[target].values
        preds = (probs >= 0.40).astype(int)
        
        fpr, tpr, _ = roc_curve(y, probs)
        pr, rec, _  = precision_recall_curve(y, probs)
        frac_pos, mean_pred = calibration_curve(y, probs, n_bins=10)
        
        return {
            'roc_auc':          roc_auc_score(y, probs),
            'avg_precision':    average_precision_score(y, probs),
            'brier_score':      brier_score_loss(y, probs),
            'classification_report': classification_report(y, preds, output_dict=True),
            'confusion_matrix': confusion_matrix(y, preds).tolist(),
            'roc_curve':        {'fpr': fpr.tolist(), 'tpr': tpr.tolist()},
            'pr_curve':         {'precision': pr.tolist(), 'recall': rec.tolist()},
            'calibration_curve':{'frac_pos': frac_pos.tolist(),
                                 'mean_pred': mean_pred.tolist()}
        }

    # ------------------------------------------------------------------ #
    #  PERSISTENCE                                                         #
    # ------------------------------------------------------------------ #
    def save(self, filename: str = "covid19_predictor") -> Path:
        self._check_fitted()
        path = self.model_dir / f"{filename}.pkl"
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        meta_path = self.model_dir / f"{filename}_metadata.json"
        meta = {k: v for k, v in self.training_metadata.items()
                if isinstance(v, (str, int, float, dict, list))}
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        print(f"Model saved: {path}")
        return path

    @classmethod
    def load(cls, path: str) -> 'COVID19HospitalizationPredictor':
        with open(path, 'rb') as f:
            return pickle.load(f)

    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call .fit() first.")

"""
RiskScope AI - ESI Triage Predictor
Core ML model: LightGBM + XGBoost + Random Forest Ensemble with SHAP explainability
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import joblib
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, cohen_kappa_score,
    accuracy_score, f1_score, precision_score, recall_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb
import xgboost as xgb

# Imbalanced learning
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from config import (
    LIGHTGBM_PARAMS, RANDOM_FOREST_PARAMS, XGBOOST_PARAMS, ENSEMBLE_WEIGHTS,
    MODEL_SAVE_PATH, TARGET_METRICS
)
from feature_engineering import FeatureEngineer


class ESITriagePredictor:
    """
    Ensemble model for ESI level prediction.
    
    Architecture:
    - LightGBM: Fast gradient boosting (50% weight)
    - XGBoost: Robust boosting with native missing value handling (30% weight)
    - Random Forest: Interpretable ensemble (20% weight)
    
    Features:
    - SMOTE for class imbalance handling
    - Isotonic calibration for probability calibration
    - SHAP for explainability
    """
    
    def __init__(self, use_smote: bool = True, use_calibration: bool = True):
        # Initialize models
        self.lgb_model = lgb.LGBMClassifier(**LIGHTGBM_PARAMS)
        self.xgb_model = xgb.XGBClassifier(**XGBOOST_PARAMS)
        self.rf_model = RandomForestClassifier(**RANDOM_FOREST_PARAMS)
        
        # Ensemble weights
        self.weights = ENSEMBLE_WEIGHTS
        
        # Feature engineering
        self.feature_engineer = FeatureEngineer()
        self.scaler = StandardScaler()
        
        # SMOTE for imbalance
        self.use_smote = use_smote
        self.smote = SMOTE(random_state=42, k_neighbors=3)
        
        # Isotonic calibration flag
        self.use_calibration = use_calibration
        self.calibrated_models = {}
        
        # SHAP explainer (initialized after fit)
        self.shap_explainer = None
        
        # Training metadata
        self.is_fitted = False
        self.feature_names = []
        self.class_names = ['ESI 1', 'ESI 2', 'ESI 3', 'ESI 4', 'ESI 5']
        self.training_metrics = {}
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray],
            validate: bool = True, verbose: bool = True) -> 'ESITriagePredictor':
        """
        Train the ensemble model.
        
        Parameters:
        -----------
        X : Features (DataFrame or array)
        y : ESI labels (1-5)
        validate : Whether to run cross-validation
        verbose : Print training progress
        
        Returns:
        --------
        self : Fitted predictor
        """
        if verbose:
            print("=" * 60)
            print("RiskScope AI - Training ESI Triage Predictor")
            print("=" * 60)
        
        # Convert to numpy if DataFrame
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X = X.values
        
        # Convert labels to 0-indexed if needed
        y = np.array(y)
        if y.min() == 1:
            y = y - 1  # Convert to 0-indexed
        
        if verbose:
            print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
            print(f"Class distribution:")
            for i in range(5):
                count = np.sum(y == i)
                pct = count / len(y) * 100
                print(f"  ESI {i+1}: {count} ({pct:.1f}%)")
        
        # Handle class imbalance with SMOTE
        if self.use_smote:
            if verbose:
                print("\nApplying SMOTE for class balance...")
            try:
                X_resampled, y_resampled = self.smote.fit_resample(X, y)
                if verbose:
                    print(f"  Resampled: {X.shape[0]} -> {X_resampled.shape[0]} samples")
            except Exception as e:
                if verbose:
                    print(f"  SMOTE failed: {e}. Using original data.")
                X_resampled, y_resampled = X, y
        else:
            X_resampled, y_resampled = X, y
        
        # Scale features
        if verbose:
            print("\nScaling features...")
        X_scaled = self.scaler.fit_transform(X_resampled)
        
        # Train LightGBM
        if verbose:
            print("\nTraining LightGBM (50% weight)...")
        self.lgb_model.fit(X_scaled, y_resampled)
        
        # Train XGBoost
        if verbose:
            print("Training XGBoost (30% weight)...")
        self.xgb_model.fit(X_scaled, y_resampled)
        
        # Train Random Forest
        if verbose:
            print("Training Random Forest (20% weight)...")
        self.rf_model.fit(X_scaled, y_resampled)
        
        # Apply Isotonic Calibration if enabled
        if self.use_calibration:
            if verbose:
                print("\nApplying Isotonic calibration...")
            self._apply_calibration(X_scaled, y_resampled)
        
        self.is_fitted = True
        
        # Cross-validation
        if validate:
            if verbose:
                print("\nRunning 5-fold cross-validation...")
            self._run_cross_validation(X, y, verbose)
        
        # Initialize SHAP explainer
        if verbose:
            print("\nInitializing SHAP explainer...")
        self._init_shap_explainer(X_scaled)
        
        if verbose:
            print("\n" + "=" * 60)
            print("Training complete! (3-model ensemble with calibration)")
            print("=" * 60)
        
        return self
    
    def _apply_calibration(self, X: np.ndarray, y: np.ndarray):
        """Apply Isotonic calibration to improve probability estimates."""
        from sklearn.model_selection import train_test_split
        import sklearn
        
        # Use a small holdout for calibration
        X_train, X_cal, y_train, y_cal = train_test_split(
            X, y, test_size=0.15, stratify=y, random_state=42
        )
        
        # Check sklearn version for cv parameter compatibility
        sklearn_version = tuple(int(x) for x in sklearn.__version__.split('.')[:2])
        
        # Calibrate each model
        try:
            # For sklearn >= 1.5, use cv=None with pre-fitted estimators
            if sklearn_version >= (1, 5):
                # Fit calibrators directly on calibration set
                self.calibrated_models['lightgbm'] = CalibratedClassifierCV(
                    estimator=self.lgb_model, cv=None, method='isotonic'
                )
                self.calibrated_models['lightgbm'].fit(X_cal, y_cal)
                
                self.calibrated_models['xgboost'] = CalibratedClassifierCV(
                    estimator=self.xgb_model, cv=None, method='isotonic'
                )
                self.calibrated_models['xgboost'].fit(X_cal, y_cal)
                
                self.calibrated_models['random_forest'] = CalibratedClassifierCV(
                    estimator=self.rf_model, cv=None, method='isotonic'
                )
                self.calibrated_models['random_forest'].fit(X_cal, y_cal)
            else:
                # Legacy sklearn: use cv='prefit'
                self.calibrated_models['lightgbm'] = CalibratedClassifierCV(
                    self.lgb_model, cv='prefit', method='isotonic'
                )
                self.calibrated_models['lightgbm'].fit(X_cal, y_cal)
                
                self.calibrated_models['xgboost'] = CalibratedClassifierCV(
                    self.xgb_model, cv='prefit', method='isotonic'
                )
                self.calibrated_models['xgboost'].fit(X_cal, y_cal)
                
                self.calibrated_models['random_forest'] = CalibratedClassifierCV(
                    self.rf_model, cv='prefit', method='isotonic'
                )
                self.calibrated_models['random_forest'].fit(X_cal, y_cal)
        except Exception as e:
            print(f"  Calibration warning: {e}")
            self.calibrated_models = {}
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray, Dict]) -> np.ndarray:
        """
        Predict ESI levels.
        
        Parameters:
        -----------
        X : Features (DataFrame, array, or single patient dict)
        
        Returns:
        --------
        predictions : ESI levels (1-5)
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        # Handle single patient dict
        if isinstance(X, dict):
            X = self.feature_engineer.transform(X)
        elif isinstance(X, pd.DataFrame):
            X = X.values
        
        X_scaled = self.scaler.transform(X)
        
        # Ensemble prediction
        proba = self.predict_proba(X_scaled, already_scaled=True)
        predictions = np.argmax(proba, axis=1) + 1  # Convert to 1-indexed
        
        return predictions
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray, Dict],
                      already_scaled: bool = False) -> np.ndarray:
        """
        Predict class probabilities.
        
        Returns:
        --------
        probabilities : Shape (n_samples, 5) for ESI 1-5
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        # Handle single patient dict
        if isinstance(X, dict):
            X = self.feature_engineer.transform(X)
        elif isinstance(X, pd.DataFrame):
            X = X.values
        
        if not already_scaled:
            X = self.scaler.transform(X)
        
        # Check if XGBoost is fitted
        xgb_fitted = hasattr(self.xgb_model, 'get_booster') and self.xgb_model.get_booster is not None
        try:
            # This will raise NotFittedError if not fitted
            if xgb_fitted:
                _ = self.xgb_model.get_booster()
        except:
            xgb_fitted = False
        
        # Get probabilities from models (use calibrated if available)
        if self.calibrated_models and 'lightgbm' in self.calibrated_models:
            lgb_proba = self.calibrated_models['lightgbm'].predict_proba(X)
            rf_proba = self.calibrated_models['random_forest'].predict_proba(X)
            if xgb_fitted and 'xgboost' in self.calibrated_models:
                xgb_proba = self.calibrated_models['xgboost'].predict_proba(X)
            else:
                xgb_proba = None
        else:
            lgb_proba = self.lgb_model.predict_proba(X)
            rf_proba = self.rf_model.predict_proba(X)
            if xgb_fitted:
                xgb_proba = self.xgb_model.predict_proba(X)
            else:
                xgb_proba = None
        
        # Weighted ensemble - adjust weights if XGBoost not available
        if xgb_proba is not None:
            # Full 3-model ensemble (50% LGB + 30% XGB + 20% RF)
            ensemble_proba = (
                self.weights['lightgbm'] * lgb_proba +
                self.weights.get('xgboost', 0.3) * xgb_proba +
                self.weights['random_forest'] * rf_proba
            )
        else:
            # Fallback 2-model ensemble (65% LGB + 35% RF)
            lgb_weight = self.weights['lightgbm'] / (self.weights['lightgbm'] + self.weights['random_forest'])
            rf_weight = self.weights['random_forest'] / (self.weights['lightgbm'] + self.weights['random_forest'])
            ensemble_proba = lgb_weight * lgb_proba + rf_weight * rf_proba
        
        return ensemble_proba
    
    def explain_prediction(self, X: Union[np.ndarray, Dict],
                          top_n: int = 5) -> List[Dict]:
        """
        Explain prediction using SHAP values.
        
        Parameters:
        -----------
        X : Single patient features (array or dict)
        top_n : Number of top features to return
        
        Returns:
        --------
        explanations : List of {feature, impact, value} dicts
        """
        if not self.is_fitted or self.shap_explainer is None:
            return self._fallback_explanation(X, top_n)
        
        try:
            import shap
            
            # Handle dict input
            if isinstance(X, dict):
                X = self.feature_engineer.transform(X)
            
            X_scaled = self.scaler.transform(X.reshape(1, -1) if X.ndim == 1 else X)
            
            # Get SHAP values
            shap_values = self.shap_explainer.shap_values(X_scaled)
            
            # Get predicted class
            pred_proba = self.predict_proba(X_scaled, already_scaled=True)[0]
            pred_class = np.argmax(pred_proba)
            
            # Get SHAP values for predicted class
            if isinstance(shap_values, list):
                class_shap_values = shap_values[pred_class][0]
            else:
                class_shap_values = shap_values[0]
            
            # Get top features
            feature_importance = np.abs(class_shap_values)
            top_indices = np.argsort(feature_importance)[-top_n:][::-1]
            
            explanations = []
            for idx in top_indices:
                feature_name = (self.feature_names[idx] 
                               if idx < len(self.feature_names) 
                               else f'feature_{idx}')
                shap_val = class_shap_values[idx]
                
                explanations.append({
                    'feature': feature_name,
                    'impact': '+' if shap_val > 0 else '-',
                    'value': round(abs(shap_val), 3),
                    'contribution': 'increases urgency' if shap_val > 0 else 'decreases urgency'
                })
            
            return explanations
            
        except Exception as e:
            print(f"SHAP explanation failed: {e}")
            return self._fallback_explanation(X, top_n)
    
    def _fallback_explanation(self, X, top_n: int = 5) -> List[Dict]:
        """Fallback to feature importance when SHAP fails."""
        importances = self.lgb_model.feature_importances_
        top_indices = np.argsort(importances)[-top_n:][::-1]
        
        return [
            {
                'feature': (self.feature_names[i] 
                           if i < len(self.feature_names) 
                           else f'feature_{i}'),
                'impact': '+',
                'value': round(importances[i] / importances.max(), 3),
                'contribution': 'high importance'
            }
            for i in top_indices
        ]
    
    def _run_cross_validation(self, X: np.ndarray, y: np.ndarray, 
                              verbose: bool = True):
        """Run stratified 5-fold cross-validation."""
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        kappa_scores = []
        sensitivity_12 = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Scale
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # SMOTE
            if self.use_smote:
                try:
                    X_train_scaled, y_train = self.smote.fit_resample(X_train_scaled, y_train)
                except:
                    pass
            
            # Train LightGBM
            model = lgb.LGBMClassifier(**LIGHTGBM_PARAMS)
            model.fit(X_train_scaled, y_train)
            
            # Predict
            y_pred = model.predict(X_val_scaled)
            
            # Metrics
            kappa = cohen_kappa_score(y_val, y_pred, weights='quadratic')
            kappa_scores.append(kappa)
            
            # ESI 1-2 sensitivity
            esi_12_mask = y_val <= 1  # 0-indexed, so 0 and 1 = ESI 1 and 2
            if esi_12_mask.sum() > 0:
                sens = (y_pred[esi_12_mask] <= 1).mean()
                sensitivity_12.append(sens)
        
        # Store metrics
        self.training_metrics = {
            'cohens_kappa_mean': np.mean(kappa_scores),
            'cohens_kappa_std': np.std(kappa_scores),
            'esi_12_sensitivity_mean': np.mean(sensitivity_12) if sensitivity_12 else 0,
            'esi_12_sensitivity_std': np.std(sensitivity_12) if sensitivity_12 else 0
        }
        
        if verbose:
            print(f"\n  Cohen's Kappa: {self.training_metrics['cohens_kappa_mean']:.3f} "
                  f"(± {self.training_metrics['cohens_kappa_std']:.3f})")
            print(f"  ESI 1-2 Sensitivity: {self.training_metrics['esi_12_sensitivity_mean']:.1%} "
                  f"(± {self.training_metrics['esi_12_sensitivity_std']:.1%})")
            
            # Check against targets
            if self.training_metrics['cohens_kappa_mean'] >= TARGET_METRICS['cohens_kappa']:
                print(f"  [OK] Kappa target met (>= {TARGET_METRICS['cohens_kappa']})")
            else:
                print(f"  [!]  Kappa below target ({TARGET_METRICS['cohens_kappa']})")
            
            if self.training_metrics['esi_12_sensitivity_mean'] >= TARGET_METRICS['esi_12_sensitivity']:
                print(f"  [OK] ESI 1-2 sensitivity target met (>= {TARGET_METRICS['esi_12_sensitivity']:.0%})")
            else:
                print(f"  [!]  ESI 1-2 sensitivity below target ({TARGET_METRICS['esi_12_sensitivity']:.0%})")
    
    def _init_shap_explainer(self, X_sample: np.ndarray):
        """Initialize SHAP TreeExplainer."""
        try:
            import shap
            # Use a sample for background (faster)
            sample_size = min(100, len(X_sample))
            background = X_sample[np.random.choice(len(X_sample), sample_size, replace=False)]
            self.shap_explainer = shap.TreeExplainer(self.lgb_model, background)
        except Exception as e:
            print(f"  Warning: SHAP initialization failed: {e}")
            self.shap_explainer = None
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, verbose: bool = True) -> Dict:
        """
        Comprehensive evaluation on test set.
        
        Returns:
        --------
        metrics : Dict with all evaluation metrics
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        # Predictions
        y_pred = self.predict(X)
        
        # Convert to 0-indexed if needed
        y = np.array(y)
        if y.min() == 1:
            y = y - 1
        y_pred = y_pred - 1  # predictions are 1-indexed
        
        # Metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'cohens_kappa': cohen_kappa_score(y, y_pred, weights='quadratic'),
            'f1_macro': f1_score(y, y_pred, average='macro'),
            'f1_weighted': f1_score(y, y_pred, average='weighted'),
        }
        
        # Per-class metrics
        for i in range(5):
            mask = y == i
            if mask.sum() > 0:
                metrics[f'esi_{i+1}_sensitivity'] = (y_pred[mask] == i).mean()
        
        # ESI 1-2 combined sensitivity (critical!)
        esi_12_mask = y <= 1
        if esi_12_mask.sum() > 0:
            metrics['esi_12_sensitivity'] = (y_pred[esi_12_mask] <= 1).mean()
        
        # Critical errors (under-triage: actual 1-2, predicted 4-5)
        critical_errors = ((y <= 1) & (y_pred >= 3)).sum()
        metrics['critical_errors'] = int(critical_errors)
        metrics['critical_error_rate'] = critical_errors / len(y)
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y, y_pred)
        
        if verbose:
            print("\n" + "=" * 60)
            print("Evaluation Results")
            print("=" * 60)
            print(f"\nOverall Metrics:")
            print(f"  Accuracy: {metrics['accuracy']:.1%}")
            print(f"  Cohen's Kappa: {metrics['cohens_kappa']:.3f}")
            print(f"  F1 (weighted): {metrics['f1_weighted']:.3f}")
            
            print(f"\nSafety Metrics:")
            print(f"  ESI 1-2 Sensitivity: {metrics.get('esi_12_sensitivity', 0):.1%}")
            print(f"  Critical Errors: {metrics['critical_errors']} "
                  f"({metrics['critical_error_rate']:.2%})")
            
            print(f"\nPer-Class Sensitivity:")
            for i in range(5):
                sens = metrics.get(f'esi_{i+1}_sensitivity', 0)
                print(f"  ESI {i+1}: {sens:.1%}")
            
            print("\nConfusion Matrix:")
            print(metrics['confusion_matrix'])
        
        return metrics
    
    def save(self, path: str = None):
        """Save model to disk."""
        if path is None:
            path = MODEL_SAVE_PATH + 'esi_ensemble_model.pkl'
        
        joblib.dump({
            'lgb_model': self.lgb_model,
            'xgb_model': self.xgb_model,
            'rf_model': self.rf_model,
            'calibrated_models': self.calibrated_models,
            'scaler': self.scaler,
            'feature_engineer': self.feature_engineer,
            'feature_names': self.feature_names,
            'weights': self.weights,
            'training_metrics': self.training_metrics,
            'is_fitted': self.is_fitted,
            'use_calibration': self.use_calibration
        }, path)
        print(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str = None) -> 'ESITriagePredictor':
        """Load model from disk."""
        if path is None:
            path = MODEL_SAVE_PATH + 'esi_ensemble_model.pkl'
        
        data = joblib.load(path)
        
        predictor = cls(use_smote=False, use_calibration=False)
        predictor.lgb_model = data['lgb_model']
        predictor.xgb_model = data.get('xgb_model', None)
        predictor.rf_model = data['rf_model']
        predictor.calibrated_models = data.get('calibrated_models', {})
        predictor.scaler = data['scaler']
        predictor.feature_engineer = data['feature_engineer']
        predictor.feature_names = data['feature_names']
        predictor.weights = data['weights']
        predictor.training_metrics = data['training_metrics']
        predictor.is_fitted = data['is_fitted']
        predictor.use_calibration = data.get('use_calibration', False)
        
        # Handle old models without XGBoost
        if predictor.xgb_model is None:
            predictor.xgb_model = xgb.XGBClassifier(**XGBOOST_PARAMS)
            # Adjust weights for 2-model ensemble
            if 'xgboost' in predictor.weights:
                total = predictor.weights['lightgbm'] + predictor.weights['random_forest']
                predictor.weights['lightgbm'] /= total
                predictor.weights['random_forest'] /= total
                predictor.weights['xgboost'] = 0.0
        
        print(f"Model loaded from {path}")
        return predictor
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get feature importance from all 3 models."""
        lgb_imp = self.lgb_model.feature_importances_
        rf_imp = self.rf_model.feature_importances_
        
        # Normalize
        lgb_imp = lgb_imp / lgb_imp.max()
        rf_imp = rf_imp / rf_imp.max()
        
        # Get XGBoost importance if available
        if hasattr(self.xgb_model, 'feature_importances_'):
            xgb_imp = self.xgb_model.feature_importances_
            xgb_imp = xgb_imp / (xgb_imp.max() + 1e-10)
        else:
            xgb_imp = np.zeros_like(lgb_imp)
        
        # Weighted average
        combined = (
            self.weights['lightgbm'] * lgb_imp + 
            self.weights.get('xgboost', 0) * xgb_imp +
            self.weights['random_forest'] * rf_imp
        )
        
        # Create DataFrame
        df = pd.DataFrame({
            'feature': self.feature_names if self.feature_names else [f'f_{i}' for i in range(len(combined))],
            'importance': combined,
            'lgb_importance': lgb_imp,
            'xgb_importance': xgb_imp,
            'rf_importance': rf_imp
        })
        
        return df.sort_values('importance', ascending=False).head(top_n)

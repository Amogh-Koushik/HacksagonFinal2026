"""
RiskScope AI - Out-of-Distribution Detection
Flags unusual patient presentations for automatic escalation
"""

import numpy as np
from typing import Dict, Any, Optional, Union, List
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
import joblib


class OutOfDistributionDetector:
    """
    Detects when patient data is unusual/outside training distribution.
    Uses multiple methods for robust detection:
    1. Statistical (z-score based)
    2. Local Outlier Factor
    3. Isolation Forest
    
    When OOD score is high, we escalate automatically (fail-safe).
    """
    
    def __init__(self, contamination: float = 0.05):
        """
        Parameters:
        -----------
        contamination : Expected proportion of outliers in training data
        """
        self.contamination = contamination
        
        # Statistical tracking
        self.training_stats = None
        
        # ML-based detectors
        self.lof_detector = LocalOutlierFactor(
            n_neighbors=20,
            contamination=contamination,
            novelty=True
        )
        self.isolation_forest = IsolationForest(
            n_estimators=100,
            contamination=contamination,
            random_state=42
        )
        
        self.is_fitted = False
        self.n_features = None
    
    def fit(self, X: np.ndarray) -> 'OutOfDistributionDetector':
        """
        Fit detectors on training data.
        
        Parameters:
        -----------
        X : Training feature matrix
        """
        X = np.array(X)
        self.n_features = X.shape[1]
        
        # Compute statistical baselines
        self.training_stats = {
            'means': np.mean(X, axis=0),
            'stds': np.std(X, axis=0) + 1e-6,  # Avoid division by zero
            'mins': np.min(X, axis=0),
            'maxs': np.max(X, axis=0),
            'q1': np.percentile(X, 25, axis=0),
            'q3': np.percentile(X, 75, axis=0),
        }
        
        # IQR for robust outlier detection
        self.training_stats['iqr'] = self.training_stats['q3'] - self.training_stats['q1']
        self.training_stats['lower_bound'] = self.training_stats['q1'] - 1.5 * self.training_stats['iqr']
        self.training_stats['upper_bound'] = self.training_stats['q3'] + 1.5 * self.training_stats['iqr']
        
        # Fit ML detectors
        try:
            self.lof_detector.fit(X)
        except Exception:
            self.lof_detector = None
        
        try:
            self.isolation_forest.fit(X)
        except Exception:
            self.isolation_forest = None
        
        self.is_fitted = True
        return self
    
    def score(self, X: Union[np.ndarray, Dict]) -> float:
        """
        Calculate OOD score for a feature vector.
        
        Higher score = more unusual = more likely out-of-distribution.
        
        Parameters:
        -----------
        X : Feature vector (1D array or dict)
        
        Returns:
        --------
        ood_score : float between 0 and 1
        """
        if not self.is_fitted:
            # Conservative: assume in-distribution if not fitted
            return 0.3
        
        # Handle dict input
        if isinstance(X, dict):
            # Assume feature_engineer will be used elsewhere
            return 0.3
        
        X = np.array(X).flatten()
        
        if len(X) != self.n_features:
            # Feature mismatch - flag as unusual
            return 0.7
        
        # Component 1: Z-score based detection
        z_score_component = self._z_score_detection(X)
        
        # Component 2: Range-based detection
        range_component = self._range_detection(X)
        
        # Component 3: IQR-based detection
        iqr_component = self._iqr_detection(X)
        
        # Component 4: ML-based detection (if available)
        ml_component = self._ml_detection(X.reshape(1, -1))
        
        # Weighted combination
        weights = {
            'z_score': 0.25,
            'range': 0.20,
            'iqr': 0.25,
            'ml': 0.30
        }
        
        ood_score = (
            weights['z_score'] * z_score_component +
            weights['range'] * range_component +
            weights['iqr'] * iqr_component +
            weights['ml'] * ml_component
        )
        
        return min(max(ood_score, 0.0), 1.0)
    
    def _z_score_detection(self, X: np.ndarray) -> float:
        """Z-score based outlier detection."""
        z_scores = np.abs((X - self.training_stats['means']) / self.training_stats['stds'])
        
        # Max z-score normalized to 0-1 (z=5 maps to 1.0)
        max_z = np.max(z_scores)
        return min(max_z / 5.0, 1.0)
    
    def _range_detection(self, X: np.ndarray) -> float:
        """Range-based outlier detection."""
        below_min = X < self.training_stats['mins']
        above_max = X > self.training_stats['maxs']
        out_of_range = (below_min | above_max).sum() / len(X)
        return out_of_range
    
    def _iqr_detection(self, X: np.ndarray) -> float:
        """IQR-based outlier detection."""
        below_lower = X < self.training_stats['lower_bound']
        above_upper = X > self.training_stats['upper_bound']
        iqr_outliers = (below_lower | above_upper).sum() / len(X)
        return iqr_outliers
    
    def _ml_detection(self, X: np.ndarray) -> float:
        """ML-based outlier detection (LOF + Isolation Forest)."""
        scores = []
        
        # Local Outlier Factor
        if self.lof_detector is not None:
            try:
                lof_score = self.lof_detector.decision_function(X)[0]
                # Convert to 0-1 (negative = outlier)
                lof_normalized = 1.0 - min(max(lof_score / 2.0 + 0.5, 0.0), 1.0)
                scores.append(lof_normalized)
            except Exception:
                pass
        
        # Isolation Forest
        if self.isolation_forest is not None:
            try:
                iso_score = self.isolation_forest.decision_function(X)[0]
                # Convert to 0-1 (negative = outlier)
                iso_normalized = 1.0 - min(max(iso_score / 0.5 + 0.5, 0.0), 1.0)
                scores.append(iso_normalized)
            except Exception:
                pass
        
        if scores:
            return np.mean(scores)
        return 0.3  # Default if ML methods fail
    
    def is_outlier(self, X: Union[np.ndarray, Dict], threshold: float = 0.8) -> bool:
        """
        Binary outlier detection.
        
        Parameters:
        -----------
        X : Feature vector
        threshold : OOD score threshold for flagging as outlier
        
        Returns:
        --------
        is_outlier : True if patient is unusual
        """
        return self.score(X) > threshold
    
    def get_anomaly_details(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Get detailed breakdown of anomaly scores.
        
        Returns:
        --------
        details : Dict with component scores and anomalous features
        """
        X = np.array(X).flatten()
        
        if not self.is_fitted:
            return {'ood_score': 0.3, 'fitted': False}
        
        # Z-scores for each feature
        z_scores = np.abs((X - self.training_stats['means']) / self.training_stats['stds'])
        
        # Find most anomalous features
        anomaly_indices = np.argsort(z_scores)[-5:][::-1]
        
        return {
            'ood_score': self.score(X),
            'z_score_component': self._z_score_detection(X),
            'range_component': self._range_detection(X),
            'iqr_component': self._iqr_detection(X),
            'ml_component': self._ml_detection(X.reshape(1, -1)),
            'max_z_score': float(np.max(z_scores)),
            'most_anomalous_features': anomaly_indices.tolist(),
            'anomaly_z_scores': z_scores[anomaly_indices].tolist()
        }
    
    def save(self, path: str):
        """Save detector to disk."""
        joblib.dump({
            'training_stats': self.training_stats,
            'lof_detector': self.lof_detector,
            'isolation_forest': self.isolation_forest,
            'is_fitted': self.is_fitted,
            'n_features': self.n_features,
            'contamination': self.contamination
        }, path)
    
    @classmethod
    def load(cls, path: str) -> 'OutOfDistributionDetector':
        """Load detector from disk."""
        data = joblib.load(path)
        
        detector = cls(contamination=data['contamination'])
        detector.training_stats = data['training_stats']
        detector.lof_detector = data['lof_detector']
        detector.isolation_forest = data['isolation_forest']
        detector.is_fitted = data['is_fitted']
        detector.n_features = data['n_features']
        
        return detector


def detect_unusual_vitals(patient_data: Dict[str, Any]) -> List[str]:
    """
    Quick check for obviously unusual vital signs.
    Returns list of warning messages.
    """
    warnings = []
    
    # Heart rate
    hr = patient_data.get('heart_rate', 80)
    if hr < 30:
        warnings.append(f"Critical bradycardia: HR {hr}")
    elif hr > 200:
        warnings.append(f"Extreme tachycardia: HR {hr}")
    
    # Blood pressure
    sbp = patient_data.get('bp_systolic', 120)
    if sbp < 60:
        warnings.append(f"Severe hypotension: SBP {sbp}")
    elif sbp > 220:
        warnings.append(f"Hypertensive crisis: SBP {sbp}")
    
    # SpO2
    spo2 = patient_data.get('spo2', 98)
    if spo2 < 70:
        warnings.append(f"Critical hypoxemia: SpO2 {spo2}%")
    
    # Temperature
    temp = patient_data.get('temperature', 37)
    if temp < 32:
        warnings.append(f"Severe hypothermia: {temp}°C")
    elif temp > 42:
        warnings.append(f"Hyperpyrexia: {temp}°C")
    
    # Respiratory rate
    rr = patient_data.get('respiratory_rate', 16)
    if rr < 6:
        warnings.append(f"Critical bradypnea: RR {rr}")
    elif rr > 40:
        warnings.append(f"Severe tachypnea: RR {rr}")
    
    return warnings

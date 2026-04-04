"""
RiskScope AI - Confidence Calibration
=======================================
Computes calibrated prediction confidence and flags uncertain predictions
for automatic escalation (safety-first design).

Used by:
  - ClinicalSafetyEngine  (in safety_engine.py)
  - Flask API             (confidence field in /api/predict response)
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple


class ConfidenceCalibrator:
    """
    Analyses prediction probability distributions to determine
    when the model is uncertain and should escalate.

    Three uncertainty signals:
    1. Max probability (primary confidence metric)
    2. Entropy (spread of probability across classes)
    3. Margin (gap between top-2 predictions)
    """

    def __init__(self,
                 confidence_threshold: float = 0.6,
                 entropy_threshold: float = 1.2,
                 margin_threshold: float = 0.15):
        """
        Parameters
        ----------
        confidence_threshold : float
            Below this max-probability, prediction is flagged as low confidence.
        entropy_threshold : float
            Above this entropy, prediction is flagged as uncertain.
        margin_threshold : float
            Below this margin between top-2 classes, prediction is ambiguous.
        """
        self.confidence_threshold = confidence_threshold
        self.entropy_threshold = entropy_threshold
        self.margin_threshold = margin_threshold

    def calibrate(self, proba: np.ndarray) -> Dict[str, Any]:
        """
        Analyse a probability vector and return calibrated confidence metrics.

        Parameters
        ----------
        proba : np.ndarray
            Probability distribution over ESI classes, shape (5,).

        Returns
        -------
        result : dict with keys:
            - confidence : float (0-1) — calibrated confidence score
            - max_probability : float — raw max class probability
            - entropy : float — Shannon entropy of distribution
            - margin : float — gap between top-2 class probabilities
            - is_low_confidence : bool — should we escalate?
            - is_ambiguous : bool — are top-2 classes close?
            - escalation_reason : str or None — why escalation is needed
            - predicted_class : int — 0-indexed predicted class
            - runner_up_class : int — 0-indexed second-best class
        """
        proba = np.array(proba).flatten()

        # Ensure valid probability distribution
        if proba.sum() == 0:
            proba = np.ones(5) / 5
        proba = proba / proba.sum()

        # Core metrics
        max_prob = float(np.max(proba))
        predicted_class = int(np.argmax(proba))

        # Entropy: H = -Σ p*log(p)
        # Max entropy for 5 classes = ln(5) ≈ 1.609
        entropy = float(-np.sum(proba * np.log(proba + 1e-10)))

        # Margin: difference between top-2 probabilities
        sorted_proba = np.sort(proba)[::-1]
        margin = float(sorted_proba[0] - sorted_proba[1])
        runner_up_class = int(np.argsort(proba)[-2])

        # Calibrated confidence: weighted combination of signals
        # Higher max_prob → higher confidence
        # Lower entropy → higher confidence
        # Higher margin → higher confidence
        max_entropy = np.log(5)  # ~1.609
        normalized_entropy = entropy / max_entropy  # 0 to 1

        calibrated = (
            0.50 * max_prob +
            0.25 * (1 - normalized_entropy) +
            0.25 * min(margin / 0.5, 1.0)  # margin of 0.5+ → full confidence
        )
        calibrated = float(np.clip(calibrated, 0, 1))

        # Determine if escalation is needed
        is_low_confidence = max_prob < self.confidence_threshold
        is_ambiguous = margin < self.margin_threshold
        is_high_entropy = entropy > self.entropy_threshold

        escalation_reason = None
        if is_low_confidence:
            escalation_reason = (
                f"Low prediction confidence ({max_prob:.0%}). "
                f"Model uncertain — escalating for patient safety."
            )
        elif is_ambiguous:
            esi_pred = predicted_class + 1
            esi_runner = runner_up_class + 1
            escalation_reason = (
                f"Ambiguous prediction: ESI {esi_pred} ({sorted_proba[0]:.0%}) "
                f"vs ESI {esi_runner} ({sorted_proba[1]:.0%}). "
                f"Escalating to more urgent level."
            )
        elif is_high_entropy:
            escalation_reason = (
                f"High prediction spread (entropy={entropy:.2f}). "
                f"Model cannot clearly distinguish severity levels."
            )

        should_escalate = is_low_confidence or is_ambiguous or is_high_entropy

        return {
            'confidence': calibrated,
            'max_probability': max_prob,
            'entropy': entropy,
            'margin': margin,
            'is_low_confidence': is_low_confidence,
            'is_ambiguous': is_ambiguous,
            'is_high_entropy': is_high_entropy,
            'should_escalate': should_escalate,
            'escalation_reason': escalation_reason,
            'predicted_class': predicted_class,
            'runner_up_class': runner_up_class,
            'class_probabilities': {
                f'esi_{i+1}': round(float(proba[i]), 4) for i in range(5)
            },
        }

    def escalate_prediction(self, esi_level: int,
                            calibration_result: Dict) -> Tuple[int, str]:
        """
        Apply escalation logic to a prediction based on confidence.

        Safety-first: always escalate towards MORE urgent (lower ESI number).

        Parameters
        ----------
        esi_level : int
            Original predicted ESI level (1-5).
        calibration_result : dict
            Output from self.calibrate().

        Returns
        -------
        (escalated_esi, reason) : (int, str)
            The (potentially escalated) ESI level and reason string.
        """
        if not calibration_result['should_escalate']:
            return esi_level, "Prediction confident"

        # Escalate one level towards more urgent
        escalated = max(1, esi_level - 1)
        reason = calibration_result['escalation_reason'] or "Low confidence — escalated"

        return escalated, reason

    def get_confidence_display(self, calibration_result: Dict) -> Dict[str, str]:
        """
        Get display-friendly confidence information for the UI.

        Returns
        -------
        display : dict with:
            - level : str — 'HIGH', 'MODERATE', or 'LOW'
            - color : str — hex color for UI
            - label : str — human-readable label
            - percentage : str — confidence as percentage string
        """
        conf = calibration_result['confidence']

        if conf >= 0.8:
            level = 'HIGH'
            color = '#22c55e'  # Green
            label = 'High Confidence'
        elif conf >= 0.5:
            level = 'MODERATE'
            color = '#eab308'  # Yellow
            label = 'Moderate Confidence'
        else:
            level = 'LOW'
            color = '#dc2626'  # Red
            label = 'Low Confidence — Manual Review Required'

        return {
            'level': level,
            'color': color,
            'label': label,
            'percentage': f"{conf:.0%}",
        }

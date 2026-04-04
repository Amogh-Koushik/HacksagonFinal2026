"""
RiskScope AI - Model Evaluation Module
Comprehensive evaluation metrics and error analysis for ESI prediction

Metrics included:
- Standard: Accuracy, Cohen's Kappa, F1
- AUC-ROC per class (One-vs-Rest)
- Calibration curve
- Under-triage / Over-triage rates
- Inference latency (ms)
- Normalized confusion matrix
- Demo case accuracy
"""

import numpy as np
import pandas as pd
import time
import json
import os
from typing import Dict, List, Any, Optional, Tuple
from sklearn.metrics import (
    confusion_matrix, classification_report, cohen_kappa_score,
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings('ignore')


class ESIEvaluator:
    """
    Comprehensive evaluation for ESI triage predictions.
    Focuses on safety-critical metrics.
    """
    
    def __init__(self, class_names: List[str] = None):
        self.class_names = class_names or ['ESI 1', 'ESI 2', 'ESI 3', 'ESI 4', 'ESI 5']
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, 
                 y_proba: np.ndarray = None,
                 verbose: bool = True) -> Dict[str, Any]:
        """
        Run comprehensive evaluation.
        
        Parameters:
        -----------
        y_true : True ESI levels (1-5)
        y_pred : Predicted ESI levels (1-5)
        y_proba : Prediction probabilities (optional)
        verbose : Print results
        
        Returns:
        --------
        metrics : Dict with all evaluation metrics
        """
        # Convert to 0-indexed internally
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        if y_true.min() == 1:
            y_true = y_true - 1
        if y_pred.min() == 1:
            y_pred = y_pred - 1
        
        metrics = {}
        
        # Overall metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['cohens_kappa'] = cohen_kappa_score(y_true, y_pred, weights='quadratic')
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro')
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted')
        
        # Per-class metrics
        for i in range(5):
            mask = y_true == i
            if mask.sum() > 0:
                metrics[f'esi_{i+1}_sensitivity'] = (y_pred[mask] == i).mean()
                metrics[f'esi_{i+1}_count'] = int(mask.sum())
        
        # Safety-critical metrics
        metrics.update(self._compute_safety_metrics(y_true, y_pred))
        
        # AUC-ROC per class (if probabilities available)
        if y_proba is not None:
            metrics.update(self._compute_auc_roc(y_true, y_proba))
            metrics.update(self._compute_calibration(y_true, y_proba))
        
        # Confusion matrix (raw and normalized)
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        
        # Error analysis
        metrics['error_analysis'] = self._error_analysis(y_true, y_pred)
        
        if verbose:
            self._print_report(metrics)
        
        return metrics
    
    def _compute_safety_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Compute safety-critical metrics."""
        metrics = {}
        
        # ESI 1-2 combined (high-acuity)
        esi_12_mask = y_true <= 1
        if esi_12_mask.sum() > 0:
            # Sensitivity: Of true emergencies, how many did we catch?
            metrics['esi_12_sensitivity'] = (y_pred[esi_12_mask] <= 1).mean()
            
            # Specificity: Of non-emergencies, how many did we correctly identify?
            non_esi_12_mask = y_true > 1
            if non_esi_12_mask.sum() > 0:
                metrics['esi_12_specificity'] = (y_pred[non_esi_12_mask] > 1).mean()
        
        # Critical errors (under-triage of emergencies)
        # Level 1-2 patient classified as Level 4-5
        critical_undertriage = ((y_true <= 1) & (y_pred >= 3)).sum()
        metrics['critical_undertriage_count'] = int(critical_undertriage)
        metrics['critical_undertriage_rate'] = critical_undertriage / len(y_true)
        
        # Severe under-triage: Level 1 classified as Level 4-5
        severe_undertriage = ((y_true == 0) & (y_pred >= 3)).sum()
        metrics['severe_undertriage_count'] = int(severe_undertriage)
        
        # Over-triage (less harmful but wastes resources)
        # Level 4-5 patient classified as Level 1-2
        overtriage = ((y_true >= 3) & (y_pred <= 1)).sum()
        metrics['overtriage_count'] = int(overtriage)
        metrics['overtriage_rate'] = overtriage / len(y_true)
        
        # Adjacent accuracy (within 1 level)
        adjacent_correct = np.abs(y_true - y_pred) <= 1
        metrics['adjacent_accuracy'] = adjacent_correct.mean()
        
        return metrics
    
    def _compute_auc_roc(self, y_true: np.ndarray, y_proba: np.ndarray) -> Dict:
        """
        Compute AUC-ROC per class using One-vs-Rest strategy.
        
        This is a standard ML metric that shows discrimination power.
        """
        metrics = {}
        
        try:
            # Binarize labels for OvR AUC calculation
            n_classes = 5
            y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
            
            # Compute AUC-ROC for each class
            auc_scores = []
            for i in range(n_classes):
                if y_true_bin[:, i].sum() > 0:  # Only if class exists in data
                    try:
                        class_auc = roc_auc_score(y_true_bin[:, i], y_proba[:, i])
                        metrics[f'auc_roc_esi_{i+1}'] = float(class_auc)
                        auc_scores.append(class_auc)
                    except:
                        metrics[f'auc_roc_esi_{i+1}'] = None
            
            # Macro-average AUC
            if auc_scores:
                metrics['auc_roc_macro'] = float(np.mean(auc_scores))
            
            # Weighted AUC (by class frequency)
            try:
                metrics['auc_roc_weighted'] = float(roc_auc_score(
                    y_true_bin, y_proba, average='weighted', multi_class='ovr'
                ))
            except:
                pass
                
        except Exception as e:
            metrics['auc_roc_error'] = str(e)
        
        return metrics
    
    def _compute_calibration(self, y_true: np.ndarray, y_proba: np.ndarray) -> Dict:
        """
        Compute calibration metrics.
        
        Good calibration means: if model says 80% confidence, it should be correct 80% of time.
        """
        metrics = {}
        
        try:
            # Get predicted class and max probability
            y_pred = np.argmax(y_proba, axis=1)
            max_proba = np.max(y_proba, axis=1)
            correct = (y_pred == y_true).astype(int)
            
            # Calibration curve (fraction of positives vs mean predicted value)
            n_bins = 10
            bin_edges = np.linspace(0, 1, n_bins + 1)
            
            calibration_data = []
            for i in range(n_bins):
                mask = (max_proba >= bin_edges[i]) & (max_proba < bin_edges[i+1])
                if mask.sum() > 0:
                    bin_center = (bin_edges[i] + bin_edges[i+1]) / 2
                    actual_accuracy = correct[mask].mean()
                    count = mask.sum()
                    calibration_data.append({
                        'bin_center': float(bin_center),
                        'predicted_prob': float(max_proba[mask].mean()),
                        'actual_accuracy': float(actual_accuracy),
                        'count': int(count)
                    })
            
            metrics['calibration_curve'] = calibration_data
            
            # Expected Calibration Error (ECE) - lower is better
            ece = 0.0
            total = len(y_true)
            for bin_data in calibration_data:
                bin_weight = bin_data['count'] / total
                bin_error = abs(bin_data['predicted_prob'] - bin_data['actual_accuracy'])
                ece += bin_weight * bin_error
            
            metrics['expected_calibration_error'] = float(ece)
            
            # Confidence histogram
            confidence_bins = np.histogram(max_proba, bins=10, range=(0, 1))
            metrics['confidence_histogram'] = {
                'counts': confidence_bins[0].tolist(),
                'bin_edges': confidence_bins[1].tolist()
            }
            
            # Average confidence
            metrics['avg_confidence'] = float(max_proba.mean())
            metrics['avg_confidence_correct'] = float(max_proba[correct == 1].mean()) if (correct == 1).sum() > 0 else 0.0
            metrics['avg_confidence_incorrect'] = float(max_proba[correct == 0].mean()) if (correct == 0).sum() > 0 else 0.0
            
        except Exception as e:
            metrics['calibration_error'] = str(e)
        
        return metrics
    
    def _error_analysis(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Detailed error analysis."""
        errors = y_true != y_pred
        
        analysis = {
            'total_errors': int(errors.sum()),
            'error_rate': float(errors.mean()),
            'error_by_class': {},
            'common_misclassifications': []
        }
        
        # Errors per class
        for i in range(5):
            class_mask = y_true == i
            if class_mask.sum() > 0:
                class_errors = (y_pred[class_mask] != i).sum()
                analysis['error_by_class'][f'esi_{i+1}'] = {
                    'total': int(class_mask.sum()),
                    'errors': int(class_errors),
                    'error_rate': float(class_errors / class_mask.sum())
                }
        
        # Common misclassifications
        cm = confusion_matrix(y_true, y_pred)
        misclass_pairs = []
        for i in range(5):
            for j in range(5):
                if i != j and cm[i, j] > 0:
                    misclass_pairs.append({
                        'true': f'ESI {i+1}',
                        'predicted': f'ESI {j+1}',
                        'count': int(cm[i, j]),
                        'direction': 'undertriage' if j > i else 'overtriage'
                    })
        
        # Sort by count
        misclass_pairs.sort(key=lambda x: x['count'], reverse=True)
        analysis['common_misclassifications'] = misclass_pairs[:10]
        
        return analysis
    
    def _print_report(self, metrics: Dict):
        """Print formatted evaluation report."""
        print("\n" + "=" * 60)
        print("RiskScope AI - Evaluation Report")
        print("=" * 60)
        
        print("\n[CHART] OVERALL METRICS")
        print(f"   Accuracy:       {metrics['accuracy']:.1%}")
        print(f"   Cohen's Kappa:  {metrics['cohens_kappa']:.3f}")
        print(f"   F1 (weighted):  {metrics['f1_weighted']:.3f}")
        print(f"   Adjacent Acc:   {metrics['adjacent_accuracy']:.1%}")
        
        print("\n[ALERT] SAFETY METRICS")
        print(f"   ESI 1-2 Sensitivity: {metrics.get('esi_12_sensitivity', 0):.1%}")
        print(f"   ESI 1-2 Specificity: {metrics.get('esi_12_specificity', 0):.1%}")
        print(f"   Critical Under-triage: {metrics['critical_undertriage_count']} "
              f"({metrics['critical_undertriage_rate']:.2%})")
        print(f"   Severe Under-triage (ESI 1->4/5): {metrics['severe_undertriage_count']}")
        print(f"   Over-triage: {metrics['overtriage_count']} ({metrics['overtriage_rate']:.2%})")
        
        print("\n[UP] PER-CLASS SENSITIVITY")
        for i in range(5):
            sens = metrics.get(f'esi_{i+1}_sensitivity', 0)
            count = metrics.get(f'esi_{i+1}_count', 0)
            bar = "#" * int(sens * 20)
            print(f"   ESI {i+1}: {bar:<20} {sens:.1%} (n={count})")
        
        # AUC-ROC per class (if available)
        if 'auc_roc_macro' in metrics:
            print("\n[ROC] AUC-ROC SCORES (One-vs-Rest)")
            for i in range(5):
                auc_key = f'auc_roc_esi_{i+1}'
                if auc_key in metrics and metrics[auc_key] is not None:
                    auc_val = metrics[auc_key]
                    bar = "#" * int(auc_val * 20)
                    print(f"   ESI {i+1}: {bar:<20} {auc_val:.3f}")
            print(f"   Macro Avg: {metrics.get('auc_roc_macro', 0):.3f}")
        
        # Calibration metrics (if available)
        if 'expected_calibration_error' in metrics:
            print("\n[CAL] CALIBRATION METRICS")
            print(f"   Expected Calibration Error (ECE): {metrics['expected_calibration_error']:.3f}")
            print(f"   Avg Confidence (correct):   {metrics.get('avg_confidence_correct', 0):.1%}")
            print(f"   Avg Confidence (incorrect): {metrics.get('avg_confidence_incorrect', 0):.1%}")
        
        # Inference latency (if available)
        if 'inference_latency_ms' in metrics:
            print(f"\n[TIME] INFERENCE LATENCY")
            print(f"   Mean: {metrics['inference_latency_ms']:.2f} ms")
            print(f"   P95:  {metrics.get('inference_latency_p95_ms', 0):.2f} ms")
            print(f"   Max:  {metrics.get('inference_latency_max_ms', 0):.2f} ms")
        
        print("\n[CM] CONFUSION MATRIX (Counts)")
        cm = metrics['confusion_matrix']
        print("        Predicted")
        print("        " + " ".join([f"ESI{i+1:2d}" for i in range(5)]))
        print("   " + "-" * 35)
        for i in range(5):
            row = " ".join([f"{cm[i,j]:5d}" for j in range(5)])
            print(f"  ESI{i+1} |{row}")
        
        # Normalized confusion matrix
        cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
        print("\n[CM] CONFUSION MATRIX (Normalized %)")
        print("        Predicted")
        print("        " + " ".join([f"ESI{i+1:2d}" for i in range(5)]))
        print("   " + "-" * 35)
        for i in range(5):
            row = " ".join([f"{cm_norm[i,j]*100:5.1f}" for j in range(5)])
            print(f"  ESI{i+1} |{row}")
        
        print("\n[!] TOP MISCLASSIFICATIONS")
        for mc in metrics['error_analysis']['common_misclassifications'][:5]:
            icon = "v" if mc['direction'] == 'undertriage' else "^"
            print(f"   {icon} {mc['true']} -> {mc['predicted']}: {mc['count']} cases")
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                             save_path: str = None):
        """Plot confusion matrix heatmap."""
        # Convert to 0-indexed
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        if y_true.min() == 1:
            y_true = y_true - 1
        if y_pred.min() == 1:
            y_pred = y_pred - 1
        
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalize
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   ax=ax)
        
        ax.set_xlabel('Predicted ESI Level')
        ax.set_ylabel('Actual ESI Level')
        ax.set_title('RiskScope AI - Confusion Matrix')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Saved confusion matrix to {save_path}")
        
        return fig
    
    def plot_error_distribution(self, y_true: np.ndarray, y_pred: np.ndarray,
                                save_path: str = None):
        """Plot distribution of prediction errors."""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        errors = y_pred - y_true
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Error distribution
        ax1 = axes[0]
        error_counts = pd.Series(errors).value_counts().sort_index()
        colors = ['red' if e < 0 else 'green' if e > 0 else 'gray' for e in error_counts.index]
        ax1.bar(error_counts.index, error_counts.values, color=colors, edgecolor='black')
        ax1.set_xlabel('Prediction Error (Predicted - Actual)')
        ax1.set_ylabel('Count')
        ax1.set_title('Distribution of Prediction Errors\n(Red = Over-triage, Green = Under-triage)')
        ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        # Error by actual class
        ax2 = axes[1]
        error_by_class = []
        for i in range(5):
            mask = y_true == i
            if mask.sum() > 0:
                class_errors = y_pred[mask] - y_true[mask]
                for e in class_errors:
                    error_by_class.append({'Actual ESI': f'ESI {i+1}', 'Error': e})
        
        df = pd.DataFrame(error_by_class)
        sns.boxplot(x='Actual ESI', y='Error', data=df, ax=ax2)
        ax2.set_title('Prediction Errors by Actual ESI Level')
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Saved error distribution to {save_path}")
        
        return fig


def measure_inference_latency(model, X_test: np.ndarray, n_iterations: int = 100) -> Dict[str, float]:
    """
    Measure inference latency for production-readiness assessment.
    
    Parameters:
    -----------
    model : Trained ESITriagePredictor
    X_test : Test features (will use single samples)
    n_iterations : Number of predictions to time
    
    Returns:
    --------
    latency_metrics : Dict with mean, p95, max latency in milliseconds
    """
    latencies = []
    
    # Use single samples to simulate real-time inference
    n_samples = min(n_iterations, len(X_test))
    
    for i in range(n_samples):
        # Get single sample
        if hasattr(X_test, 'iloc'):
            sample = X_test.iloc[[i]].values
        else:
            sample = X_test[i:i+1]
        
        # Time the prediction
        start = time.perf_counter()
        _ = model.predict(sample)
        end = time.perf_counter()
        
        latencies.append((end - start) * 1000)  # Convert to ms
    
    latencies = np.array(latencies)
    
    return {
        'inference_latency_ms': float(np.mean(latencies)),
        'inference_latency_std_ms': float(np.std(latencies)),
        'inference_latency_p95_ms': float(np.percentile(latencies, 95)),
        'inference_latency_max_ms': float(np.max(latencies)),
        'inference_latency_min_ms': float(np.min(latencies)),
        'n_samples_timed': n_samples
    }


def evaluate_demo_cases(model, demo_cases_path: str = None, safety_engine=None) -> Dict[str, Any]:
    """
    Evaluate model on demo cases from demo_cases.json.
    
    These are clear-cut cases where we expect 100% accuracy.
    
    Parameters:
    -----------
    model : Trained ESITriagePredictor or None (if using safety_engine)
    demo_cases_path : Path to demo_cases.json
    safety_engine : ClinicalSafetyEngine instance (optional, for rule-based cases)
    
    Returns:
    --------
    results : Dict with per-case results and overall accuracy
    """
    # Default path
    if demo_cases_path is None:
        demo_cases_path = os.path.join(os.path.dirname(__file__), '..', 'demo_cases.json')
        if not os.path.exists(demo_cases_path):
            demo_cases_path = os.path.join(os.path.dirname(__file__), 'demo_cases.json')
    
    # Try to load demo cases
    if not os.path.exists(demo_cases_path):
        # Use built-in demo cases
        demo_cases = [
            {
                "name": "ESI 1 - Cardiac Arrest Signs",
                "expected_esi": 1,
                "patient": {
                    "age": 65, "gender": "M",
                    "heart_rate": 30, "bp_systolic": 70, "bp_diastolic": 40,
                    "spo2": 75, "temperature": 35.5, "respiratory_rate": 6,
                    "chest_pain": True, "altered_mental_status": True
                }
            },
            {
                "name": "ESI 1 - Stroke FAST+",
                "expected_esi": 1,
                "patient": {
                    "age": 72, "gender": "F",
                    "heart_rate": 88, "bp_systolic": 180, "bp_diastolic": 110,
                    "spo2": 96, "temperature": 36.8, "respiratory_rate": 18,
                    "facial_droop": True, "arm_weakness": True, "speech_difficulty": True
                }
            },
            {
                "name": "ESI 1 - Respiratory Failure",
                "expected_esi": 1,
                "patient": {
                    "age": 58, "gender": "M",
                    "heart_rate": 120, "bp_systolic": 100, "bp_diastolic": 60,
                    "spo2": 82, "temperature": 38.5, "respiratory_rate": 32,
                    "dyspnea": True, "shortness_of_breath": True
                }
            },
            {
                "name": "ESI 2 - Chest Pain Stable",
                "expected_esi": 2,
                "patient": {
                    "age": 55, "gender": "M",
                    "heart_rate": 95, "bp_systolic": 150, "bp_diastolic": 90,
                    "spo2": 96, "temperature": 37.0, "respiratory_rate": 18,
                    "chest_pain": True
                }
            },
            {
                "name": "ESI 3 - Abdominal Pain",
                "expected_esi": 3,
                "patient": {
                    "age": 35, "gender": "F",
                    "heart_rate": 85, "bp_systolic": 120, "bp_diastolic": 80,
                    "spo2": 98, "temperature": 37.2, "respiratory_rate": 16,
                    "abdominal_pain": True, "nausea": True
                }
            },
            {
                "name": "ESI 5 - Common Cold",
                "expected_esi": 5,
                "patient": {
                    "age": 25, "gender": "F",
                    "heart_rate": 70, "bp_systolic": 110, "bp_diastolic": 70,
                    "spo2": 99, "temperature": 37.3, "respiratory_rate": 14,
                    "cough": True
                }
            }
        ]
    else:
        with open(demo_cases_path, 'r') as f:
            demo_cases = json.load(f)
    
    results = {
        'cases': [],
        'total': len(demo_cases),
        'correct': 0,
        'incorrect': 0,
        'accuracy': 0.0
    }
    
    print("\n" + "=" * 60)
    print("DEMO CASE EVALUATION")
    print("=" * 60)
    
    for case in demo_cases:
        name = case.get('name', 'Unknown Case')
        expected = case.get('expected_esi', case.get('expected', 3))
        patient = case.get('patient', case)
        
        # Get prediction
        try:
            if safety_engine is not None:
                # Use full triage pipeline (rules + ML)
                triage_result = safety_engine.triage(patient)
                predicted = triage_result['esi_level']
                method = triage_result.get('method', 'UNKNOWN')
            elif model is not None:
                # Direct ML prediction
                from safety_engine import patient_to_raw_features
                features = patient_to_raw_features(patient)
                predicted = model.predict(features)[0]
                method = 'ML_DIRECT'
            else:
                predicted = 3
                method = 'NO_MODEL'
            
            correct = (predicted == expected)
            if correct:
                results['correct'] += 1
                status = "[OK]"
            else:
                results['incorrect'] += 1
                status = "[FAIL]"
            
            results['cases'].append({
                'name': name,
                'expected': expected,
                'predicted': predicted,
                'method': method,
                'correct': correct
            })
            
            print(f"  {status} {name}")
            print(f"       Expected: ESI {expected}, Predicted: ESI {predicted} ({method})")
            
        except Exception as e:
            results['incorrect'] += 1
            results['cases'].append({
                'name': name,
                'expected': expected,
                'predicted': None,
                'error': str(e),
                'correct': False
            })
            print(f"  [ERR] {name}: {e}")
    
    results['accuracy'] = results['correct'] / results['total'] if results['total'] > 0 else 0
    
    print(f"\n  DEMO ACCURACY: {results['correct']}/{results['total']} ({results['accuracy']:.0%})")
    
    # Check for 100% accuracy on clear-cut cases
    if results['accuracy'] == 1.0:
        print("  [OK] 100% accuracy on clear-cut demo cases!")
    else:
        print("  [!] Some demo cases were misclassified")
    
    return results


def run_evaluation(model, X_test, y_test, output_dir: str = None, 
                   measure_latency: bool = True):
    """
    Run full evaluation pipeline with all Phase 4 metrics.
    
    Parameters:
    -----------
    model : Trained ESITriagePredictor
    X_test : Test features
    y_test : True labels
    output_dir : Directory to save plots (optional)
    measure_latency : Whether to measure inference latency
    
    Returns:
    --------
    metrics : Dict with all evaluation metrics
    """
    evaluator = ESIEvaluator()
    
    # Get predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Run core evaluation
    metrics = evaluator.evaluate(y_test, y_pred, y_proba, verbose=True)
    
    # Measure inference latency
    if measure_latency:
        print("\n[TIME] Measuring inference latency...")
        latency_metrics = measure_inference_latency(model, X_test)
        metrics.update(latency_metrics)
        print(f"   Mean: {latency_metrics['inference_latency_ms']:.2f} ms")
        print(f"   P95:  {latency_metrics['inference_latency_p95_ms']:.2f} ms")
    
    # Generate plots if output directory specified
    if output_dir:
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            os.makedirs(output_dir, exist_ok=True)
            
            evaluator.plot_confusion_matrix(
                y_test, y_pred, 
                save_path=os.path.join(output_dir, 'confusion_matrix.png')
            )
            
            evaluator.plot_error_distribution(
                y_test, y_pred,
                save_path=os.path.join(output_dir, 'error_distribution.png')
            )
            
            # Plot calibration curve if available
            if 'calibration_curve' in metrics:
                plot_calibration_curve(metrics['calibration_curve'], 
                                      save_path=os.path.join(output_dir, 'calibration_curve.png'))
            
        except ImportError:
            print("  [!] matplotlib/seaborn not available, skipping plots")
    
    return metrics


def plot_calibration_curve(calibration_data: List[Dict], save_path: str = None):
    """Plot calibration curve showing predicted probability vs actual accuracy."""
    try:
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Extract data
        predicted = [d['predicted_prob'] for d in calibration_data]
        actual = [d['actual_accuracy'] for d in calibration_data]
        counts = [d['count'] for d in calibration_data]
        
        # Plot calibration curve
        ax.plot(predicted, actual, 'o-', label='Model', markersize=8)
        ax.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
        
        # Add count annotations
        for p, a, c in zip(predicted, actual, counts):
            ax.annotate(f'n={c}', (p, a), textcoords="offset points", 
                       xytext=(0, 10), ha='center', fontsize=8)
        
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives (Accuracy)')
        ax.set_title('RiskScope AI - Calibration Curve')
        ax.legend(loc='lower right')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Saved calibration curve to {save_path}")
        
        return fig
    except ImportError:
        print("  [!] matplotlib not available for calibration plot")
        return None


def run_full_benchmark(model, X_test, y_test, safety_engine=None, 
                       demo_cases_path: str = None, output_dir: str = None):
    """
    Run complete benchmark including all Phase 4 metrics.
    
    This is the main entry point for comprehensive model evaluation.
    
    Parameters:
    -----------
    model : Trained ESITriagePredictor
    X_test : Test features
    y_test : True labels
    safety_engine : ClinicalSafetyEngine for demo case evaluation
    demo_cases_path : Path to demo_cases.json
    output_dir : Directory to save outputs
    
    Returns:
    --------
    benchmark : Dict with all metrics and demo results
    """
    print("=" * 60)
    print("RiskScope AI - Full Benchmark Suite")
    print("=" * 60)
    
    benchmark = {}
    
    # 1. Core evaluation with AUC-ROC and calibration
    benchmark['evaluation'] = run_evaluation(model, X_test, y_test, 
                                             output_dir=output_dir, 
                                             measure_latency=True)
    
    # 2. Demo case evaluation
    benchmark['demo_cases'] = evaluate_demo_cases(model, demo_cases_path, safety_engine)
    
    # 3. Summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    
    eval_metrics = benchmark['evaluation']
    print(f"\n  Core Metrics:")
    print(f"    Cohen's Kappa:        {eval_metrics['cohens_kappa']:.3f}")
    print(f"    ESI 1-2 Sensitivity:  {eval_metrics.get('esi_12_sensitivity', 0):.1%}")
    print(f"    AUC-ROC (macro):      {eval_metrics.get('auc_roc_macro', 0):.3f}")
    print(f"    Calibration (ECE):    {eval_metrics.get('expected_calibration_error', 0):.3f}")
    
    print(f"\n  Safety Metrics:")
    print(f"    Under-triage rate:    {eval_metrics['critical_undertriage_rate']:.2%}")
    print(f"    Over-triage rate:     {eval_metrics['overtriage_rate']:.2%}")
    
    print(f"\n  Production Metrics:")
    print(f"    Inference latency:    {eval_metrics.get('inference_latency_ms', 0):.2f} ms")
    print(f"    Demo case accuracy:   {benchmark['demo_cases']['accuracy']:.0%}")
    
    # Save benchmark results
    if output_dir:
        benchmark_path = os.path.join(output_dir, 'benchmark_results.json')
        
        # Convert numpy types to Python types for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(i) for i in obj]
            return obj
        
        serializable = convert_for_json(benchmark)
        with open(benchmark_path, 'w') as f:
            json.dump(serializable, f, indent=2)
        print(f"\n  Benchmark saved to: {benchmark_path}")
    
    return benchmark


if __name__ == '__main__':
    print("RiskScope AI - Evaluation Module")
    print("Usage: from evaluate import run_full_benchmark")
    print("       benchmark = run_full_benchmark(model, X_test, y_test)")

"""
RiskScope AI - Model Evaluation Module
Comprehensive evaluation metrics and error analysis for ESI prediction
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from sklearn.metrics import (
    confusion_matrix, classification_report, cohen_kappa_score,
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, precision_recall_curve, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns


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
        
        # Confusion matrix
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
        
        print("\n📊 OVERALL METRICS")
        print(f"   Accuracy:       {metrics['accuracy']:.1%}")
        print(f"   Cohen's Kappa:  {metrics['cohens_kappa']:.3f}")
        print(f"   F1 (weighted):  {metrics['f1_weighted']:.3f}")
        print(f"   Adjacent Acc:   {metrics['adjacent_accuracy']:.1%}")
        
        print("\n🚨 SAFETY METRICS")
        print(f"   ESI 1-2 Sensitivity: {metrics.get('esi_12_sensitivity', 0):.1%}")
        print(f"   ESI 1-2 Specificity: {metrics.get('esi_12_specificity', 0):.1%}")
        print(f"   Critical Under-triage: {metrics['critical_undertriage_count']} "
              f"({metrics['critical_undertriage_rate']:.2%})")
        print(f"   Severe Under-triage (ESI 1→4/5): {metrics['severe_undertriage_count']}")
        print(f"   Over-triage: {metrics['overtriage_count']} ({metrics['overtriage_rate']:.2%})")
        
        print("\n📈 PER-CLASS SENSITIVITY")
        for i in range(5):
            sens = metrics.get(f'esi_{i+1}_sensitivity', 0)
            count = metrics.get(f'esi_{i+1}_count', 0)
            bar = "█" * int(sens * 20)
            print(f"   ESI {i+1}: {bar:<20} {sens:.1%} (n={count})")
        
        print("\n🔢 CONFUSION MATRIX")
        cm = metrics['confusion_matrix']
        print("        Predicted")
        print("        " + " ".join([f"ESI{i+1:2d}" for i in range(5)]))
        print("   " + "-" * 35)
        for i in range(5):
            row = " ".join([f"{cm[i,j]:5d}" for j in range(5)])
            print(f"  ESI{i+1} |{row}")
        
        print("\n⚠️  TOP MISCLASSIFICATIONS")
        for mc in metrics['error_analysis']['common_misclassifications'][:5]:
            icon = "↓" if mc['direction'] == 'undertriage' else "↑"
            print(f"   {icon} {mc['true']} → {mc['predicted']}: {mc['count']} cases")
    
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


def run_evaluation(model, X_test, y_test, output_dir: str = None):
    """
    Run full evaluation pipeline.
    
    Parameters:
    -----------
    model : Trained ESITriagePredictor
    X_test : Test features
    y_test : True labels
    output_dir : Directory to save plots (optional)
    """
    evaluator = ESIEvaluator()
    
    # Get predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Run evaluation
    metrics = evaluator.evaluate(y_test, y_pred, y_proba, verbose=True)
    
    # Generate plots if output directory specified
    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        evaluator.plot_confusion_matrix(
            y_test, y_pred, 
            save_path=os.path.join(output_dir, 'confusion_matrix.png')
        )
        
        evaluator.plot_error_distribution(
            y_test, y_pred,
            save_path=os.path.join(output_dir, 'error_distribution.png')
        )
    
    return metrics


if __name__ == '__main__':
    # Example usage
    print("ESI Evaluator - Run from train.py or evaluate.py")

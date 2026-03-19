"""
Validator Skill
===============
Validate model performance and experiment reproducibility.

USER-CONTROLLED: Only the user can modify this skill.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

# Import skill registry
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from agent.base_agent import skills


# ============================================================================
# VALIDATION THRESHOLDS
# ============================================================================

DEFAULT_THRESHOLDS = {
    "mcc": 0.3,           # Minimum acceptable MCC
    "auc": 0.7,           # Minimum acceptable AUC
    "sensitivity": 0.6,   # Minimum sensitivity
    "specificity": 0.6,   # Minimum specificity
    "sample_size": 100    # Minimum sample size
}


# ============================================================================
# SKILL FUNCTIONS
# ============================================================================

@skills.register("validate_predictions", "Validate model predictions")
def validate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray = None
) -> Dict[str, Any]:
    """
    Validate model predictions with comprehensive metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional)

    Returns:
        Validation metrics and assessment
    """
    try:
        from sklearn.metrics import (
            matthews_corrcoef,
            roc_auc_score,
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            confusion_matrix,
            classification_report
        )
    except ImportError:
        return {"status": "error", "message": "scikit-learn not installed"}

    metrics = {
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0))
    }

    # Calculate specificity from confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0
    metrics["confusion_matrix"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}

    # AUC if probabilities provided
    if y_prob is not None:
        metrics["auc"] = float(roc_auc_score(y_true, y_prob))

    # Assessment
    assessment = {
        "mcc_pass": metrics["mcc"] >= DEFAULT_THRESHOLDS["mcc"],
        "sensitivity_pass": metrics["sensitivity"] >= DEFAULT_THRESHOLDS["sensitivity"],
        "specificity_pass": metrics["specificity"] >= DEFAULT_THRESHOLDS["specificity"]
    }

    if y_prob is not None:
        assessment["auc_pass"] = metrics.get("auc", 0) >= DEFAULT_THRESHOLDS["auc"]

    assessment["overall_pass"] = all(assessment.values())

    return {
        "status": "success",
        "metrics": metrics,
        "assessment": assessment,
        "thresholds": DEFAULT_THRESHOLDS,
        "n_samples": len(y_true)
    }


@skills.register("validate_data_quality", "Validate dataset quality")
def validate_data_quality(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str] = None
) -> Dict[str, Any]:
    """
    Validate data quality for ML.

    Args:
        X: Feature matrix
        y: Labels
        feature_names: Feature names (optional)

    Returns:
        Data quality assessment
    """
    issues = []
    warnings = []

    # Check sample size
    n_samples = X.shape[0]
    if n_samples < DEFAULT_THRESHOLDS["sample_size"]:
        issues.append(f"Sample size ({n_samples}) below minimum ({DEFAULT_THRESHOLDS['sample_size']})")

    # Check class balance
    unique, counts = np.unique(y, return_counts=True)
    class_ratio = min(counts) / max(counts)
    if class_ratio < 0.2:
        warnings.append(f"Imbalanced classes: ratio = {class_ratio:.2f}")

    # Check for NaN/Inf
    nan_count = np.sum(np.isnan(X))
    inf_count = np.sum(np.isinf(X))
    if nan_count > 0:
        issues.append(f"NaN values found: {nan_count}")
    if inf_count > 0:
        issues.append(f"Inf values found: {inf_count}")

    # Check for constant features
    constant_features = np.sum(np.std(X, axis=0) == 0)
    if constant_features > 0:
        warnings.append(f"Constant features: {constant_features}")

    # Check for highly correlated features
    if X.shape[1] < 1000:  # Skip for high-dimensional
        try:
            corr_matrix = np.corrcoef(X.T)
            high_corr = np.sum(np.abs(corr_matrix) > 0.95) - X.shape[1]  # Exclude diagonal
            if high_corr > 0:
                warnings.append(f"Highly correlated feature pairs: {high_corr // 2}")
        except:
            pass

    return {
        "status": "success",
        "n_samples": n_samples,
        "n_features": X.shape[1],
        "n_classes": len(unique),
        "class_distribution": dict(zip(unique.tolist(), counts.tolist())),
        "class_ratio": float(class_ratio),
        "issues": issues,
        "warnings": warnings,
        "is_valid": len(issues) == 0
    }


@skills.register("validate_cv_stability", "Validate cross-validation stability")
def validate_cv_stability(
    cv_scores: List[float],
    metric_name: str = "mcc"
) -> Dict[str, Any]:
    """
    Validate stability of cross-validation results.

    Args:
        cv_scores: List of CV fold scores
        metric_name: Name of the metric

    Returns:
        Stability assessment
    """
    scores = np.array(cv_scores)

    mean_score = float(np.mean(scores))
    std_score = float(np.std(scores))
    min_score = float(np.min(scores))
    max_score = float(np.max(scores))
    cv_coefficient = std_score / mean_score if mean_score != 0 else float('inf')

    # Stability assessment
    is_stable = cv_coefficient < 0.2  # CV < 20%
    has_outliers = (max_score - min_score) > 3 * std_score

    warnings = []
    if not is_stable:
        warnings.append(f"High variability in CV scores (CV = {cv_coefficient:.2f})")
    if has_outliers:
        warnings.append("Potential outlier folds detected")
    if min_score < 0:
        warnings.append(f"Negative {metric_name} in some folds")

    return {
        "status": "success",
        "metric": metric_name,
        "mean": mean_score,
        "std": std_score,
        "min": min_score,
        "max": max_score,
        "cv_coefficient": cv_coefficient,
        "n_folds": len(scores),
        "is_stable": is_stable,
        "warnings": warnings
    }


@skills.register("validate_external", "Validate on external test set")
def validate_external(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    dataset_name: str = "external"
) -> Dict[str, Any]:
    """
    Validate model on external test set.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        dataset_name: Name of external dataset

    Returns:
        External validation metrics
    """
    try:
        y_pred = model.predict(X_test)

        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = None

        validation_result = validate_predictions(y_test, y_pred, y_prob)
        validation_result["dataset"] = dataset_name
        validation_result["validation_type"] = "external"

        return validation_result

    except Exception as e:
        return {"status": "error", "message": str(e)}


@skills.register("compare_to_baseline", "Compare model to baseline performance")
def compare_to_baseline(
    model_metrics: Dict[str, float],
    baseline_name: str = "random"
) -> Dict[str, Any]:
    """
    Compare model performance to baseline.

    Args:
        model_metrics: Dictionary of model metrics
        baseline_name: Type of baseline ('random', 'majority', 'deepdili')

    Returns:
        Comparison results
    """
    baselines = {
        "random": {
            "mcc": 0.0,
            "auc": 0.5,
            "accuracy": 0.5
        },
        "majority": {
            "mcc": 0.0,
            "auc": 0.5,
            "accuracy": 0.6  # Typical for imbalanced DILI data
        },
        "deepdili": {
            "mcc": 0.45,  # Approximate from literature
            "auc": 0.80,
            "accuracy": 0.75
        }
    }

    baseline = baselines.get(baseline_name, baselines["random"])

    improvements = {}
    for metric, baseline_value in baseline.items():
        if metric in model_metrics:
            model_value = model_metrics[metric]
            if baseline_value != 0:
                improvement = ((model_value - baseline_value) / abs(baseline_value)) * 100
            else:
                improvement = model_value * 100  # Percentage points
            improvements[metric] = {
                "model": model_value,
                "baseline": baseline_value,
                "improvement_pct": improvement
            }

    # Overall assessment
    beats_baseline = all(
        improvements.get(m, {}).get("model", 0) >= improvements.get(m, {}).get("baseline", 0)
        for m in ["mcc", "auc"]
    )

    return {
        "status": "success",
        "baseline_name": baseline_name,
        "comparisons": improvements,
        "beats_baseline": beats_baseline
    }


@skills.register("generate_validation_report", "Generate validation report")
def generate_validation_report(
    validation_results: List[Dict[str, Any]],
    output_path: str = None
) -> Dict[str, Any]:
    """
    Generate comprehensive validation report.

    Args:
        validation_results: List of validation results
        output_path: Path to save report (optional)

    Returns:
        Report summary
    """
    report = {
        "generated": datetime.now().isoformat(),
        "n_validations": len(validation_results),
        "summary": {},
        "results": validation_results
    }

    # Aggregate metrics
    mcc_values = [r.get("metrics", {}).get("mcc", 0) for r in validation_results if r.get("status") == "success"]
    auc_values = [r.get("metrics", {}).get("auc", 0) for r in validation_results if r.get("status") == "success" and "auc" in r.get("metrics", {})]

    if mcc_values:
        report["summary"]["avg_mcc"] = float(np.mean(mcc_values))
        report["summary"]["best_mcc"] = float(max(mcc_values))

    if auc_values:
        report["summary"]["avg_auc"] = float(np.mean(auc_values))
        report["summary"]["best_auc"] = float(max(auc_values))

    # Pass/fail summary
    pass_count = sum(1 for r in validation_results if r.get("assessment", {}).get("overall_pass", False))
    report["summary"]["pass_rate"] = pass_count / len(validation_results) if validation_results else 0

    # Save if path provided
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        report["saved_to"] = output_path

    return {
        "status": "success",
        "report": report
    }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def set_threshold(metric: str, value: float):
    """Set a validation threshold."""
    if metric in DEFAULT_THRESHOLDS:
        DEFAULT_THRESHOLDS[metric] = value


def get_thresholds() -> Dict[str, float]:
    """Get current validation thresholds."""
    return DEFAULT_THRESHOLDS.copy()

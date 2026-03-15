"""
Model Optimizer Skill
=====================
Train and optimize ML models for DILI prediction.

USER-CONTROLLED: Only the user can modify this skill.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict

# Import skill registry
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from agent.base_agent import skills


# ============================================================================
# MODEL CONFIGURATIONS
# ============================================================================

MODEL_CONFIGS = {
    "random_forest": {
        "class": "RandomForestClassifier",
        "params": {
            "n_estimators": [100, 200, 500],
            "max_depth": [5, 10, 20, None],
            "min_samples_split": [2, 5, 10],
            "class_weight": ["balanced"]
        }
    },
    "xgboost": {
        "class": "XGBClassifier",
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1, 0.3],
            "scale_pos_weight": [1, 2, 5]
        }
    },
    "gradient_boosting": {
        "class": "GradientBoostingClassifier",
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [3, 5],
            "learning_rate": [0.01, 0.1]
        }
    },
    "svm": {
        "class": "SVC",
        "params": {
            "C": [0.1, 1, 10],
            "kernel": ["rbf", "linear"],
            "gamma": ["scale", "auto"],
            "probability": [True],
            "class_weight": ["balanced"]
        }
    },
    "logistic_regression": {
        "class": "LogisticRegression",
        "params": {
            "C": [0.1, 1, 10],
            "class_weight": ["balanced"],
            "max_iter": [1000]
        }
    },
    "knn": {
        "class": "KNeighborsClassifier",
        "params": {
            "n_neighbors": [3, 5, 7, 11],
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan"]
        }
    },
    "mlp": {
        "class": "MLPClassifier",
        "params": {
            "hidden_layer_sizes": [(100,), (100, 50), (200, 100)],
            "activation": ["relu", "tanh"],
            "alpha": [0.0001, 0.001],
            "max_iter": [500]
        }
    }
}


# ============================================================================
# SKILL FUNCTIONS
# ============================================================================

@skills.register("train_model", "Train a single ML model")
def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_type: str = "random_forest",
    params: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Train a single ML model.

    Args:
        X_train: Training features
        y_train: Training labels
        model_type: Type of model to train
        params: Model hyperparameters

    Returns:
        Trained model info and performance
    """
    try:
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.svm import SVC
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.neural_network import MLPClassifier
    except ImportError:
        return {"status": "error", "message": "scikit-learn not installed"}

    model_classes = {
        "random_forest": RandomForestClassifier,
        "gradient_boosting": GradientBoostingClassifier,
        "svm": SVC,
        "logistic_regression": LogisticRegression,
        "knn": KNeighborsClassifier,
        "mlp": MLPClassifier
    }

    # Handle XGBoost separately
    if model_type == "xgboost":
        try:
            from xgboost import XGBClassifier
            model_classes["xgboost"] = XGBClassifier
        except ImportError:
            return {"status": "error", "message": "XGBoost not installed"}

    if model_type not in model_classes:
        return {"status": "error", "message": f"Unknown model type: {model_type}"}

    # Get default params if not provided
    if params is None:
        config = MODEL_CONFIGS.get(model_type, {})
        params = {k: v[0] if isinstance(v, list) else v
                  for k, v in config.get("params", {}).items()}

    try:
        model = model_classes[model_type](**params)
        model.fit(X_train, y_train)

        return {
            "status": "success",
            "model_type": model_type,
            "params": params,
            "model": model,  # Note: model object for in-memory use
            "n_samples": len(y_train),
            "n_features": X_train.shape[1]
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@skills.register("cross_validate", "Perform cross-validation")
def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = "random_forest",
    params: Dict[str, Any] = None,
    cv_folds: int = 5,
    cv_repeats: int = 3
) -> Dict[str, Any]:
    """
    Perform repeated stratified cross-validation.

    Args:
        X: Features
        y: Labels
        model_type: Model type
        params: Model parameters
        cv_folds: Number of CV folds
        cv_repeats: Number of CV repeats

    Returns:
        CV metrics including MCC, AUC, sensitivity, specificity
    """
    try:
        from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate as sklearn_cv
        from sklearn.metrics import make_scorer, matthews_corrcoef, roc_auc_score
    except ImportError:
        return {"status": "error", "message": "scikit-learn not installed"}

    # Get model
    train_result = train_model(X[:10], y[:10], model_type, params)  # Dummy fit for model object
    if train_result["status"] != "success":
        return train_result

    model = train_result["model"].__class__(**params if params else {})

    # Define scoring
    scoring = {
        'mcc': make_scorer(matthews_corrcoef),
        'accuracy': 'accuracy',
        'roc_auc': 'roc_auc',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1'
    }

    # Cross-validation
    cv = RepeatedStratifiedKFold(n_splits=cv_folds, n_repeats=cv_repeats, random_state=42)

    try:
        cv_results = sklearn_cv(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)

        metrics = {
            "mcc": {
                "mean": float(np.mean(cv_results['test_mcc'])),
                "std": float(np.std(cv_results['test_mcc']))
            },
            "auc": {
                "mean": float(np.mean(cv_results['test_roc_auc'])),
                "std": float(np.std(cv_results['test_roc_auc']))
            },
            "accuracy": {
                "mean": float(np.mean(cv_results['test_accuracy'])),
                "std": float(np.std(cv_results['test_accuracy']))
            },
            "sensitivity": {
                "mean": float(np.mean(cv_results['test_recall'])),
                "std": float(np.std(cv_results['test_recall']))
            },
            "precision": {
                "mean": float(np.mean(cv_results['test_precision'])),
                "std": float(np.std(cv_results['test_precision']))
            }
        }

        return {
            "status": "success",
            "model_type": model_type,
            "params": params,
            "cv_folds": cv_folds,
            "cv_repeats": cv_repeats,
            "metrics": metrics
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


@skills.register("hyperparameter_search", "Search for optimal hyperparameters")
def hyperparameter_search(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = "random_forest",
    search_method: str = "random",
    n_iter: int = 20,
    cv_folds: int = 5
) -> Dict[str, Any]:
    """
    Search for optimal hyperparameters.

    Args:
        X: Features
        y: Labels
        model_type: Model type
        search_method: 'grid' or 'random'
        n_iter: Number of iterations for random search
        cv_folds: CV folds for evaluation

    Returns:
        Best parameters and CV score
    """
    try:
        from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
        from sklearn.metrics import make_scorer, matthews_corrcoef
    except ImportError:
        return {"status": "error", "message": "scikit-learn not installed"}

    # Get model and param grid
    config = MODEL_CONFIGS.get(model_type)
    if config is None:
        return {"status": "error", "message": f"No config for {model_type}"}

    train_result = train_model(X[:10], y[:10], model_type, None)
    if train_result["status"] != "success":
        return train_result

    model = train_result["model"].__class__()
    param_grid = config["params"]

    scorer = make_scorer(matthews_corrcoef)

    try:
        if search_method == "random":
            search = RandomizedSearchCV(
                model, param_grid, n_iter=n_iter, cv=cv_folds,
                scoring=scorer, n_jobs=-1, random_state=42
            )
        else:
            search = GridSearchCV(
                model, param_grid, cv=cv_folds,
                scoring=scorer, n_jobs=-1
            )

        search.fit(X, y)

        return {
            "status": "success",
            "model_type": model_type,
            "best_params": search.best_params_,
            "best_mcc": float(search.best_score_),
            "search_method": search_method,
            "n_iterations": n_iter if search_method == "random" else None
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


@skills.register("train_ensemble", "Train ensemble of base models")
def train_ensemble(
    X: np.ndarray,
    y: np.ndarray,
    base_models: List[str] = None,
    meta_learner: str = "logistic_regression"
) -> Dict[str, Any]:
    """
    Train an ensemble with stacking.

    Args:
        X: Features
        y: Labels
        base_models: List of base model types
        meta_learner: Model type for meta-learner

    Returns:
        Ensemble model info
    """
    try:
        from sklearn.ensemble import StackingClassifier
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import make_scorer, matthews_corrcoef
    except ImportError:
        return {"status": "error", "message": "scikit-learn not installed"}

    if base_models is None:
        base_models = ["random_forest", "xgboost", "svm", "logistic_regression"]

    # Build estimators list
    estimators = []
    for model_type in base_models:
        result = train_model(X[:10], y[:10], model_type)
        if result["status"] == "success":
            estimators.append((model_type, result["model"].__class__()))

    if not estimators:
        return {"status": "error", "message": "No valid base models"}

    # Get meta-learner
    meta_result = train_model(X[:10], y[:10], meta_learner)
    if meta_result["status"] != "success":
        return {"status": "error", "message": f"Failed to create meta-learner: {meta_learner}"}

    try:
        ensemble = StackingClassifier(
            estimators=estimators,
            final_estimator=meta_result["model"].__class__(),
            cv=5,
            n_jobs=-1
        )

        # Cross-validate ensemble
        scorer = make_scorer(matthews_corrcoef)
        scores = cross_val_score(ensemble, X, y, cv=5, scoring=scorer)

        # Fit final ensemble
        ensemble.fit(X, y)

        return {
            "status": "success",
            "base_models": base_models,
            "meta_learner": meta_learner,
            "cv_mcc_mean": float(np.mean(scores)),
            "cv_mcc_std": float(np.std(scores)),
            "model": ensemble
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


@skills.register("compare_models", "Compare multiple model types")
def compare_models(
    X: np.ndarray,
    y: np.ndarray,
    model_types: List[str] = None,
    cv_folds: int = 5
) -> Dict[str, Any]:
    """
    Compare multiple model types on the same data.

    Args:
        X: Features
        y: Labels
        model_types: List of model types to compare
        cv_folds: CV folds

    Returns:
        Comparison results sorted by MCC
    """
    if model_types is None:
        model_types = list(MODEL_CONFIGS.keys())

    results = []

    for model_type in model_types:
        cv_result = cross_validate(X, y, model_type, cv_folds=cv_folds, cv_repeats=1)
        if cv_result["status"] == "success":
            results.append({
                "model_type": model_type,
                "mcc": cv_result["metrics"]["mcc"]["mean"],
                "auc": cv_result["metrics"]["auc"]["mean"],
                "accuracy": cv_result["metrics"]["accuracy"]["mean"]
            })

    # Sort by MCC
    results.sort(key=lambda x: x["mcc"], reverse=True)

    return {
        "status": "success",
        "comparison": results,
        "best_model": results[0]["model_type"] if results else None,
        "best_mcc": results[0]["mcc"] if results else 0
    }


@skills.register("save_model", "Save trained model to disk")
def save_model(
    model: Any,
    model_name: str,
    save_dir: str = "models"
) -> Dict[str, Any]:
    """
    Save a trained model to disk.

    Args:
        model: Trained model object
        model_name: Name for the saved model
        save_dir: Directory to save model

    Returns:
        Save path and info
    """
    import joblib

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    model_path = save_path / f"{model_name}.joblib"

    try:
        joblib.dump(model, model_path)
        return {
            "status": "success",
            "path": str(model_path),
            "model_name": model_name
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_available_models() -> Dict[str, Any]:
    """Get list of available model configurations."""
    return MODEL_CONFIGS

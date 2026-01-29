"""
Base model reproduction for the DeepDILI project.

This script implements a simplified version of the DeepDILI pipeline using
readily available Python libraries (scikit‑learn, XGBoost and TensorFlow/Keras).
It reads a CSV containing Mold2 descriptors and DILI labels, splits the data
chronologically, trains an ensemble of base learners (KNN, random forest,
SVM, gradient boosting and extra trees) and then feeds their predicted
probabilities into a small neural network meta‑learner.  Evaluation metrics
(AUPRC, MCC, sensitivity, specificity) are computed on the held‑out test set.

Note: This script is intended as a template.  It assumes that a
`data.csv` file exists with columns:
    - `DILI_label` (binary 0/1 for negative/positive DILI)
    - `final_year` (integer approval year)
    - descriptor columns (Mold2 features)

The descriptors should already be standardised or otherwise processed.  The
script performs its own scaling prior to modelling.  The script does not
handle missing values; users should impute or remove missing values prior to
running.

Usage:
    from base_model import run_base_model
    results = run_base_model('path/to/dataset.csv')
    print(results)

"""

from typing import Dict, Tuple, List, Any

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    matthews_corrcoef,
    confusion_matrix,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


def load_dataset(csv_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the dataset and split it chronologically.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing descriptors, DILI labels and final years.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        A tuple of (full data, training data, test data).
    """
    df = pd.read_csv(csv_path)
    # Ensure required columns exist
    if 'DILI_label' not in df.columns or 'final_year' not in df.columns:
        raise ValueError('Input CSV must contain DILI_label and final_year columns')
    train_df = df[df['final_year'] < 1997].copy().reset_index(drop=True)
    test_df = df[df['final_year'] >= 1997].copy().reset_index(drop=True)
    return df, train_df, test_df


def get_base_models() -> Dict[str, Any]:
    """Instantiate the base models used in the DeepDILI ensemble.

    Returns
    -------
    Dict[str, Any]
        A dictionary mapping model names to their instantiated sklearn objects.
    """
    models = {
        'knn': KNeighborsClassifier(n_neighbors=5),
        'random_forest': RandomForestClassifier(
            n_estimators=500,
            max_depth=11,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=42,
        ),
        'svm': SVC(
            kernel='rbf',
            probability=True,
            gamma='scale',
            class_weight='balanced',
            random_state=42,
        ),
        'xgboost': XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='logloss',
            use_label_encoder=False,
        ),
        'extra_trees': ExtraTreesClassifier(
            n_estimators=500,
            max_depth=10,
            min_samples_leaf=2,
            random_state=42,
        ),
    }
    return models


def train_base_models(
    models: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit base models and return their probability predictions on train and test sets.

    Parameters
    ----------
    models : Dict[str, Any]
        Dictionary of base classifiers.
    X_train : np.ndarray
        Training features (scaled).
    y_train : np.ndarray
        Training labels.
    X_test : np.ndarray
        Test features (scaled).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple of (train probabilities, test probabilities) with shape (n_samples, n_models).
    """
    train_probs = []
    test_probs = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        train_prob = model.predict_proba(X_train)[:, 1]
        test_prob = model.predict_proba(X_test)[:, 1]
        train_probs.append(train_prob)
        test_probs.append(test_prob)
    return np.column_stack(train_probs), np.column_stack(test_probs)


def build_meta_model(input_dim: int) -> Sequential:
    """Construct a simple feed‑forward neural network for meta‑learning.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input vector (number of base models).

    Returns
    -------
    Sequential
        A compiled Keras model.
    """
    model = Sequential()
    model.add(Dense(20, activation='relu', input_shape=(input_dim,)))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )
    return model


def evaluate_binary_predictions(y_true: np.ndarray, y_pred_prob: np.ndarray) -> Dict[str, float]:
    """Compute evaluation metrics for binary classification.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    y_pred_prob : np.ndarray
        Predicted probabilities for the positive class.

    Returns
    -------
    Dict[str, float]
        Dictionary with AUPRC, MCC, sensitivity and specificity.
    """
    # Threshold at 0.5 for class predictions
    y_pred = (y_pred_prob >= 0.5).astype(int)
    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
    auprc = average_precision_score(y_true, y_pred_prob)
    mcc = matthews_corrcoef(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        'AUPRC': auprc,
        'MCC': mcc,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
    }


def run_base_model(csv_path: str) -> Dict[str, Any]:
    """Execute the full DeepDILI base model pipeline.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing descriptors and labels.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing evaluation metrics and objects.
    """
    df, train_df, test_df = load_dataset(csv_path)
    # Separate features and labels
    # Filter out non-numeric columns (e.g., compound names) to avoid dtype errors
    feature_df_train = train_df.drop(columns=['DILI_label', 'final_year'])
    feature_df_test = test_df.drop(columns=['DILI_label', 'final_year'])
    # Select only numeric columns
    numeric_cols = feature_df_train.select_dtypes(include=[np.number]).columns.tolist()
    X_train = feature_df_train[numeric_cols].values
    y_train = train_df['DILI_label'].values
    X_test = feature_df_test[numeric_cols].values
    y_test = test_df['DILI_label'].values

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train base models
    base_models = get_base_models()
    train_probs, test_probs = train_base_models(base_models, X_train_scaled, y_train, X_test_scaled)

    # Train meta‑learner
    meta_model = build_meta_model(input_dim=train_probs.shape[1])
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    meta_model.fit(
        train_probs,
        y_train,
        epochs=200,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=0,
    )
    # Predict on test probabilities
    test_pred_prob = meta_model.predict(test_probs).flatten()

    # Evaluate
    metrics_dict = evaluate_binary_predictions(y_test, test_pred_prob)
    return {
        'metrics': metrics_dict,
        'base_models': base_models,
        'meta_model': meta_model,
        'scaler': scaler,
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run DeepDILI base model pipeline')
    parser.add_argument('csv_path', type=str, help='Path to input CSV file')
    args = parser.parse_args()
    results = run_base_model(args.csv_path)
    print('Evaluation metrics:')
    for metric, value in results['metrics'].items():
        print(f'{metric}: {value:.4f}')
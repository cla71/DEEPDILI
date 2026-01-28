"""
V3 model for the DeepDILI project – metabolic complexity.

This script extends the V2 model by adding cytochrome P450 (CYP) binding data
and an interaction term between CYP binding and cytotoxicity.  The expected
CYP features are numeric pIC50/pKi values for the three major drug‑metabolising
isoforms (CYP3A4, CYP2D6, CYP2C9).  Users may optionally provide binary
imputation flags for each CYP feature.  The interaction term is computed as
the product of the aggregated CYP binding score (sum of the three isoform
activities) and the cytotoxicity feature from V2.

Expected additional columns:
    - CYP3A4, CYP2D6, CYP2C9: numerical features (pIC50/pKi)
    - LC50_Cyto: reused from V2 for interaction term
    - Optional: CYP3A4_imputed, CYP2D6_imputed, CYP2C9_imputed (binary flags)

Usage:
    from v3_model import run_v3_model
    results = run_v3_model('path/to/dataset_with_mechanistic_and_cyp.csv')

"""

from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    matthews_corrcoef,
    confusion_matrix,
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from base_model import (
    load_dataset,
    get_base_models,
    train_base_models,
    evaluate_binary_predictions,
)


def run_v3_model(
    csv_path: str,
    bsep_col: str = 'IC50_BSEP',
    cyto_col: str = 'LC50_Cyto',
    bsep_flag_col: str = 'IC50_BSEP_imputed',
    cyto_flag_col: str = 'LC50_Cyto_imputed',
    cyp_cols: Tuple[str, str, str] = ('CYP3A4', 'CYP2D6', 'CYP2C9'),
    cyp_flag_cols: Tuple[str, str, str] = ('CYP3A4_imputed', 'CYP2D6_imputed', 'CYP2C9_imputed'),
) -> Dict[str, Any]:
    """Run the DeepDILI V3 model with CYP binding and interaction features.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing descriptors, mechanistic features,
        CYP binding data, labels and final years.
    bsep_col, cyto_col, bsep_flag_col, cyto_flag_col : str
        Column names for BSEP and cytotoxicity features and their imputation flags.
    cyp_cols : tuple of str
        Column names for CYP3A4, CYP2D6 and CYP2C9 pIC50/pKi values.
    cyp_flag_cols : tuple of str
        Column names for binary flags indicating whether CYP values were imputed.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing evaluation metrics and trained models.
    """
    df, train_df, test_df = load_dataset(csv_path)

    # Validate required columns
    mech_required = {bsep_col, cyto_col, bsep_flag_col, cyto_flag_col}
    cyp_required = set(cyp_cols)
    missing_mech = mech_required - set(df.columns)
    missing_cyp = cyp_required - set(df.columns)
    if missing_mech:
        raise ValueError(f'Missing mechanistic columns: {missing_mech}')
    if missing_cyp:
        raise ValueError(f'Missing CYP columns: {missing_cyp}')
    # Determine which CYP flag columns exist
    available_cyp_flags = [col for col in cyp_flag_cols if col in df.columns]

    # Base descriptor columns (exclude label, year, mechanistic and CYP columns and flags)
    excluded_cols = {
        'DILI_label', 'final_year',
        bsep_col, cyto_col, bsep_flag_col, cyto_flag_col,
        *cyp_cols,
        *available_cyp_flags,
    }
    base_cols = [col for col in df.columns if col not in excluded_cols]

    # Prepare training and test sets
    X_train_base = train_df[base_cols].values
    X_train_mech = train_df[[bsep_col, cyto_col]].values
    X_train_cyp = train_df[list(cyp_cols)].values
    X_train_flags = train_df[[bsep_flag_col, cyto_flag_col] + available_cyp_flags].values if available_cyp_flags else np.zeros((len(train_df), 0))
    y_train = train_df['DILI_label'].values

    X_test_base = test_df[base_cols].values
    X_test_mech = test_df[[bsep_col, cyto_col]].values
    X_test_cyp = test_df[list(cyp_cols)].values
    X_test_flags = test_df[[bsep_flag_col, cyto_flag_col] + available_cyp_flags].values if available_cyp_flags else np.zeros((len(test_df), 0))
    y_test = test_df['DILI_label'].values

    # Standardise each feature block separately
    scaler_base = StandardScaler()
    X_train_base_scaled = scaler_base.fit_transform(X_train_base)
    X_test_base_scaled = scaler_base.transform(X_test_base)

    scaler_mech = StandardScaler()
    X_train_mech_scaled = scaler_mech.fit_transform(X_train_mech)
    X_test_mech_scaled = scaler_mech.transform(X_test_mech)

    scaler_cyp = StandardScaler()
    X_train_cyp_scaled = scaler_cyp.fit_transform(X_train_cyp)
    X_test_cyp_scaled = scaler_cyp.transform(X_test_cyp)

    if X_train_flags.size > 0:
        scaler_flags = StandardScaler()
        X_train_flags_scaled = scaler_flags.fit_transform(X_train_flags)
        X_test_flags_scaled = scaler_flags.transform(X_test_flags)
    else:
        scaler_flags = None
        X_train_flags_scaled = X_train_flags
        X_test_flags_scaled = X_test_flags

    # Train base models on base descriptors to obtain probability vectors
    base_models = get_base_models()
    train_probs, test_probs = train_base_models(base_models, X_train_base_scaled, y_train, X_test_base_scaled)

    # Aggregate CYP binding score (sum) and compute interaction term
    train_cyp_sum = X_train_cyp_scaled.sum(axis=1, keepdims=True)
    test_cyp_sum = X_test_cyp_scaled.sum(axis=1, keepdims=True)
    # Interaction: cytotoxicity (scaled) * cyp_sum
    # cytotoxicity is second column in mech_scaled
    train_interaction = (X_train_mech_scaled[:, 1:2] * train_cyp_sum)
    test_interaction = (X_test_mech_scaled[:, 1:2] * test_cyp_sum)

    # Concatenate all features: base probabilities + mechanistic + CYP + flags + interaction
    train_ext = np.hstack([
        train_probs,
        X_train_mech_scaled,
        X_train_cyp_scaled,
        X_train_flags_scaled,
        train_interaction,
    ])
    test_ext = np.hstack([
        test_probs,
        X_test_mech_scaled,
        X_test_cyp_scaled,
        X_test_flags_scaled,
        test_interaction,
    ])

    # Build and train meta‑model
    input_dim = train_ext.shape[1]
    meta_model = Sequential()
    meta_model.add(Dense(48, activation='relu', input_shape=(input_dim,)))
    meta_model.add(Dropout(0.4))
    meta_model.add(Dense(24, activation='relu'))
    meta_model.add(Dropout(0.3))
    meta_model.add(Dense(12, activation='relu'))
    meta_model.add(Dropout(0.2))
    meta_model.add(Dense(1, activation='sigmoid'))
    meta_model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )
    early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    meta_model.fit(
        train_ext,
        y_train,
        epochs=400,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=0,
    )

    # Predict probabilities on test set
    y_pred_prob = meta_model.predict(test_ext).flatten()
    metrics_dict = evaluate_binary_predictions(y_test, y_pred_prob)
    return {
        'metrics': metrics_dict,
        'base_models': base_models,
        'meta_model': meta_model,
        'scaler_base': scaler_base,
        'scaler_mech': scaler_mech,
        'scaler_cyp': scaler_cyp,
        'scaler_flags': scaler_flags,
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run DeepDILI V3 model with CYP binding and interaction features')
    parser.add_argument('csv_path', type=str, help='Path to input CSV file with mechanistic and CYP features')
    args = parser.parse_args()
    results = run_v3_model(args.csv_path)
    print('Evaluation metrics:')
    for metric, value in results['metrics'].items():
        print(f'{metric}: {value:.4f}')
"""
V2 model for the DeepDILI project – mechanistic enhancement.

This script extends the base model by incorporating mechanistic features such as
bile salt export pump (BSEP) inhibition and human hepatocyte cytotoxicity.
The mechanistic features should be provided as additional columns in the input
CSV file, with names matching the default arguments below.  Missing values
should be imputed beforehand or indicated with binary flags.

Mechanistic features expected:
    - IC50_BSEP: numerical feature (e.g., pIC50 or -log IC50)
    - LC50_Cyto: numerical feature (e.g., -log LC50)
    - IC50_BSEP_imputed: binary flag (1 if value imputed, 0 otherwise)
    - LC50_Cyto_imputed: binary flag (1 if value imputed, 0 otherwise)

The script trains the same base learners as the base model to obtain
probability vectors and then appends the mechanistic features to produce an
extended feature vector for the meta‑learner.  Regularisation (dropout) is
employed to mitigate overfitting.

Usage:
    from v2_model import run_v2_model
    results = run_v2_model('path/to/dataset_with_mechanistic.csv')

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
    build_meta_model,
    evaluate_binary_predictions,
)


def run_v2_model(
    csv_path: str,
    bsep_col: str = 'IC50_BSEP',
    cyto_col: str = 'LC50_Cyto',
    bsep_flag_col: str = 'IC50_BSEP_imputed',
    cyto_flag_col: str = 'LC50_Cyto_imputed',
) -> Dict[str, Any]:
    """Run the DeepDILI V2 model with mechanistic features.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing descriptors, mechanistic features, labels and final years.
    bsep_col : str, optional
        Name of the BSEP inhibition feature column.
    cyto_col : str, optional
        Name of the human hepatocyte cytotoxicity feature column.
    bsep_flag_col : str, optional
        Name of the binary flag indicating whether BSEP value was imputed.
    cyto_flag_col : str, optional
        Name of the binary flag indicating whether cytotoxicity value was imputed.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing evaluation metrics and trained models.
    """
    df, train_df, test_df = load_dataset(csv_path)

    # Check mechanistic columns
    required_cols = {bsep_col, cyto_col, bsep_flag_col, cyto_flag_col}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f'Missing mechanistic columns: {missing_cols}')

    # Separate base descriptors and mechanistic features
    feature_cols = [col for col in df.columns if col not in {'DILI_label', 'final_year', bsep_col, cyto_col, bsep_flag_col, cyto_flag_col}]
    mech_cols = [bsep_col, cyto_col, bsep_flag_col, cyto_flag_col]

    # Training sets
    X_train_base = train_df[feature_cols].values
    X_train_mech = train_df[mech_cols].values
    y_train = train_df['DILI_label'].values
    X_test_base = test_df[feature_cols].values
    X_test_mech = test_df[mech_cols].values
    y_test = test_df['DILI_label'].values

    # Scale base features and mechanistic features separately
    scaler_base = StandardScaler()
    X_train_base_scaled = scaler_base.fit_transform(X_train_base)
    X_test_base_scaled = scaler_base.transform(X_test_base)

    scaler_mech = StandardScaler()
    X_train_mech_scaled = scaler_mech.fit_transform(X_train_mech)
    X_test_mech_scaled = scaler_mech.transform(X_test_mech)

    # Train base models to obtain probability vectors
    base_models = get_base_models()
    train_probs, test_probs = train_base_models(base_models, X_train_base_scaled, y_train, X_test_base_scaled)

    # Concatenate mechanistic features to probability vectors
    train_ext = np.hstack([train_probs, X_train_mech_scaled])
    test_ext = np.hstack([test_probs, X_test_mech_scaled])

    # Build a slightly larger meta‑model for the extended feature vector
    input_dim = train_ext.shape[1]
    meta_model = Sequential()
    meta_model.add(Dense(32, activation='relu', input_shape=(input_dim,)))
    meta_model.add(Dropout(0.3))
    meta_model.add(Dense(16, activation='relu'))
    meta_model.add(Dropout(0.2))
    meta_model.add(Dense(1, activation='sigmoid'))
    meta_model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )

    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    meta_model.fit(
        train_ext,
        y_train,
        epochs=300,
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
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run DeepDILI V2 model with mechanistic features')
    parser.add_argument('csv_path', type=str, help='Path to input CSV file with mechanistic features')
    args = parser.parse_args()
    results = run_v2_model(args.csv_path)
    print('Evaluation metrics:')
    for metric, value in results['metrics'].items():
        print(f'{metric}: {value:.4f}')
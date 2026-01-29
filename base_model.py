"""
Aligned DeepDILI Base Model
Refactored to match the original Ting Li et al. (FDA) implementation logic.
"""

import numpy as np
import pandas as pd
from functools import reduce
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (
    average_precision_score,
    matthews_corrcoef,
    confusion_matrix,
    recall_score
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

def get_base_classifiers_configs():
    """Returns the specific hyperparams used in the notebook cells."""
    return {
        'knn': KNeighborsClassifier(n_neighbors=7),
        'lr': LogisticRegression(C=0.1, max_iter=300, class_weight='balanced'),
        'svm': SVC(kernel='rbf', C=1, gamma='scale', probability=True, class_weight='balanced', random_state=1),
        'rf': RandomForestClassifier(n_estimators=700, max_depth=11, min_samples_leaf=5,
                                    class_weight='balanced', max_features='log2', random_state=1),
        'xgboost': XGBClassifier(learning_rate=0.01, n_estimators=700, max_depth=11,
                                 subsample=0.7, scale_pos_weight=0.66, eval_metric='logloss')
    }

def model_predict_probs(X, model):
    """Mirroring model_predict logic: extracts probability of positive class."""
    return model.predict_proba(X)[:, 1]

def run_deepdili_pipeline(train_csv, test_csv, train_index_path, selected_models_path):
    """
    Implements the logic from the 'DeepDILI predictions' cell.
    """
    # 1. Load Data
    data = pd.read_csv(train_csv)
    X_org = data.iloc[:, 4:]
    y_org = data['DILI_label']

    # 2. Chronological or Stratified Split (Paper uses chronological, Notebook uses random_state=7)
    # To replicate your 0.14 MCC issue, check if the paper's 1997 split is used vs this random split.
    from sklearn.model_selection import train_test_split
    X_train_full, X_val, y_train_full, y_val = train_test_split(
        X_org, y_org, test_size=0.2, stratify=y_org, random_state=7
    )

    external = pd.read_csv(test_csv)
    X_test = external[X_train_full.columns]

    # 3. Base Classifier Training (MLR Construction)
    # The notebook iterates through 'train_index_df' to train multiple versions of base models
    train_index_df = pd.read_csv(train_index_path)
    configs = get_base_classifiers_configs()

    # Dictionary to hold MLR (Model Level Representation) probabilities
    val_probs_list = []
    test_probs_list = []

    # Iterate through the model seeds/indices defined in the FDA training index
    for i, col_name in enumerate(train_index_df.columns[5:]):
        train_idx = train_index_df[train_index_df[col_name] == 1].id.unique()

        # Subset training data based on index
        X_train_sub = X_train_full[X_train_full.index.isin(train_idx)]
        y_train_sub = y_train_full[y_train_full.index.isin(train_idx)]

        # Scaling (Notebook uses MinMaxScaler for base models)
        sc = MinMaxScaler()
        X_tr_s = sc.fit_transform(X_train_sub)
        X_val_s = sc.transform(X_val)
        X_te_s = sc.transform(X_test)

        # Train each type and collect probs
        row_val, row_te = {}, {}
        for name, clf in configs.items():
            clf.fit(X_tr_s, y_train_sub)
            row_val[f"{name}_{i}"] = model_predict_probs(X_val_s, clf)
            row_te[f"{name}_{i}"] = model_predict_probs(X_te_s, clf)

        val_probs_list.append(pd.DataFrame(row_val))
        test_probs_list.append(pd.DataFrame(row_te))

    # 4. Meta-Learner Preparation
    val_prob_df = pd.concat(val_probs_list, axis=1)
    test_prob_df = pd.concat(test_probs_list, axis=1)

    # Normalization for DeepDILI Meta-model (StandardScaler)
    meta_scaler = StandardScaler()
    val_meta_s = meta_scaler.fit_transform(val_prob_df)
    test_meta_s = meta_scaler.transform(test_prob_df)

    # 5. Final DeepDILI Meta-Model (DNN)
    # Building the architecture defined in your build_meta_model or loading 'best_model.h5'
    meta_model = Sequential([
        Dense(20, activation='relu', input_shape=(val_meta_s.shape[1],)),
        Dropout(0.2),
        Dense(10, activation='relu'),
        Dropout(0.1),
        Dense(1, activation='sigmoid')
    ])
    meta_model.compile(optimizer=Adam(learning_rate=1e-3), loss='binary_crossentropy')

    # In replication, we fit. In production, we'd use: load_model('best_model.h5')
    meta_model.fit(val_meta_s, y_val, epochs=50, verbose=0)

    # 6. Prediction & Evaluation
    final_probs = meta_model.predict(test_meta_s).flatten()
    final_class = (final_probs > 0.5).astype(int)

    return final_probs, final_class

def calculate_metrics(y_true, y_prob, y_class):
    tn, fp, fn, tp = confusion_matrix(y_true, y_class).ravel()
    return {
        "AUPRC": average_precision_score(y_true, y_prob),
        "MCC": matthews_corrcoef(y_true, y_class),
        "Sensitivity": tp / (tp + fn),
        "Specificity": tn / (tn + fp)
    }
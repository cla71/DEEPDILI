"""
Containerized DeepDILI (Ting Li et al.) base model using Mold2 descriptors.

This module mirrors the workflow from `Full_DeepDILI/full_deep_dili_model.ipynb`:
- Load the 1002-compound training set with Mold2 descriptors.
- Train the base classifiers across the FDA train_index splits.
- Select the curated subset of base classifiers.
- Standardize the selected probability features.
- Run the pre-trained DeepDILI meta-model (`best_model.h5`).

The base model expects an SDF file that already contains Mold2 descriptor
properties (descriptor generation is handled separately).

Usage:
    python base_model.py path/to/mold2_descriptors.sdf --output predictions.csv
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd

from rdkit import Chem

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from keras.models import load_model


DATA_DIR = Path(__file__).resolve().parent / "Full_DeepDILI"
TRAINING_CSV = DATA_DIR / "mold2_1002_full.csv"
TRAIN_INDEX_CSV = DATA_DIR / "train_index.csv"
SELECTED_MODELS_CSV = DATA_DIR / "selected_full_model_mcc.csv"
META_MODEL_PATH = DATA_DIR / "best_model.h5"


@dataclass
class DeepDILIArtifacts:
    training_csv: Path = TRAINING_CSV
    train_index_csv: Path = TRAIN_INDEX_CSV
    selected_models_csv: Path = SELECTED_MODELS_CSV
    meta_model_path: Path = META_MODEL_PATH


def _descriptor_columns(training_df: pd.DataFrame) -> List[str]:
    return training_df.columns[3:].tolist()


def _read_mold2_sdf(sdf_path: Path, descriptor_columns: Iterable[str]) -> pd.DataFrame:
    """Load Mold2 descriptors from an SDF file into a DataFrame."""
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    rows = []
    missing = set(descriptor_columns)

    for idx, mol in enumerate(supplier):
        if mol is None:
            continue
        props = mol.GetPropsAsDict()
        row = {"id": idx}
        name = None
        for key in ("CompoundName", "Name", "_Name"):
            if mol.HasProp(key):
                name = mol.GetProp(key)
                break
        if name:
            row["CompoundName"] = name

        for col in descriptor_columns:
            if col in props:
                row[col] = props[col]
            elif mol.HasProp(col):
                row[col] = mol.GetProp(col)
            else:
                missing.add(col)
        rows.append(row)

    if missing:
        sample = ", ".join(sorted(list(missing))[:10])
        raise ValueError(
            "SDF is missing Mold2 descriptor properties required by the model. "
            f"Example missing columns: {sample}"
        )

    df = pd.DataFrame(rows)
    for col in descriptor_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df[descriptor_columns].isna().any().any():
        nan_cols = df[descriptor_columns].columns[df[descriptor_columns].isna().any()].tolist()
        sample = ", ".join(nan_cols[:10])
        raise ValueError(
            "Some Mold2 descriptor values could not be parsed as numbers. "
            f"Example columns with NaNs: {sample}"
        )
    return df


def _model_predict_probs(X: np.ndarray, model) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def _prediction_frames(X_val: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pred_val = pd.DataFrame({"id": X_val.index})
    pred_test = pd.DataFrame({"id": X_test.index})
    return pred_val, pred_test


def _combine_predictions(
    X_val_s: np.ndarray,
    X_test_s: np.ndarray,
    X_val_index: pd.Index,
    X_test_index: pd.Index,
    pred_val: pd.DataFrame,
    pred_test: pd.DataFrame,
    clf,
    col_name: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    val_prob = pd.DataFrame({"id": X_val_index, col_name: _model_predict_probs(X_val_s, clf)})
    pred_val = pd.merge(pred_val, val_prob, on="id", how="left")

    test_prob = pd.DataFrame({"id": X_test_index, col_name: _model_predict_probs(X_test_s, clf)})
    pred_test = pd.merge(pred_test, test_prob, on="id", how="left")
    return pred_val, pred_test


def _select_base_classifiers(
    knns: pd.DataFrame,
    lrs: pd.DataFrame,
    svms: pd.DataFrame,
    rfs: pd.DataFrame,
    xgboosts: pd.DataFrame,
    selected_models: pd.DataFrame,
) -> pd.DataFrame:
    knns = knns[["id", *selected_models[selected_models.model == "knn"].seed.unique()]]
    knns.columns = ["id", *[f"knn_{col}" for col in knns.columns[1:]]]

    lrs = lrs[["id", *selected_models[selected_models.model == "lr"].seed.unique()]]
    lrs.columns = ["id", *[f"lr_{col}" for col in lrs.columns[1:]]]

    svms = svms[["id", *selected_models[selected_models.model == "svm"].seed.unique()]]
    svms.columns = ["id", *[f"svm_{col}" for col in svms.columns[1:]]]

    rfs = rfs[["id", *selected_models[selected_models.model == "rf"].seed.unique()]]
    rfs.columns = ["id", *[f"rf_{col}" for col in rfs.columns[1:]]]

    xgboosts = xgboosts[["id", *selected_models[selected_models.model == "xgboost"].seed.unique()]]
    xgboosts.columns = ["id", *[f"xgboost_{col}" for col in xgboosts.columns[1:]]]

    frames = [knns, lrs, svms, rfs, xgboosts]
    return frames[0].merge(frames[1], on="id").merge(frames[2], on="id").merge(frames[3], on="id").merge(frames[4], on="id")


def _base_classifiers(
    X: pd.DataFrame,
    y: pd.Series,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    train_index_df: pd.DataFrame,
    selected_models: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pred_val_knn, pred_test_knn = _prediction_frames(X_val, X_test)
    pred_val_lr, pred_test_lr = _prediction_frames(X_val, X_test)
    pred_val_svm, pred_test_svm = _prediction_frames(X_val, X_test)
    pred_val_rf, pred_test_rf = _prediction_frames(X_val, X_test)
    pred_val_xgb, pred_test_xgb = _prediction_frames(X_val, X_test)

    for col_name in train_index_df.columns[5:]:
        train_index = train_index_df.loc[train_index_df[col_name] == 1, "id"].unique()

        X_train = X[X.index.isin(train_index)]
        y_train = y[y.index.isin(train_index)]

        scaler = MinMaxScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        X_test_s = scaler.transform(X_test)

        knn = KNeighborsClassifier(n_neighbors=7)
        knn.fit(X_train_s, y_train)
        pred_val_knn, pred_test_knn = _combine_predictions(
            X_val_s, X_test_s, X_val.index, X_test.index, pred_val_knn, pred_test_knn, knn, col_name
        )

        lr = LogisticRegression(C=0.1, max_iter=300, class_weight="balanced")
        lr.fit(X_train_s, y_train)
        pred_val_lr, pred_test_lr = _combine_predictions(
            X_val_s, X_test_s, X_val.index, X_test.index, pred_val_lr, pred_test_lr, lr, col_name
        )

        svm = SVC(kernel="rbf", C=1, gamma="scale", probability=True, class_weight="balanced", random_state=1)
        svm.fit(X_train_s, y_train)
        pred_val_svm, pred_test_svm = _combine_predictions(
            X_val_s, X_test_s, X_val.index, X_test.index, pred_val_svm, pred_test_svm, svm, col_name
        )

        rf = RandomForestClassifier(
            random_state=1,
            n_estimators=700,
            max_depth=11,
            min_samples_leaf=5,
            class_weight="balanced",
            bootstrap=True,
            max_features="log2",
        )
        rf.fit(X_train_s, y_train)
        pred_val_rf, pred_test_rf = _combine_predictions(
            X_val_s, X_test_s, X_val.index, X_test.index, pred_val_rf, pred_test_rf, rf, col_name
        )

        xgboost = XGBClassifier(
            learning_rate=0.01,
            n_estimators=700,
            max_depth=11,
            subsample=0.7,
            scale_pos_weight=0.66,
            eval_metric="logloss",
        )
        xgboost.fit(X_train_s, y_train)
        pred_val_xgb, pred_test_xgb = _combine_predictions(
            X_val_s, X_test_s, X_val.index, X_test.index, pred_val_xgb, pred_test_xgb, xgboost, col_name
        )

    val_prob = _select_base_classifiers(
        pred_val_knn, pred_val_lr, pred_val_svm, pred_val_rf, pred_val_xgb, selected_models
    )
    test_prob = _select_base_classifiers(
        pred_test_knn, pred_test_lr, pred_test_svm, pred_test_rf, pred_test_xgb, selected_models
    )
    return val_prob, test_prob


def run_deepdili(sdf_path: str, artifacts: DeepDILIArtifacts | None = None) -> pd.DataFrame:
    """Run the DeepDILI base model on an SDF with Mold2 descriptors."""
    artifacts = artifacts or DeepDILIArtifacts()

    training_df = pd.read_csv(artifacts.training_csv, low_memory=False)
    descriptor_cols = _descriptor_columns(training_df)

    X_org = training_df[descriptor_cols]
    y_org = training_df["DILI_label"]

    X, X_val, y, y_val = train_test_split(
        X_org, y_org, test_size=0.2, stratify=y_org, random_state=7
    )

    sdf_df = _read_mold2_sdf(Path(sdf_path), descriptor_cols)
    X_test = sdf_df[descriptor_cols]

    train_index_df = pd.read_csv(artifacts.train_index_csv)
    selected_models = pd.read_csv(artifacts.selected_models_csv)

    val_prob, test_prob = _base_classifiers(X, y, X_val, X_test, train_index_df, selected_models)

    scaler = StandardScaler()
    scaler.fit(val_prob.iloc[:, 1:])
    test_prob_s = scaler.transform(test_prob.iloc[:, 1:])

    deepdili = load_model(artifacts.meta_model_path)
    y_pred = deepdili.predict(test_prob_s).flatten()
    y_class = (y_pred > 0.5).astype(int)

    results = pd.DataFrame(
        {
            "id": test_prob["id"].values,
            "prob_DeepDILI": y_pred,
            "class_DeepDILI": y_class,
        }
    )

    if "CompoundName" in sdf_df.columns:
        results = results.merge(sdf_df[["id", "CompoundName"]], on="id", how="left")

    return results


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the DeepDILI base model on an SDF with Mold2 descriptors."
    )
    parser.add_argument("sdf", help="Path to Mold2 SDF with descriptor properties.")
    parser.add_argument(
        "--output",
        "-o",
        help="Optional path to write CSV predictions.",
        default=None,
    )
    args = parser.parse_args()

    results = run_deepdili(args.sdf)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(args.output, index=False)
    else:
        print(results.to_csv(index=False))


if __name__ == "__main__":
    main()

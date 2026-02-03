"""
Compact DeepDILI base model pipeline using Mold2 descriptors.

This module provides an end-to-end workflow for:
1. CSV to SDF conversion from compound data with SMILES
2. Training on DILIrank 2.0 or similar DILI datasets
3. Running the DeepDILI base classifiers and meta-learner

Based on the workflow from `Full_DeepDILI/full_deep_dili_model.ipynb`.

Usage:
    # Generate SDF from CSV
    python base_model.py csv-to-sdf input.csv output.sdf --smiles-col SMILES --id-col Compound

    # Run predictions on SDF with Mold2 descriptors
    python base_model.py predict path/to/mold2_descriptors.sdf --output predictions.csv

    # Train on DILIrank 2.0 dataset
    python base_model.py train --dilirank path/to/dilirank.xlsx --output-dir models/
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=FutureWarning)
np.random.seed(1)

# Lazy import keras to avoid TensorFlow warnings at module level
_keras_model = None


def _load_keras():
    global _keras_model
    if _keras_model is None:
        from keras.models import load_model
        _keras_model = load_model
    return _keras_model


# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = Path(__file__).resolve().parent / "Full_DeepDILI"
DILIRANK_FILE = DATA_DIR / "Drug Induced Liver Injury Rank (DILIrank 2.0) Dataset  FDA (3).xlsx"
TRAINING_CSV = DATA_DIR / "mold2_1002_full.csv"
TRAIN_INDEX_CSV = DATA_DIR / "train_index.csv"
SELECTED_MODELS_CSV = DATA_DIR / "selected_full_model_mcc.csv"
META_MODEL_PATH = DATA_DIR / "best_model.h5"


@dataclass
class DeepDILIConfig:
    """Configuration for DeepDILI pipeline."""
    training_csv: Path = TRAINING_CSV
    train_index_csv: Path = TRAIN_INDEX_CSV
    selected_models_csv: Path = SELECTED_MODELS_CSV
    meta_model_path: Path = META_MODEL_PATH
    dilirank_file: Path = DILIRANK_FILE
    test_size: float = 0.2
    random_state: int = 7
    n_splits: int = 5
    n_repeats: int = 20


# =============================================================================
# CSV to SDF Conversion
# =============================================================================

def csv_to_sdf(
    input_csv: str,
    output_sdf: str,
    smiles_col: str = "SMILES",
    id_col: str = "Compound",
    dili_col: Optional[str] = "DILI",
) -> int:
    """
    Generate a valid Mold2-compatible SDF file from a CSV with SMILES.

    Args:
        input_csv: Path to input CSV file
        output_sdf: Path to output SDF file
        smiles_col: Name of the SMILES column
        id_col: Name of the compound ID column
        dili_col: Name of the DILI label column (optional)

    Returns:
        Number of molecules successfully written
    """
    df = pd.read_csv(input_csv)
    writer = Chem.SDWriter(output_sdf)
    count = 0

    for idx, row in df.iterrows():
        smiles = str(row[smiles_col])
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Warning: Could not parse SMILES at row {idx}: {smiles[:50]}...")
            continue

        # Standardize for descriptors
        mol = Chem.AddHs(mol)
        AllChem.Compute2DCoords(mol)
        try:
            Chem.Kekulize(mol)
        except Exception:
            pass  # Some molecules cannot be kekulized

        # Set properties
        compound_id = str(row[id_col]) if id_col in df.columns else str(idx)
        mol.SetProp("_Name", compound_id)
        mol.SetProp("Name", compound_id)
        mol.SetProp("CompoundID", compound_id)

        if dili_col and dili_col in df.columns:
            mol.SetProp("DILI_label", str(int(row[dili_col])))

        writer.write(mol)
        count += 1

    writer.close()
    print(f"Successfully wrote {count} molecules to {output_sdf}")
    return count


def _clean_sdf_format(sdf_path: str) -> None:
    """Post-process SDF to ensure Mold2 compatibility (V2000 on line 3)."""
    with open(sdf_path, "r") as f:
        lines = f.readlines()

    cleaned_lines = []
    i = 0
    while i < len(lines):
        if i + 3 < len(lines) and "V2000" in lines[i + 3]:
            cleaned_lines.append(lines[i])      # Title
            cleaned_lines.append(lines[i + 1])  # Software info
            cleaned_lines.append(lines[i + 3])  # V2000 line
            i += 4
            while i < len(lines) and "$$$$" not in lines[i]:
                cleaned_lines.append(lines[i])
                i += 1
            if i < len(lines):
                cleaned_lines.append(lines[i])
        else:
            cleaned_lines.append(lines[i])
        i += 1

    with open(sdf_path, "w", newline="\n") as f:
        f.writelines(cleaned_lines)


# =============================================================================
# DILIrank 2.0 Data Loading
# =============================================================================

DILI_CONCERN_MAPPING = {
    "vMOST-DILI-concern": 1,
    "vMost-DILI-concern": 1,
    "vLess-DILI-concern": 1,
    "Ambiguous-DILI-concern": None,  # Exclude ambiguous
    "vNo-DILI-concern": 0,
    "vNo-DILI-Concern": 0,
}


def load_dilirank_dataset(
    dilirank_path: str,
    smiles_csv: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load DILIrank 2.0 dataset and prepare for training.

    Args:
        dilirank_path: Path to DILIrank 2.0 Excel file
        smiles_csv: Optional CSV with CompoundName and SMILES columns for mapping

    Returns:
        DataFrame with CompoundName, DILI_label, and optionally SMILES
    """
    df = pd.read_excel(dilirank_path, header=1)

    # Map DILI concern to binary label
    df["DILI_label"] = df["vDILI-Concern"].map(DILI_CONCERN_MAPPING)

    # Remove ambiguous compounds
    df = df[df["DILI_label"].notna()].copy()
    df["DILI_label"] = df["DILI_label"].astype(int)

    result = df[["CompoundName", "DILI_label", "SeverityClass"]].copy()

    # Merge with SMILES if provided
    if smiles_csv:
        smiles_df = pd.read_csv(smiles_csv)
        # Try common column name variations
        name_col = None
        for col in ["CompoundName", "Name", "Compound", "name", "compound_name"]:
            if col in smiles_df.columns:
                name_col = col
                break

        if name_col and "SMILES" in smiles_df.columns:
            smiles_df = smiles_df[[name_col, "SMILES"]].rename(columns={name_col: "CompoundName"})
            result = result.merge(smiles_df, on="CompoundName", how="left")

    return result


# =============================================================================
# Core Model Functions
# =============================================================================

def _get_descriptor_columns(training_df: pd.DataFrame) -> List[str]:
    """Extract Mold2 descriptor column names from training data."""
    # Columns after CompoundName, DILI_label, final_year are descriptors
    return [c for c in training_df.columns if c.startswith("D") and c[1:].isdigit()]


def _read_mold2_from_csv(csv_path: str, descriptor_cols: List[str]) -> pd.DataFrame:
    """Load Mold2 descriptors from CSV."""
    df = pd.read_csv(csv_path, low_memory=False)
    return df


def _read_mold2_sdf(sdf_path: str, descriptor_cols: List[str]) -> pd.DataFrame:
    """Load Mold2 descriptors from SDF file."""
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    rows = []

    for idx, mol in enumerate(supplier):
        if mol is None:
            continue
        props = mol.GetPropsAsDict()
        row = {"id": idx}

        # Get compound name
        for key in ("CompoundName", "Name", "_Name"):
            if mol.HasProp(key):
                row["CompoundName"] = mol.GetProp(key)
                break

        # Get descriptors
        for col in descriptor_cols:
            if col in props:
                row[col] = props[col]
            elif mol.HasProp(col):
                row[col] = mol.GetProp(col)

        rows.append(row)

    df = pd.DataFrame(rows)
    for col in descriptor_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def _train_base_classifier(clf, X_train: np.ndarray, y_train: np.ndarray):
    """Fit a single base classifier."""
    clf.fit(X_train, y_train)
    return clf


def _get_predictions(clf, X: np.ndarray) -> np.ndarray:
    """Get probability predictions from a classifier."""
    return clf.predict_proba(X)[:, 1]


def train_base_classifiers(
    X: pd.DataFrame,
    y: pd.Series,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    train_index_df: pd.DataFrame,
    selected_models: pd.DataFrame,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Train all base classifiers and return validation and test predictions.

    Args:
        X: Training features
        y: Training labels
        X_val: Validation features
        X_test: Test features
        train_index_df: DataFrame with train/val split indices
        selected_models: DataFrame with selected model indices
        verbose: Print progress

    Returns:
        Tuple of (validation predictions, test predictions) DataFrames
    """
    # Initialize prediction containers
    preds = {
        model: {"val": pd.DataFrame({"id": X_val.index}), "test": pd.DataFrame({"id": X_test.index})}
        for model in ["knn", "lr", "svm", "rf", "xgboost"]
    }

    split_cols = [c for c in train_index_df.columns if "skf" in c]

    for i, col_name in enumerate(split_cols):
        if verbose and i % 10 == 0:
            print(f"Training split {i + 1}/{len(split_cols)}...")

        train_idx = train_index_df.loc[train_index_df[col_name] == 1, "id"].unique()
        X_train = X[X.index.isin(train_idx)]
        y_train = y[y.index.isin(train_idx)]

        scaler = MinMaxScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        X_test_s = scaler.transform(X_test)

        # KNN
        knn = KNeighborsClassifier(n_neighbors=7)
        knn.fit(X_train_s, y_train)
        preds["knn"]["val"][col_name] = _get_predictions(knn, X_val_s)
        preds["knn"]["test"][col_name] = _get_predictions(knn, X_test_s)

        # Logistic Regression
        lr = LogisticRegression(C=0.1, max_iter=300, class_weight="balanced")
        lr.fit(X_train_s, y_train)
        preds["lr"]["val"][col_name] = _get_predictions(lr, X_val_s)
        preds["lr"]["test"][col_name] = _get_predictions(lr, X_test_s)

        # SVM
        svm = SVC(kernel="rbf", C=1, gamma="scale", probability=True, class_weight="balanced", random_state=1)
        svm.fit(X_train_s, y_train)
        preds["svm"]["val"][col_name] = _get_predictions(svm, X_val_s)
        preds["svm"]["test"][col_name] = _get_predictions(svm, X_test_s)

        # Random Forest
        rf = RandomForestClassifier(
            random_state=1, n_estimators=700, max_depth=11,
            min_samples_leaf=5, class_weight="balanced",
            bootstrap=True, max_features="log2"
        )
        rf.fit(X_train_s, y_train)
        preds["rf"]["val"][col_name] = _get_predictions(rf, X_val_s)
        preds["rf"]["test"][col_name] = _get_predictions(rf, X_test_s)

        # XGBoost
        xgb = XGBClassifier(
            learning_rate=0.01, n_estimators=700, max_depth=11,
            subsample=0.7, scale_pos_weight=0.66, eval_metric="logloss"
        )
        xgb.fit(X_train_s, y_train)
        preds["xgboost"]["val"][col_name] = _get_predictions(xgb, X_val_s)
        preds["xgboost"]["test"][col_name] = _get_predictions(xgb, X_test_s)

    # Select and combine base classifier outputs
    val_prob = _select_and_combine(preds, selected_models, "val")
    test_prob = _select_and_combine(preds, selected_models, "test")

    return val_prob, test_prob


def _select_and_combine(
    preds: dict,
    selected_models: pd.DataFrame,
    split: str,
) -> pd.DataFrame:
    """Select and combine predictions from base classifiers."""
    dfs = []
    for model_name in ["knn", "lr", "svm", "rf", "xgboost"]:
        model_df = preds[model_name][split].copy()
        selected = selected_models[selected_models.model == model_name].seed.unique()
        cols = ["id"] + [c for c in model_df.columns if c in selected]
        model_df = model_df[cols]
        model_df.columns = ["id"] + [f"{model_name}_{c}" for c in cols[1:]]
        dfs.append(model_df)

    result = dfs[0]
    for df in dfs[1:]:
        result = result.merge(df, on="id", how="left")

    return result


def run_meta_learner(
    val_prob: pd.DataFrame,
    test_prob: pd.DataFrame,
    meta_model_path: str,
) -> pd.DataFrame:
    """Run the DeepDILI meta-learner on base classifier outputs."""
    load_model = _load_keras()

    scaler = StandardScaler()
    scaler.fit(val_prob.iloc[:, 1:])
    test_prob_s = scaler.transform(test_prob.iloc[:, 1:])

    model = load_model(meta_model_path)
    y_pred = model.predict(test_prob_s).flatten()
    y_class = (y_pred > 0.5).astype(int)

    return pd.DataFrame({
        "id": test_prob["id"].values,
        "prob_DeepDILI": y_pred,
        "class_DeepDILI": y_class,
    })


# =============================================================================
# Main Pipeline Functions
# =============================================================================

def run_prediction(
    test_data_path: str,
    config: Optional[DeepDILIConfig] = None,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Run DeepDILI prediction on test data.

    Args:
        test_data_path: Path to SDF or CSV with Mold2 descriptors
        config: Pipeline configuration
        output_path: Optional path to save predictions

    Returns:
        DataFrame with predictions
    """
    config = config or DeepDILIConfig()

    # Load training data
    training_df = pd.read_csv(config.training_csv, low_memory=False)
    descriptor_cols = _get_descriptor_columns(training_df)

    X_org = training_df[descriptor_cols]
    y_org = training_df["DILI_label"]

    X, X_val, y, y_val = train_test_split(
        X_org, y_org, test_size=config.test_size,
        stratify=y_org, random_state=config.random_state
    )

    # Load test data
    if test_data_path.endswith(".sdf"):
        test_df = _read_mold2_sdf(test_data_path, descriptor_cols)
    else:
        test_df = pd.read_csv(test_data_path, low_memory=False)

    # Ensure test data has required columns
    available_cols = [c for c in descriptor_cols if c in test_df.columns]
    if len(available_cols) < len(descriptor_cols):
        missing = set(descriptor_cols) - set(available_cols)
        raise ValueError(f"Test data missing {len(missing)} descriptor columns. Example: {list(missing)[:5]}")

    X_test = test_df[descriptor_cols]

    # Load model artifacts
    train_index_df = pd.read_csv(config.train_index_csv)
    selected_models = pd.read_csv(config.selected_models_csv)

    print("Training base classifiers...")
    val_prob, test_prob = train_base_classifiers(
        X, y, X_val, X_test, train_index_df, selected_models
    )

    print("Running meta-learner...")
    predictions = run_meta_learner(val_prob, test_prob, str(config.meta_model_path))

    # Merge with compound names if available
    if "CompoundName" in test_df.columns:
        predictions = predictions.merge(
            test_df[["CompoundName"]].reset_index().rename(columns={"index": "id"}),
            on="id", how="left"
        )

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")

    return predictions


def generate_train_index(
    n_samples: int,
    n_splits: int = 5,
    n_repeats: int = 20,
    random_state: int = 1,
) -> pd.DataFrame:
    """Generate train/validation split indices for cross-validation."""
    indices = pd.DataFrame({"id": range(n_samples)})

    for repeat in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state + repeat)
        # Use dummy labels for splitting
        y_dummy = np.zeros(n_samples)
        for fold, (train_idx, val_idx) in enumerate(skf.split(range(n_samples), y_dummy)):
            col_name = f"{repeat}_skf_{fold}"
            indices[col_name] = 0
            indices.loc[train_idx, col_name] = 1

    return indices


# =============================================================================
# Command Line Interface
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="DeepDILI base model pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Convert CSV with SMILES to SDF
    python base_model.py csv-to-sdf compounds.csv output.sdf --smiles-col SMILES --id-col Name

    # Run predictions on SDF with Mold2 descriptors
    python base_model.py predict mold2_descriptors.sdf -o predictions.csv

    # Load DILIrank 2.0 dataset
    python base_model.py load-dilirank "DILIrank 2.0.xlsx" -o dilirank_processed.csv
        """
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # CSV to SDF command
    csv_parser = subparsers.add_parser("csv-to-sdf", help="Convert CSV with SMILES to SDF")
    csv_parser.add_argument("input_csv", help="Input CSV file")
    csv_parser.add_argument("output_sdf", help="Output SDF file")
    csv_parser.add_argument("--smiles-col", default="SMILES", help="SMILES column name")
    csv_parser.add_argument("--id-col", default="Compound", help="ID column name")
    csv_parser.add_argument("--dili-col", default="DILI", help="DILI label column name")

    # Prediction command
    pred_parser = subparsers.add_parser("predict", help="Run DeepDILI predictions")
    pred_parser.add_argument("test_data", help="Path to SDF or CSV with Mold2 descriptors")
    pred_parser.add_argument("-o", "--output", help="Output CSV file for predictions")

    # Load DILIrank command
    dili_parser = subparsers.add_parser("load-dilirank", help="Load and process DILIrank 2.0 dataset")
    dili_parser.add_argument("dilirank_file", help="Path to DILIrank 2.0 Excel file")
    dili_parser.add_argument("--smiles-csv", help="CSV with compound names and SMILES")
    dili_parser.add_argument("-o", "--output", help="Output CSV file")

    args = parser.parse_args()

    if args.command == "csv-to-sdf":
        csv_to_sdf(
            args.input_csv,
            args.output_sdf,
            smiles_col=args.smiles_col,
            id_col=args.id_col,
            dili_col=args.dili_col,
        )
        _clean_sdf_format(args.output_sdf)

    elif args.command == "predict":
        results = run_prediction(args.test_data, output_path=args.output)
        if not args.output:
            print(results.to_csv(index=False))

    elif args.command == "load-dilirank":
        df = load_dilirank_dataset(args.dilirank_file, args.smiles_csv)
        if args.output:
            df.to_csv(args.output, index=False)
            print(f"Processed DILIrank data saved to {args.output}")
            print(f"Total compounds: {len(df)}")
            print(f"DILI positive: {df['DILI_label'].sum()}")
            print(f"DILI negative: {(df['DILI_label'] == 0).sum()}")
        else:
            print(df.to_csv(index=False))


if __name__ == "__main__":
    main()

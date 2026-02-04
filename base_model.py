#!/usr/bin/env python3
"""
DeepDILI and DILIrank Model Pipeline

This module provides a comprehensive drug-induced liver injury (DILI) prediction
framework comparing two approaches:

1. **Original DeepDILI** (Ting Li et al.): Uses Mold2 descriptors with pre-trained
   base classifiers and meta-learner from `Full_DeepDILI/full_deep_dili_model.ipynb`

2. **DILIrank Model**: Uses RDKit molecular descriptors and fingerprints, trains
   new base classifiers and meta-learner on the DILIrank 2.0 dataset

"""

from __future__ import annotations

import json
import os
import pickle
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# RDKit imports
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, MACCSkeys
from rdkit.ML.Descriptors import MoleculeDescriptors

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
np.random.seed(42)

# =============================================================================
# Lazy Imports
# =============================================================================

_keras_loaded = False
_keras_modules = {}


def _load_keras():
    """Lazy load Keras/TensorFlow to avoid startup warnings."""
    global _keras_loaded, _keras_modules
    if not _keras_loaded:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        from keras.models import Sequential, load_model
        from keras.layers import Dense, Dropout, BatchNormalization
        from keras.callbacks import EarlyStopping
        from keras.optimizers import Adam
        _keras_modules = {
            'Sequential': Sequential,
            'load_model': load_model,
            'Dense': Dense,
            'Dropout': Dropout,
            'BatchNormalization': BatchNormalization,
            'EarlyStopping': EarlyStopping,
            'Adam': Adam,
        }
        _keras_loaded = True
    return _keras_modules


# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = Path(__file__).resolve().parent / "Full_DeepDILI"
MODELS_DIR = Path(__file__).resolve().parent / "models"

# Original DeepDILI artifacts
TRAINING_CSV = DATA_DIR / "mold2_1002_full.csv"
TRAIN_INDEX_CSV = DATA_DIR / "train_index.csv"
SELECTED_MODELS_CSV = DATA_DIR / "selected_full_model_mcc.csv"
META_MODEL_PATH = DATA_DIR / "best_model.h5"
DILIRANK_FILE = DATA_DIR / "Drug Induced Liver Injury Rank (DILIrank 2.0) Dataset  FDA (3).xlsx"

# DILIrank model artifacts (to be created)
DILIRANK_MODEL_DIR = MODELS_DIR / "dilirank"


@dataclass
class DeepDILIConfig:
    """Configuration for original DeepDILI pipeline."""
    training_csv: Path = TRAINING_CSV
    train_index_csv: Path = TRAIN_INDEX_CSV
    selected_models_csv: Path = SELECTED_MODELS_CSV
    meta_model_path: Path = META_MODEL_PATH
    test_size: float = 0.2
    random_state: int = 7


@dataclass
class DILIrankConfig:
    """Configuration for DILIrank pipeline."""
    dilirank_file: Path = DILIRANK_FILE
    model_dir: Path = DILIRANK_MODEL_DIR
    n_splits: int = 5
    n_repeats: int = 10
    test_size: float = 0.2
    random_state: int = 42
    descriptor_types: List[str] = field(default_factory=lambda: ["rdkit", "morgan", "maccs"])


# DILIrank concern to binary label mapping
DILI_CONCERN_MAPPING = {
    "vMOST-DILI-concern": 1,
    "vMost-DILI-concern": 1,
    "vLess-DILI-concern": 1,
    "Ambiguous-DILI-concern": None,  # Exclude
    "vNo-DILI-concern": 0,
    "vNo-DILI-Concern": 0,
}


# =============================================================================
# Molecular Descriptor Generation
# =============================================================================

# RDKit 2D descriptor names (subset of most informative)
RDKIT_DESCRIPTOR_NAMES = [
    'MolWt', 'MolLogP', 'MolMR', 'TPSA', 'LabuteASA',
    'NumHDonors', 'NumHAcceptors', 'NumRotatableBonds',
    'NumHeteroatoms', 'NumValenceElectrons', 'NumAromaticRings',
    'NumSaturatedRings', 'NumAliphaticRings', 'RingCount',
    'FractionCSP3', 'HeavyAtomCount', 'NumRadicalElectrons',
    'MaxPartialCharge', 'MinPartialCharge', 'MaxAbsPartialCharge',
    'MinAbsPartialCharge', 'BalabanJ', 'BertzCT', 'Chi0', 'Chi0n',
    'Chi0v', 'Chi1', 'Chi1n', 'Chi1v', 'Chi2n', 'Chi2v', 'Chi3n',
    'Chi3v', 'Chi4n', 'Chi4v', 'HallKierAlpha', 'Ipc', 'Kappa1',
    'Kappa2', 'Kappa3', 'EState_VSA1', 'EState_VSA2', 'EState_VSA3',
    'EState_VSA4', 'EState_VSA5', 'EState_VSA6', 'EState_VSA7',
    'EState_VSA8', 'EState_VSA9', 'EState_VSA10', 'VSA_EState1',
    'VSA_EState2', 'VSA_EState3', 'VSA_EState4', 'VSA_EState5',
    'VSA_EState6', 'VSA_EState7', 'VSA_EState8', 'VSA_EState9',
    'VSA_EState10', 'PEOE_VSA1', 'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA4',
    'PEOE_VSA5', 'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 'PEOE_VSA9',
    'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12', 'PEOE_VSA13', 'PEOE_VSA14',
    'SMR_VSA1', 'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5',
    'SMR_VSA6', 'SMR_VSA7', 'SMR_VSA8', 'SMR_VSA9', 'SMR_VSA10',
    'SlogP_VSA1', 'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5',
    'SlogP_VSA6', 'SlogP_VSA7', 'SlogP_VSA8', 'SlogP_VSA9', 'SlogP_VSA10',
    'SlogP_VSA11', 'SlogP_VSA12', 'fr_Al_COO', 'fr_Al_OH', 'fr_Al_OH_noTert',
    'fr_ArN', 'fr_Ar_COO', 'fr_Ar_N', 'fr_Ar_NH', 'fr_Ar_OH', 'fr_COO',
    'fr_COO2', 'fr_C_O', 'fr_C_O_noCOO', 'fr_C_S', 'fr_HOCCN', 'fr_Imine',
    'fr_NH0', 'fr_NH1', 'fr_NH2', 'fr_N_O', 'fr_Ndealkylation1',
    'fr_Ndealkylation2', 'fr_Nhpyrrole', 'fr_SH', 'fr_aldehyde',
    'fr_alkyl_carbamate', 'fr_alkyl_halide', 'fr_allylic_oxid',
    'fr_amide', 'fr_amidine', 'fr_aniline', 'fr_aryl_methyl',
    'fr_azide', 'fr_azo', 'fr_barbitur', 'fr_benzene', 'fr_benzodiazepine',
    'fr_bicyclic', 'fr_diazo', 'fr_dihydropyridine', 'fr_epoxide',
    'fr_ester', 'fr_ether', 'fr_furan', 'fr_guanido', 'fr_halogen',
    'fr_hdrzine', 'fr_hdrzone', 'fr_imidazole', 'fr_imide', 'fr_isocyan',
    'fr_isothiocyan', 'fr_ketone', 'fr_ketone_Topliss', 'fr_lactam',
    'fr_lactone', 'fr_methoxy', 'fr_morpholine', 'fr_nitrile',
    'fr_nitro', 'fr_nitro_arom', 'fr_nitro_arom_nonortho', 'fr_nitroso',
    'fr_oxazole', 'fr_oxime', 'fr_para_hydroxylation', 'fr_phenol',
    'fr_phenol_noOrthoHbond', 'fr_phos_acid', 'fr_phos_ester',
    'fr_piperdine', 'fr_piperzine', 'fr_priamide', 'fr_prisulfonamd',
    'fr_pyridine', 'fr_quatN', 'fr_sulfide', 'fr_sulfonamd', 'fr_sulfone',
    'fr_term_acetylene', 'fr_tetrazole', 'fr_thiazole', 'fr_thiocyan',
    'fr_thiophene', 'fr_unbrch_alkane', 'fr_urea', 'qed'
]


def smiles_to_mol(smiles: str) -> Optional[Chem.Mol]:
    """Convert SMILES string to RDKit Mol object."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            mol = Chem.AddHs(mol)
        return mol
    except Exception:
        return None


def calculate_rdkit_descriptors(mol: Chem.Mol) -> Dict[str, float]:
    """Calculate RDKit 2D molecular descriptors."""
    if mol is None:
        return {}

    # Get available descriptors
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(RDKIT_DESCRIPTOR_NAMES)
    try:
        mol_no_h = Chem.RemoveHs(mol)
        desc_values = calc.CalcDescriptors(mol_no_h)
        return dict(zip(RDKIT_DESCRIPTOR_NAMES, desc_values))
    except Exception:
        return {}


def calculate_morgan_fingerprint(
    mol: Chem.Mol,
    radius: int = 2,
    n_bits: int = 1024
) -> np.ndarray:
    """Calculate Morgan (ECFP) fingerprint."""
    if mol is None:
        return np.zeros(n_bits)

    try:
        mol_no_h = Chem.RemoveHs(mol)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol_no_h, radius, nBits=n_bits)
        arr = np.zeros(n_bits, dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    except Exception:
        return np.zeros(n_bits)


def calculate_maccs_fingerprint(mol: Chem.Mol) -> np.ndarray:
    """Calculate MACCS keys fingerprint (166 bits)."""
    if mol is None:
        return np.zeros(167)

    try:
        mol_no_h = Chem.RemoveHs(mol)
        fp = MACCSkeys.GenMACCSKeys(mol_no_h)
        arr = np.zeros(167, dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    except Exception:
        return np.zeros(167)


def generate_descriptors(
    smiles_list: List[str],
    compound_names: Optional[List[str]] = None,
    descriptor_types: List[str] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Generate molecular descriptors for a list of SMILES.

    Args:
        smiles_list: List of SMILES strings
        compound_names: Optional list of compound identifiers
        descriptor_types: Types of descriptors to generate
            - "rdkit": RDKit 2D descriptors (200 features)
            - "morgan": Morgan/ECFP4 fingerprint (1024 bits)
            - "maccs": MACCS keys (167 bits)
        verbose: Print progress

    Returns:
        DataFrame with descriptors for each compound
    """
    if descriptor_types is None:
        descriptor_types = ["rdkit", "morgan", "maccs"]

    if compound_names is None:
        compound_names = [f"Compound_{i}" for i in range(len(smiles_list))]

    results = []
    n_total = len(smiles_list)

    for i, (smiles, name) in enumerate(zip(smiles_list, compound_names)):
        if verbose and (i + 1) % 100 == 0:
            print(f"Processing {i + 1}/{n_total}...")

        mol = smiles_to_mol(smiles)
        row = {"CompoundName": name, "SMILES": smiles, "valid_mol": mol is not None}

        if "rdkit" in descriptor_types:
            rdkit_desc = calculate_rdkit_descriptors(mol)
            for k, v in rdkit_desc.items():
                row[f"RDKit_{k}"] = v

        if "morgan" in descriptor_types:
            morgan_fp = calculate_morgan_fingerprint(mol)
            for j, bit in enumerate(morgan_fp):
                row[f"Morgan_{j}"] = bit

        if "maccs" in descriptor_types:
            maccs_fp = calculate_maccs_fingerprint(mol)
            for j, bit in enumerate(maccs_fp):
                row[f"MACCS_{j}"] = bit

        results.append(row)

    df = pd.DataFrame(results)

    # Handle NaN/Inf values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df[numeric_cols] = df[numeric_cols].fillna(0)

    if verbose:
        valid_count = df['valid_mol'].sum()
        print(f"Generated descriptors for {valid_count}/{n_total} valid molecules")
        if "rdkit" in descriptor_types:
            rdkit_cols = [c for c in df.columns if c.startswith("RDKit_")]
            print(f"  RDKit descriptors: {len(rdkit_cols)}")
        if "morgan" in descriptor_types:
            morgan_cols = [c for c in df.columns if c.startswith("Morgan_")]
            print(f"  Morgan fingerprint bits: {len(morgan_cols)}")
        if "maccs" in descriptor_types:
            maccs_cols = [c for c in df.columns if c.startswith("MACCS_")]
            print(f"  MACCS keys: {len(maccs_cols)}")

    return df


# =============================================================================
# CSV to SDF Conversion
# =============================================================================

def csv_to_sdf(
    input_csv: str,
    output_sdf: str,
    smiles_col: str = "SMILES",
    id_col: str = "CompoundName",
    label_col: Optional[str] = None,
) -> int:
    """
    Convert CSV with SMILES to Mold2-compatible SDF file.

    Args:
        input_csv: Input CSV file path
        output_sdf: Output SDF file path
        smiles_col: Column name containing SMILES
        id_col: Column name for compound identifiers
        label_col: Optional column with labels to include

    Returns:
        Number of successfully written molecules
    """
    df = pd.read_csv(input_csv)
    writer = Chem.SDWriter(output_sdf)
    count = 0

    for idx, row in df.iterrows():
        smiles = str(row[smiles_col])
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Warning: Invalid SMILES at row {idx}: {smiles[:50]}...")
            continue

        mol = Chem.AddHs(mol)
        AllChem.Compute2DCoords(mol)

        # Set properties
        name = str(row[id_col]) if id_col in df.columns else f"Compound_{idx}"
        mol.SetProp("_Name", name)
        mol.SetProp("CompoundName", name)

        if label_col and label_col in df.columns:
            mol.SetProp("DILI_label", str(int(row[label_col])))

        writer.write(mol)
        count += 1

    writer.close()
    print(f"Wrote {count} molecules to {output_sdf}")
    return count


# =============================================================================
# DILIrank Dataset Loading
# =============================================================================

def load_dilirank_dataset(
    dilirank_path: str,
    smiles_csv: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load and process DILIrank 2.0 dataset.

    Args:
        dilirank_path: Path to DILIrank Excel file
        smiles_csv: Optional CSV with CompoundName and SMILES mapping

    Returns:
        DataFrame with CompoundName, DILI_label, and optionally SMILES
    """
    df = pd.read_excel(dilirank_path, header=1)

    # Map DILI concern to binary
    df["DILI_label"] = df["vDILI-Concern"].map(DILI_CONCERN_MAPPING)
    df = df[df["DILI_label"].notna()].copy()
    df["DILI_label"] = df["DILI_label"].astype(int)

    result = df[["CompoundName", "DILI_label"]].copy()
    if "SeverityClass" in df.columns:
        result["SeverityClass"] = df["SeverityClass"]

    # Merge SMILES if provided
    if smiles_csv:
        smiles_df = pd.read_csv(smiles_csv)
        for col in ["CompoundName", "Name", "Compound", "name", "compound_name"]:
            if col in smiles_df.columns:
                smiles_df = smiles_df.rename(columns={col: "CompoundName"})
                break

        if "SMILES" in smiles_df.columns:
            smiles_df = smiles_df[["CompoundName", "SMILES"]].drop_duplicates()
            result = result.merge(smiles_df, on="CompoundName", how="left")

    return result


# =============================================================================
# Model Definitions
# =============================================================================

def get_base_classifiers() -> Dict[str, Any]:
    """Get dictionary of base classifier configurations."""
    return {
        "knn": KNeighborsClassifier(n_neighbors=7),
        "lr": LogisticRegression(C=0.1, max_iter=500, class_weight="balanced", random_state=42),
        "svm": SVC(kernel="rbf", C=1, gamma="scale", probability=True,
                   class_weight="balanced", random_state=42),
        "rf": RandomForestClassifier(n_estimators=500, max_depth=10, min_samples_leaf=5,
                                     class_weight="balanced", random_state=42, n_jobs=-1),
        "xgboost": XGBClassifier(learning_rate=0.01, n_estimators=500, max_depth=8,
                                  subsample=0.8, scale_pos_weight=1.5,
                                  eval_metric="logloss", random_state=42),
        "gb": GradientBoostingClassifier(n_estimators=300, max_depth=6,
                                          learning_rate=0.05, random_state=42),
        "mlp": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500,
                             early_stopping=True, random_state=42),
    }


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Calculate classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5,
    }


# =============================================================================
# Original DeepDILI Pipeline
# =============================================================================

def _get_mold2_descriptor_cols(df: pd.DataFrame) -> List[str]:
    """Get Mold2 descriptor column names (D001-D777 format)."""
    return [c for c in df.columns if c.startswith("D") and c[1:].replace("0", "").isdigit()]


def run_deepdili_prediction(
    test_data_path: str,
    config: Optional[DeepDILIConfig] = None,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Run original DeepDILI prediction pipeline.

    Args:
        test_data_path: Path to CSV with Mold2 descriptors
        config: DeepDILI configuration
        output_path: Optional output path for predictions

    Returns:
        DataFrame with predictions
    """
    config = config or DeepDILIConfig()
    keras = _load_keras()

    # Load training data
    training_df = pd.read_csv(config.training_csv, low_memory=False)
    descriptor_cols = _get_mold2_descriptor_cols(training_df)

    X_org = training_df[descriptor_cols]
    y_org = training_df["DILI_label"]

    X_train, X_val, y_train, y_val = train_test_split(
        X_org, y_org, test_size=config.test_size,
        stratify=y_org, random_state=config.random_state
    )

    # Load test data
    test_df = pd.read_csv(test_data_path, low_memory=False)
    missing_cols = set(descriptor_cols) - set(test_df.columns)
    if missing_cols:
        raise ValueError(f"Test data missing {len(missing_cols)} Mold2 descriptor columns")

    X_test = test_df[descriptor_cols]

    # Load artifacts
    train_index_df = pd.read_csv(config.train_index_csv)
    selected_models = pd.read_csv(config.selected_models_csv)

    print("Training DeepDILI base classifiers...")
    val_prob, test_prob = _train_deepdili_base_classifiers(
        X_train, y_train, X_val, X_test, train_index_df, selected_models
    )

    print("Running DeepDILI meta-learner...")
    scaler = StandardScaler()
    scaler.fit(val_prob.iloc[:, 1:])
    test_prob_scaled = scaler.transform(test_prob.iloc[:, 1:])

    meta_model = keras['load_model'](str(config.meta_model_path))
    y_pred_prob = meta_model.predict(test_prob_scaled, verbose=0).flatten()
    y_pred_class = (y_pred_prob > 0.5).astype(int)

    predictions = pd.DataFrame({
        "id": range(len(test_df)),
        "prob_DeepDILI": y_pred_prob,
        "class_DeepDILI": y_pred_class,
    })

    if "CompoundName" in test_df.columns:
        predictions["CompoundName"] = test_df["CompoundName"].values

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(output_path, index=False)
        print(f"DeepDILI predictions saved to {output_path}")

    return predictions


def _train_deepdili_base_classifiers(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    train_index_df: pd.DataFrame,
    selected_models: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Train DeepDILI base classifiers and return predictions."""

    models = ["knn", "lr", "svm", "rf", "xgboost"]
    preds = {m: {"val": pd.DataFrame({"id": X_val.index}),
                 "test": pd.DataFrame({"id": X_test.index})} for m in models}

    split_cols = [c for c in train_index_df.columns if "skf" in c]

    for i, col in enumerate(split_cols):
        if i % 20 == 0:
            print(f"  Training split {i + 1}/{len(split_cols)}...")

        train_idx = train_index_df.loc[train_index_df[col] == 1, "id"].unique()
        X_tr = X_train[X_train.index.isin(train_idx)]
        y_tr = y_train[y_train.index.isin(train_idx)]

        scaler = MinMaxScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)
        X_test_s = scaler.transform(X_test)

        for model_name, clf in [
            ("knn", KNeighborsClassifier(n_neighbors=7)),
            ("lr", LogisticRegression(C=0.1, max_iter=300, class_weight="balanced")),
            ("svm", SVC(kernel="rbf", C=1, gamma="scale", probability=True,
                       class_weight="balanced", random_state=1)),
            ("rf", RandomForestClassifier(n_estimators=700, max_depth=11, min_samples_leaf=5,
                                          class_weight="balanced", random_state=1)),
            ("xgboost", XGBClassifier(learning_rate=0.01, n_estimators=700, max_depth=11,
                                       subsample=0.7, scale_pos_weight=0.66,
                                       eval_metric="logloss")),
        ]:
            clf.fit(X_tr_s, y_tr)
            preds[model_name]["val"][col] = clf.predict_proba(X_val_s)[:, 1]
            preds[model_name]["test"][col] = clf.predict_proba(X_test_s)[:, 1]

    # Select and combine
    val_prob = _select_base_predictions(preds, selected_models, "val")
    test_prob = _select_base_predictions(preds, selected_models, "test")

    val_prob = _select_base_classifiers(
        pred_val_knn, pred_val_lr, pred_val_svm, pred_val_rf, pred_val_xgb, selected_models
    )
    test_prob = _select_base_classifiers(
        pred_test_knn, pred_test_lr, pred_test_svm, pred_test_rf, pred_test_xgb, selected_models
    )
    return val_prob, test_prob


def _select_base_predictions(preds: dict, selected: pd.DataFrame, split: str) -> pd.DataFrame:
    """Select and combine base classifier predictions."""
    dfs = []
    for model in ["knn", "lr", "svm", "rf", "xgboost"]:
        df = preds[model][split].copy()
        selected_seeds = selected[selected.model == model].seed.unique()
        cols = ["id"] + [c for c in df.columns if c in selected_seeds]
        df = df[cols]
        df.columns = ["id"] + [f"{model}_{c}" for c in cols[1:]]
        dfs.append(df)

    result = dfs[0]
    for df in dfs[1:]:
        result = result.merge(df, on="id", how="left")
    return result


# =============================================================================
# DILIrank Model Pipeline
# =============================================================================

class DILIrankModel:
    """DILIrank-based DILI prediction model using RDKit descriptors."""

    def __init__(self, config: Optional[DILIrankConfig] = None):
        self.config = config or DILIrankConfig()
        self.base_classifiers: Dict[str, List[Any]] = {}
        self.selected_indices: Dict[str, List[int]] = {}
        self.scalers: List[Any] = []
        self.meta_model = None
        self.feature_cols: List[str] = []
        self.is_trained = False

    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get feature columns based on descriptor types."""
        cols = []
        for desc_type in self.config.descriptor_types:
            if desc_type == "rdkit":
                cols.extend([c for c in df.columns if c.startswith("RDKit_")])
            elif desc_type == "morgan":
                cols.extend([c for c in df.columns if c.startswith("Morgan_")])
            elif desc_type == "maccs":
                cols.extend([c for c in df.columns if c.startswith("MACCS_")])
        return cols

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Train DILIrank model with multiple base classifiers and meta-learner.

        Args:
            X: Features DataFrame
            y: Labels Series
            verbose: Print progress

        Returns:
            Dictionary with training results and metrics
        """
        keras = _load_keras()

        self.feature_cols = self._get_feature_columns(X)
        X_features = X[self.feature_cols].values
        y_values = y.values

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_features, y_values, test_size=self.config.test_size,
            stratify=y_values, random_state=self.config.random_state
        )

        if verbose:
            print(f"Training set: {len(X_train)}, Test set: {len(X_test)}")
            print(f"Features: {len(self.feature_cols)}")
            print(f"Class distribution - Train: {np.bincount(y_train)}, Test: {np.bincount(y_test)}")

        # Train base classifiers with cross-validation
        base_models = get_base_classifiers()
        all_val_preds = {name: [] for name in base_models}
        all_val_indices = []
        all_test_preds = {name: [] for name in base_models}
        base_metrics = {name: [] for name in base_models}

        skf = StratifiedKFold(n_splits=self.config.n_splits, shuffle=True,
                              random_state=self.config.random_state)

        for repeat in range(self.config.n_repeats):
            if verbose and repeat % 2 == 0:
                print(f"Training repeat {repeat + 1}/{self.config.n_repeats}...")

            for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]

                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_tr)
                X_val_s = scaler.transform(X_val)
                X_test_s = scaler.transform(X_test)

                self.scalers.append(scaler)
                all_val_indices.append(val_idx)

                for name, clf in base_models.items():
                    # Clone classifier
                    clf_clone = clf.__class__(**clf.get_params())
                    clf_clone.fit(X_tr_s, y_tr)

                    val_prob = clf_clone.predict_proba(X_val_s)[:, 1]
                    test_prob = clf_clone.predict_proba(X_test_s)[:, 1]

                    all_val_preds[name].append((val_idx, val_prob))
                    all_test_preds[name].append(test_prob)

                    # Calculate MCC for model selection
                    val_pred = (val_prob > 0.5).astype(int)
                    mcc = matthews_corrcoef(y_val, val_pred)
                    base_metrics[name].append(mcc)

                    if name not in self.base_classifiers:
                        self.base_classifiers[name] = []
                    self.base_classifiers[name].append(clf_clone)

        # Select top models based on MCC
        for name in base_models:
            mccs = base_metrics[name]
            n_select = max(1, int(len(mccs) * 0.7))  # Top 70%
            top_indices = np.argsort(mccs)[-n_select:]
            self.selected_indices[name] = top_indices.tolist()

        # Create meta-learner training data
        n_train = len(X_train)
        meta_train_preds = np.zeros((n_train, len(base_models)))
        meta_train_counts = np.zeros(n_train)

        for model_idx, name in enumerate(base_models):
            for fold_idx in self.selected_indices[name]:
                val_idx, val_prob = all_val_preds[name][fold_idx]
                meta_train_preds[val_idx, model_idx] += val_prob
                meta_train_counts[val_idx] += 1

        # Average predictions
        meta_train_counts[meta_train_counts == 0] = 1
        meta_train_preds /= meta_train_counts[:, np.newaxis]

        # Create test predictions
        meta_test_preds = np.zeros((len(X_test), len(base_models)))
        for model_idx, name in enumerate(base_models):
            selected_preds = [all_test_preds[name][i] for i in self.selected_indices[name]]
            meta_test_preds[:, model_idx] = np.mean(selected_preds, axis=0)

        # Train meta-learner
        if verbose:
            print("Training meta-learner...")

        meta_scaler = StandardScaler()
        meta_train_s = meta_scaler.fit_transform(meta_train_preds)
        meta_test_s = meta_scaler.transform(meta_test_preds)

        self.meta_model = self._build_meta_learner(
            meta_train_s, y_train, meta_test_s, y_test, verbose
        )
        self.meta_scaler = meta_scaler
        self.is_trained = True

        # Final evaluation
        y_test_prob = self.meta_model.predict(meta_test_s, verbose=0).flatten()
        y_test_pred = (y_test_prob > 0.5).astype(int)

        final_metrics = calculate_metrics(y_test, y_test_pred, y_test_prob)

        # Base classifier metrics
        base_results = {}
        for name in base_models:
            selected_preds = [all_test_preds[name][i] for i in self.selected_indices[name]]
            avg_pred = np.mean(selected_preds, axis=0)
            avg_class = (avg_pred > 0.5).astype(int)
            base_results[name] = calculate_metrics(y_test, avg_class, avg_pred)

        if verbose:
            print("\n" + "=" * 60)
            print("DILIrank Model Training Results")
            print("=" * 60)
            print("\nBase Classifier Performance (Test Set):")
            for name, metrics in base_results.items():
                print(f"  {name:10s}: MCC={metrics['mcc']:.3f}, AUC={metrics['auc']:.3f}, "
                      f"F1={metrics['f1']:.3f}")
            print(f"\nMeta-Learner Performance (Test Set):")
            print(f"  MCC={final_metrics['mcc']:.3f}, AUC={final_metrics['auc']:.3f}, "
                  f"Accuracy={final_metrics['accuracy']:.3f}")
            print(f"  Precision={final_metrics['precision']:.3f}, "
                  f"Recall={final_metrics['recall']:.3f}, F1={final_metrics['f1']:.3f}")

        return {
            "base_metrics": base_results,
            "meta_metrics": final_metrics,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "n_features": len(self.feature_cols),
        }

    def _build_meta_learner(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        verbose: bool
    ):
        """Build and train the meta-learner neural network."""
        keras = _load_keras()

        model = keras['Sequential']([
            keras['Dense'](64, activation='relu', input_shape=(X_train.shape[1],)),
            keras['BatchNormalization'](),
            keras['Dropout'](0.3),
            keras['Dense'](32, activation='relu'),
            keras['BatchNormalization'](),
            keras['Dropout'](0.2),
            keras['Dense'](16, activation='relu'),
            keras['Dense'](1, activation='sigmoid'),
        ])

        model.compile(
            optimizer=keras['Adam'](learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        early_stop = keras['EarlyStopping'](
            monitor='val_loss', patience=20, restore_best_weights=True
        )

        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=200,
            batch_size=32,
            callbacks=[early_stop],
            verbose=1 if verbose else 0
        )

        return model

    def predict(
        self,
        X: pd.DataFrame,
        return_base_predictions: bool = False
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Make predictions on new data.

        Args:
            X: Features DataFrame (must have same descriptor columns)
            return_base_predictions: Also return individual base classifier predictions

        Returns:
            DataFrame with predictions, optionally base predictions
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")

        # Get features
        missing = set(self.feature_cols) - set(X.columns)
        if missing:
            raise ValueError(f"Missing {len(missing)} feature columns: {list(missing)[:5]}...")

        X_features = X[self.feature_cols].values

        # Get base predictions
        base_models = list(self.base_classifiers.keys())
        base_preds = np.zeros((len(X_features), len(base_models)))

        for model_idx, name in enumerate(base_models):
            model_preds = []
            for clf_idx in self.selected_indices[name]:
                clf = self.base_classifiers[name][clf_idx]
                scaler = self.scalers[clf_idx]
                X_scaled = scaler.transform(X_features)
                model_preds.append(clf.predict_proba(X_scaled)[:, 1])
            base_preds[:, model_idx] = np.mean(model_preds, axis=0)

        # Meta-learner prediction
        meta_scaled = self.meta_scaler.transform(base_preds)
        y_prob = self.meta_model.predict(meta_scaled, verbose=0).flatten()
        y_class = (y_prob > 0.5).astype(int)

        predictions = pd.DataFrame({
            "prob_DILIrank": y_prob,
            "class_DILIrank": y_class,
        })

        if "CompoundName" in X.columns:
            predictions.insert(0, "CompoundName", X["CompoundName"].values)

        if return_base_predictions:
            base_df = pd.DataFrame(base_preds, columns=[f"prob_{m}" for m in base_models])
            return predictions, base_df

        return predictions

    def save(self, model_dir: str):
        """Save trained model to directory."""
        if not self.is_trained:
            raise RuntimeError("Model not trained.")

        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save meta-learner
        self.meta_model.save(model_dir / "meta_learner.h5")

        # Save other components
        with open(model_dir / "model_state.pkl", "wb") as f:
            pickle.dump({
                "base_classifiers": self.base_classifiers,
                "selected_indices": self.selected_indices,
                "scalers": self.scalers,
                "meta_scaler": self.meta_scaler,
                "feature_cols": self.feature_cols,
                "config": self.config,
            }, f)

        print(f"Model saved to {model_dir}")

    @classmethod
    def load(cls, model_dir: str) -> "DILIrankModel":
        """Load trained model from directory."""
        keras = _load_keras()
        model_dir = Path(model_dir)

        instance = cls()

        with open(model_dir / "model_state.pkl", "rb") as f:
            state = pickle.load(f)

        instance.base_classifiers = state["base_classifiers"]
        instance.selected_indices = state["selected_indices"]
        instance.scalers = state["scalers"]
        instance.meta_scaler = state["meta_scaler"]
        instance.feature_cols = state["feature_cols"]
        instance.config = state["config"]
        instance.meta_model = keras['load_model'](model_dir / "meta_learner.h5")
        instance.is_trained = True

        return instance


def train_dilirank_model(
    dilirank_path: str,
    smiles_csv: str,
    output_dir: str,
    config: Optional[DILIrankConfig] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Train a new DILIrank model.

    Args:
        dilirank_path: Path to DILIrank Excel file
        smiles_csv: Path to CSV with compound SMILES
        output_dir: Directory to save trained model
        config: Model configuration
        verbose: Print progress

    Returns:
        Training results dictionary
    """
    config = config or DILIrankConfig()

    # Load DILIrank dataset
    if verbose:
        print("Loading DILIrank dataset...")
    dilirank_df = load_dilirank_dataset(dilirank_path, smiles_csv)

    # Filter to compounds with SMILES
    if "SMILES" not in dilirank_df.columns:
        raise ValueError("No SMILES column found. Provide smiles_csv with compound SMILES.")

    dilirank_df = dilirank_df[dilirank_df["SMILES"].notna()].copy()

    if verbose:
        print(f"Loaded {len(dilirank_df)} compounds with SMILES")
        print(f"  DILI positive: {dilirank_df['DILI_label'].sum()}")
        print(f"  DILI negative: {(dilirank_df['DILI_label'] == 0).sum()}")

    # Generate descriptors
    if verbose:
        print("\nGenerating molecular descriptors...")

    desc_df = generate_descriptors(
        dilirank_df["SMILES"].tolist(),
        dilirank_df["CompoundName"].tolist(),
        descriptor_types=config.descriptor_types,
        verbose=verbose
    )

    # Merge with labels
    desc_df = desc_df.merge(
        dilirank_df[["CompoundName", "DILI_label"]],
        on="CompoundName", how="inner"
    )

    # Filter valid molecules
    desc_df = desc_df[desc_df["valid_mol"]].copy()

    if verbose:
        print(f"\nTraining on {len(desc_df)} valid compounds")

    # Train model
    model = DILIrankModel(config)
    results = model.train(desc_df, desc_df["DILI_label"], verbose=verbose)

    # Save model
    model.save(output_dir)

    # Save training info
    with open(Path(output_dir) / "training_info.json", "w") as f:
        json.dump({
            "n_compounds": len(desc_df),
            "n_positive": int(desc_df["DILI_label"].sum()),
            "n_negative": int((desc_df["DILI_label"] == 0).sum()),
            "descriptor_types": config.descriptor_types,
            "metrics": {k: float(v) for k, v in results["meta_metrics"].items()},
        }, f, indent=2)

    return results


def run_dilirank_prediction(
    test_data_path: str,
    model_dir: Optional[str] = None,
    smiles_col: Optional[str] = None,
    id_col: str = "CompoundName",
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Run DILIrank model predictions.

    Args:
        test_data_path: Path to CSV (with SMILES or descriptors)
        model_dir: Path to trained model directory
        smiles_col: SMILES column name (if generating descriptors)
        id_col: Compound ID column name
        output_path: Optional output path

    Returns:
        DataFrame with predictions
    """
    model_dir = model_dir or str(DILIRANK_MODEL_DIR)

    if not Path(model_dir).exists():
        raise FileNotFoundError(
            f"DILIrank model not found at {model_dir}. "
            "Train a model first with 'train-dilirank' command."
        )

    # Load model
    print("Loading DILIrank model...")
    model = DILIrankModel.load(model_dir)

    # Load test data
    test_df = pd.read_csv(test_data_path)

    # Check if we need to generate descriptors
    feature_cols = model.feature_cols
    has_features = all(c in test_df.columns for c in feature_cols[:10])

    if not has_features:
        if smiles_col is None:
            # Try to find SMILES column
            for col in ["SMILES", "smiles", "Smiles", "canonical_smiles"]:
                if col in test_df.columns:
                    smiles_col = col
                    break

        if smiles_col is None or smiles_col not in test_df.columns:
            raise ValueError(
                "Test data missing descriptor columns and no SMILES column found. "
                "Use --smiles-col to specify the SMILES column."
            )

        print("Generating molecular descriptors...")
        compound_names = test_df[id_col].tolist() if id_col in test_df.columns else None
        test_df = generate_descriptors(
            test_df[smiles_col].tolist(),
            compound_names,
            descriptor_types=model.config.descriptor_types,
        )

    # Make predictions
    print("Running predictions...")
    predictions = model.predict(test_df)

    if output_path:
        predictions.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")

    return predictions

    scaler = StandardScaler()
    scaler.fit(val_prob.iloc[:, 1:])
    test_prob_s = scaler.transform(test_prob.iloc[:, 1:])

# =============================================================================
# Model Comparison
# =============================================================================

def compare_models(
    test_csv: str,
    smiles_col: str = "SMILES",
    id_col: str = "CompoundName",
    label_col: Optional[str] = None,
    mold2_csv: Optional[str] = None,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compare DeepDILI and DILIrank predictions on test compounds.

    Args:
        test_csv: CSV with test compounds and SMILES
        smiles_col: SMILES column name
        id_col: Compound ID column name
        label_col: Optional true label column for evaluation
        mold2_csv: Optional CSV with pre-computed Mold2 descriptors
        output_path: Output path for comparison results

    Returns:
        DataFrame with predictions from both models
    """
    test_df = pd.read_csv(test_csv)

    results = pd.DataFrame()
    if id_col in test_df.columns:
        results["CompoundName"] = test_df[id_col]
    else:
        results["CompoundName"] = [f"Compound_{i}" for i in range(len(test_df))]

    if label_col and label_col in test_df.columns:
        results["true_label"] = test_df[label_col]

    # DILIrank predictions
    print("\n" + "=" * 60)
    print("Running DILIrank Model")
    print("=" * 60)

    try:
        dilirank_preds = run_dilirank_prediction(
            test_csv, smiles_col=smiles_col, id_col=id_col
        )
        results["prob_DILIrank"] = dilirank_preds["prob_DILIrank"]
        results["class_DILIrank"] = dilirank_preds["class_DILIrank"]
    except Exception as e:
        print(f"DILIrank prediction failed: {e}")
        results["prob_DILIrank"] = np.nan
        results["class_DILIrank"] = np.nan

    # DeepDILI predictions (requires Mold2 descriptors)
    print("\n" + "=" * 60)
    print("Running DeepDILI Model")
    print("=" * 60)

    if mold2_csv and Path(mold2_csv).exists():
        try:
            deepdili_preds = run_deepdili_prediction(mold2_csv)
            results["prob_DeepDILI"] = deepdili_preds["prob_DeepDILI"]
            results["class_DeepDILI"] = deepdili_preds["class_DeepDILI"]
        except Exception as e:
            print(f"DeepDILI prediction failed: {e}")
            results["prob_DeepDILI"] = np.nan
            results["class_DeepDILI"] = np.nan
    else:
        print("Mold2 descriptors not provided. Skipping DeepDILI.")
        print("To include DeepDILI predictions:")
        print("  1. Convert CSV to SDF: python base_model.py csv-to-sdf ...")
        print("  2. Generate Mold2 descriptors with external Mold2 tool")
        print("  3. Run comparison with --mold2-csv option")
        results["prob_DeepDILI"] = np.nan
        results["class_DeepDILI"] = np.nan

    # Calculate metrics if labels available
    if label_col and label_col in test_df.columns:
        y_true = test_df[label_col].values

        print("\n" + "=" * 60)
        print("Model Comparison Metrics")
        print("=" * 60)

        for model_name in ["DILIrank", "DeepDILI"]:
            prob_col = f"prob_{model_name}"
            class_col = f"class_{model_name}"

            if prob_col in results.columns and not results[prob_col].isna().all():
                y_prob = results[prob_col].values
                y_pred = results[class_col].values

                metrics = calculate_metrics(y_true, y_pred, y_prob)

                print(f"\n{model_name}:")
                print(f"  Accuracy:     {metrics['accuracy']:.3f}")
                print(f"  Balanced Acc: {metrics['balanced_accuracy']:.3f}")
                print(f"  Precision:    {metrics['precision']:.3f}")
                print(f"  Recall:       {metrics['recall']:.3f}")
                print(f"  F1 Score:     {metrics['f1']:.3f}")
                print(f"  MCC:          {metrics['mcc']:.3f}")
                print(f"  AUC-ROC:      {metrics['auc']:.3f}")

    # Agreement analysis
    if (not results["class_DILIrank"].isna().all() and
        not results["class_DeepDILI"].isna().all()):

        valid_mask = ~(results["class_DILIrank"].isna() | results["class_DeepDILI"].isna())
        if valid_mask.sum() > 0:
            agreement = (results.loc[valid_mask, "class_DILIrank"] ==
                        results.loc[valid_mask, "class_DeepDILI"]).mean()

            print(f"\nModel Agreement: {agreement:.1%}")

            # Correlation
            corr = results.loc[valid_mask, ["prob_DILIrank", "prob_DeepDILI"]].corr().iloc[0, 1]
            print(f"Probability Correlation: {corr:.3f}")

    if output_path:
        results.to_csv(output_path, index=False)
        print(f"\nComparison results saved to {output_path}")

    return results

    return results


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="DeepDILI and DILIrank DILI Prediction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=""


if __name__ == "__main__":
    main()

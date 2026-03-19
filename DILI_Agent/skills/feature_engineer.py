"""
Feature Engineering Skill
=========================
Generate molecular descriptors and fingerprints for DILI prediction.

USER-CONTROLLED: Only the user can modify this skill.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass

# Import skill registry
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from agent.base_agent import skills


# ============================================================================
# DESCRIPTOR CONFIGURATIONS
# ============================================================================

FINGERPRINT_CONFIGS = {
    "morgan": {
        "description": "Morgan/ECFP circular fingerprints",
        "radius": 2,
        "n_bits": 2048
    },
    "maccs": {
        "description": "MACCS structural keys (166 bits)",
        "n_bits": 167
    },
    "rdkit": {
        "description": "RDKit topological fingerprints",
        "n_bits": 2048
    },
    "atompair": {
        "description": "Atom pair fingerprints",
        "n_bits": 2048
    }
}

DESCRIPTOR_SETS = {
    "rdkit_2d": {
        "description": "RDKit 2D molecular descriptors (~200 features)",
        "n_features": 200
    },
    "mordred": {
        "description": "Mordred descriptors (1613+ features)",
        "n_features": 1613
    },
    "physicochemical": {
        "description": "Basic physicochemical properties",
        "features": ["MolWt", "LogP", "TPSA", "HBD", "HBA", "RotatableBonds"]
    }
}


# ============================================================================
# SKILL FUNCTIONS
# ============================================================================

@skills.register("generate_fingerprints", "Generate molecular fingerprints from SMILES")
def generate_fingerprints(
    smiles_list: List[str],
    fp_type: str = "morgan",
    n_bits: int = 2048,
    radius: int = 2
) -> Dict[str, Any]:
    """
    Generate molecular fingerprints from SMILES strings.

    Args:
        smiles_list: List of SMILES strings
        fp_type: Fingerprint type (morgan, maccs, rdkit, atompair)
        n_bits: Number of bits for folded fingerprints
        radius: Radius for Morgan fingerprints

    Returns:
        Dictionary with fingerprint array and metadata
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, MACCSkeys, RDKFingerprint
        from rdkit.Chem.AtomPairs import Pairs
    except ImportError:
        return {
            "status": "error",
            "message": "RDKit not installed. Install with: pip install rdkit"
        }

    fingerprints = []
    valid_indices = []
    errors = []

    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            errors.append({"index": i, "smiles": smi, "error": "Invalid SMILES"})
            continue

        try:
            if fp_type == "morgan":
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            elif fp_type == "maccs":
                fp = MACCSkeys.GenMACCSKeys(mol)
            elif fp_type == "rdkit":
                fp = RDKFingerprint(mol, fpSize=n_bits)
            elif fp_type == "atompair":
                fp = Pairs.GetAtomPairFingerprintAsBitVect(mol)
            else:
                return {"status": "error", "message": f"Unknown fingerprint type: {fp_type}"}

            fingerprints.append(np.array(fp))
            valid_indices.append(i)
        except Exception as e:
            errors.append({"index": i, "smiles": smi, "error": str(e)})

    if not fingerprints:
        return {"status": "error", "message": "No valid fingerprints generated"}

    fp_array = np.array(fingerprints)

    return {
        "status": "success",
        "fingerprints": fp_array,
        "shape": fp_array.shape,
        "fp_type": fp_type,
        "valid_indices": valid_indices,
        "n_errors": len(errors),
        "errors": errors[:10]  # Return first 10 errors
    }


@skills.register("generate_rdkit_descriptors", "Generate RDKit 2D descriptors")
def generate_rdkit_descriptors(
    smiles_list: List[str],
    descriptor_subset: List[str] = None
) -> Dict[str, Any]:
    """
    Generate RDKit 2D molecular descriptors.

    Args:
        smiles_list: List of SMILES strings
        descriptor_subset: Specific descriptors to calculate (None = all)

    Returns:
        Dictionary with descriptor matrix and names
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        from rdkit.ML.Descriptors import MoleculeDescriptors
    except ImportError:
        return {
            "status": "error",
            "message": "RDKit not installed"
        }

    # Get descriptor names
    if descriptor_subset is None:
        descriptor_names = [x[0] for x in Descriptors._descList]
    else:
        descriptor_names = descriptor_subset

    calc = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)

    descriptors = []
    valid_indices = []
    errors = []

    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            errors.append({"index": i, "smiles": smi})
            continue

        try:
            desc = calc.CalcDescriptors(mol)
            descriptors.append(list(desc))
            valid_indices.append(i)
        except Exception as e:
            errors.append({"index": i, "smiles": smi, "error": str(e)})

    if not descriptors:
        return {"status": "error", "message": "No valid descriptors generated"}

    desc_array = np.array(descriptors)

    # Handle NaN/Inf values
    desc_array = np.nan_to_num(desc_array, nan=0.0, posinf=0.0, neginf=0.0)

    return {
        "status": "success",
        "descriptors": desc_array,
        "descriptor_names": descriptor_names,
        "shape": desc_array.shape,
        "valid_indices": valid_indices,
        "n_errors": len(errors)
    }


@skills.register("generate_mordred_descriptors", "Generate Mordred descriptors")
def generate_mordred_descriptors(
    smiles_list: List[str],
    ignore_3d: bool = True
) -> Dict[str, Any]:
    """
    Generate Mordred molecular descriptors.

    Args:
        smiles_list: List of SMILES strings
        ignore_3d: Skip 3D descriptors (faster)

    Returns:
        Dictionary with descriptor matrix
    """
    try:
        from mordred import Calculator, descriptors
        from rdkit import Chem
    except ImportError:
        return {
            "status": "error",
            "message": "Mordred not installed. Install with: pip install mordred"
        }

    calc = Calculator(descriptors, ignore_3D=ignore_3d)

    mols = []
    valid_indices = []

    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            mols.append(mol)
            valid_indices.append(i)

    if not mols:
        return {"status": "error", "message": "No valid molecules"}

    # Calculate descriptors
    df = calc.pandas(mols)

    # Convert to numeric, handling errors
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.fillna(0)

    return {
        "status": "success",
        "descriptors": df.values,
        "descriptor_names": list(df.columns),
        "shape": df.shape,
        "valid_indices": valid_indices
    }


@skills.register("combine_features", "Combine multiple feature sets")
def combine_features(
    feature_sets: List[Dict[str, Any]],
    method: str = "concatenate"
) -> Dict[str, Any]:
    """
    Combine multiple feature sets into a single matrix.

    Args:
        feature_sets: List of feature dictionaries (from other skills)
        method: Combination method ('concatenate', 'pca', 'select')

    Returns:
        Combined feature matrix
    """
    if not feature_sets:
        return {"status": "error", "message": "No feature sets provided"}

    arrays = []
    all_names = []

    for fs in feature_sets:
        if fs.get("status") != "success":
            continue
        arr = fs.get("fingerprints") or fs.get("descriptors")
        if arr is not None:
            arrays.append(arr)
            names = fs.get("descriptor_names") or [f"fp_{i}" for i in range(arr.shape[1])]
            all_names.extend(names)

    if not arrays:
        return {"status": "error", "message": "No valid feature arrays"}

    if method == "concatenate":
        combined = np.hstack(arrays)
    else:
        # Future: PCA, feature selection
        combined = np.hstack(arrays)

    return {
        "status": "success",
        "features": combined,
        "feature_names": all_names,
        "shape": combined.shape,
        "method": method
    }


@skills.register("select_features", "Select important features")
def select_features(
    features: np.ndarray,
    labels: np.ndarray,
    method: str = "mutual_info",
    n_features: int = 100
) -> Dict[str, Any]:
    """
    Select most important features for DILI prediction.

    Args:
        features: Feature matrix
        labels: Target labels
        method: Selection method (mutual_info, variance, random_forest)
        n_features: Number of features to select

    Returns:
        Selected feature indices and scores
    """
    try:
        from sklearn.feature_selection import (
            mutual_info_classif,
            VarianceThreshold,
            SelectKBest
        )
    except ImportError:
        return {"status": "error", "message": "scikit-learn not installed"}

    if method == "mutual_info":
        selector = SelectKBest(mutual_info_classif, k=min(n_features, features.shape[1]))
        selector.fit(features, labels)
        scores = selector.scores_
        selected_indices = np.argsort(scores)[-n_features:]

    elif method == "variance":
        selector = VarianceThreshold()
        selector.fit(features)
        variances = selector.variances_
        selected_indices = np.argsort(variances)[-n_features:]
        scores = variances

    else:
        return {"status": "error", "message": f"Unknown method: {method}"}

    return {
        "status": "success",
        "selected_indices": selected_indices.tolist(),
        "scores": scores[selected_indices].tolist(),
        "n_selected": len(selected_indices),
        "method": method
    }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_available_descriptors() -> Dict[str, Any]:
    """Get list of available descriptor types."""
    return {
        "fingerprints": FINGERPRINT_CONFIGS,
        "descriptors": DESCRIPTOR_SETS
    }

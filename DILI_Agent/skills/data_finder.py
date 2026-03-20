"""
Data Finder Skill
=================
Locate and load DILI-relevant datasets from public and local sources.

USER-CONTROLLED: Only the user can modify this skill.
"""

import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# Import skill registry from base agent
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from agent.base_agent import skills


@dataclass
class DataSource:
    """Represents a data source."""
    name: str
    path: str
    source_type: str  # 'local', 'pubchem', 'chembl', 'url'
    description: str
    columns: List[str]
    n_compounds: int


# ============================================================================
# PUBLIC DATA SOURCES
# ============================================================================

KNOWN_SOURCES = {
    "dilirank": {
        "description": "DILIrank 2.0 FDA dataset - curated DILI classifications",
        "expected_columns": ["Compound Name", "SMILES", "vDILI-Concern"]
    },
    "livertox": {
        "description": "LiverTox database annotations",
        "url": "https://www.ncbi.nlm.nih.gov/books/NBK547852/"
    },
    "toxcast": {
        "description": "ToxCast hepatotoxicity assays",
        "endpoint": "pubchem"
    },
    "hepg2": {
        "description": "HepG2 cytotoxicity screening data",
        "endpoint": "chembl"
    }
}


# ============================================================================
# SKILL FUNCTIONS
# ============================================================================

@skills.register("find_local_data", "Search local directories for DILI datasets")
def find_local_data(
    search_dir: str = ".",
    patterns: List[str] = None
) -> List[Dict[str, Any]]:
    """
    Search local directories for potential DILI datasets.

    Args:
        search_dir: Directory to search
        patterns: File patterns to match (default: csv, xlsx, sdf)

    Returns:
        List of found data sources
    """
    if patterns is None:
        patterns = ["*.csv", "*.xlsx", "*.xls", "*.sdf", "*.smi"]

    search_path = Path(search_dir)
    found = []

    for pattern in patterns:
        for file_path in search_path.rglob(pattern):
            info = {
                "path": str(file_path),
                "name": file_path.name,
                "size_mb": file_path.stat().st_size / (1024 * 1024),
                "type": file_path.suffix
            }

            # Try to get column info for tabular files
            if file_path.suffix in ['.csv', '.xlsx', '.xls']:
                try:
                    if file_path.suffix == '.csv':
                        df = pd.read_csv(file_path, nrows=5)
                    else:
                        df = pd.read_excel(file_path, nrows=5)
                    info["columns"] = list(df.columns)
                    info["n_rows_preview"] = len(df)
                except Exception as e:
                    info["error"] = str(e)

            found.append(info)

    return found


@skills.register("load_dilirank", "Load DILIrank 2.0 dataset")
def load_dilirank(
    file_path: str,
    label_column: str = "vDILI-Concern"
) -> Dict[str, Any]:
    """
    Load and process DILIrank 2.0 dataset.

    Args:
        file_path: Path to DILIrank Excel/CSV file
        label_column: Column containing DILI labels

    Returns:
        Processed dataset info
    """
    path = Path(file_path)

    if path.suffix in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path)
    else:
        df = pd.read_csv(file_path)

    # Process DILI labels
    if label_column in df.columns:
        # Map to binary: Most-DILI-Concern/Less-DILI-Concern -> 1, No-DILI-Concern -> 0
        label_mapping = {
            'Most-DILI-Concern': 1,
            'Less-DILI-Concern': 1,
            'No-DILI-Concern': 0,
            'Ambiguous-DILI-Concern': None  # Exclude ambiguous
        }
        df['DILI_label'] = df[label_column].map(label_mapping)
        df = df.dropna(subset=['DILI_label'])

    return {
        "status": "success",
        "n_compounds": len(df),
        "columns": list(df.columns),
        "label_distribution": df['DILI_label'].value_counts().to_dict() if 'DILI_label' in df.columns else {},
        "smiles_column": _find_smiles_column(df)
    }


@skills.register("fetch_pubchem", "Fetch compound data from PubChem")
def fetch_pubchem(
    compound_ids: List[str] = None,
    assay_id: str = None
) -> Dict[str, Any]:
    """
    Fetch compound data from PubChem.

    Args:
        compound_ids: List of PubChem CIDs
        assay_id: PubChem assay ID (AID)

    Returns:
        Fetched data info
    """
    # STUB: Implement PubChem API calls
    # https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest

    return {
        "status": "stub",
        "message": "PubChem fetch not yet implemented",
        "requires": ["requests", "pubchempy"]
    }


@skills.register("fetch_chembl", "Fetch compound data from ChEMBL")
def fetch_chembl(
    target_id: str = None,
    assay_type: str = "hepatotoxicity"
) -> Dict[str, Any]:
    """
    Fetch compound/assay data from ChEMBL.

    Args:
        target_id: ChEMBL target ID
        assay_type: Type of assay to search

    Returns:
        Fetched data info
    """
    # STUB: Implement ChEMBL API calls
    # https://www.ebi.ac.uk/chembl/api/data/docs

    return {
        "status": "stub",
        "message": "ChEMBL fetch not yet implemented",
        "requires": ["chembl_webresource_client"]
    }


@skills.register("load_data", "Auto-discover and load the best available DILI dataset")
def load_data(
    search_dir: str = None,
    prefer: str = "dilirank"
) -> Dict[str, Any]:
    """
    Alias skill: Auto-discover and load DILI data from standard locations.
    Tries INPUTDATA, data/, and project root.

    Args:
        search_dir: Directory to search (default: auto-detect)
        prefer: Preferred dataset type ('dilirank')

    Returns:
        Loaded dataset info
    """
    # Build search directories
    agent_root = Path(__file__).parent.parent
    project_root = agent_root.parent

    search_dirs = []
    if search_dir:
        search_dirs.append(Path(search_dir))
    search_dirs.extend([
        project_root / "INPUTDATA",
        project_root / "data",
        agent_root / "data",
        project_root,
    ])

    # Search for files
    candidates = []
    for d in search_dirs:
        if d.exists():
            found = find_local_data(str(d))
            candidates.extend(found)

    if not candidates:
        return {
            "status": "not_found",
            "message": "No DILI datasets found in standard locations",
            "searched": [str(d) for d in search_dirs]
        }

    # Prefer DILIrank file
    dilirank_files = [f for f in candidates if 'dilirank' in f['name'].lower() or 'DILIrank' in f['name']]
    chosen = dilirank_files[0] if dilirank_files else candidates[0]

    # Try to load it
    if chosen['type'] in ['.xlsx', '.xls', '.csv']:
        return load_dilirank(chosen['path'])

    return {
        "status": "found",
        "file": chosen['path'],
        "message": f"Found {len(candidates)} dataset(s). Load manually with load_dilirank.",
        "all_found": [f['path'] for f in candidates]
    }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _find_smiles_column(df: pd.DataFrame) -> Optional[str]:
    """Find the SMILES column in a dataframe."""
    smiles_patterns = ['smiles', 'SMILES', 'canonical_smiles', 'smi', 'structure']
    for col in df.columns:
        if any(pattern in col.lower() for pattern in ['smiles', 'smi']):
            return col
    return None


def list_known_sources() -> Dict[str, Any]:
    """List known DILI data sources."""
    return KNOWN_SOURCES

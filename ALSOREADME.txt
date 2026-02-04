Key Features
1. Molecular Descriptor Generation
RDKit 2D descriptors: 200+ physicochemical properties
Morgan fingerprints: ECFP4-style, 1024 bits
MACCS keys: 167-bit structural fingerprint
2. DILIrank Model (DILIrankModel class)
7 Base Classifiers: KNN, Logistic Regression, SVM, Random Forest, XGBoost, Gradient Boosting, MLP
Cross-validation: 5-fold with 10 repeats, model selection via MCC
Meta-learner: Neural network with BatchNorm and Dropout
Full serialization: Save/load trained models
3. Original DeepDILI Pipeline
Preserved workflow from full_deep_dili_model.ipynb
Uses Mold2 descriptors with pre-trained meta-learner
4. Comparison Framework
Side-by-side evaluation with metrics (MCC, AUC, F1, etc.)
Agreement analysis between models

Use Case Examples

Example 1: Test on Original DeepDILI (requires Mold2)
# Step 1: Convert CSV to SDF
python base_model.py csv-to-sdf my_compounds.csv my_compounds.sdf \
    --smiles-col SMILES --id-col CompoundName

# Step 2: Generate Mold2 descriptors (external tool)
# java -jar Mold2.jar -i my_compounds.sdf -o my_compounds_mold2.csv

# Step 3: Run predictions
python base_model.py predict-deepdili my_compounds_mold2.csv -o predictions.csv

Example 2: Test on DILIrank Model (RDKit-based, no external tools)
# Option A: Two-step
python base_model.py generate-descriptors my_compounds.csv \
    --smiles-col SMILES -o descriptors.csv
python base_model.py predict-dilirank descriptors.csv -o predictions.csv

# Option B: One command (auto-generates descriptors)
python base_model.py predict-dilirank my_compounds.csv \
    --smiles-col SMILES -o predictions.csv

Example 3: Train a New DILIrank Model
python base_model.py train-dilirank \
    --dilirank-file "Full_DeepDILI/DILIrank 2.0.xlsx" \
    --smiles-csv compound_smiles.csv \
    --output-dir models/dilirank/

Example 4: Compare Both Models
python base_model.py compare my_compounds.csv \
    --smiles-col SMILES \
    --label-col DILI_label \
    --mold2-csv my_compounds_mold2.csv \
    -o comparison_results.csv

CLI Commands Summary
Command	Description
csv-to-sdf	Convert CSV with SMILES to SDF for Mold2
generate-descriptors	Generate RDKit descriptors from SMILES
predict-deepdili	Run original DeepDILI (requires Mold2)
predict-dilirank	Run DILIrank model (RDKit-based)
train-dilirank	Train new DILIrank model on DILIrank 2.0
compare	Compare both models on test compounds
load-dilirank	Load and process DILIrank 2.0 dataset

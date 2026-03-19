# DILI Optimization Agent

An agentic system for optimizing Machine Learning workflows to predict Drug-Induced Liver Injury (DILI).

## Overview

This agent autonomously explores data, engineers features, trains models, and optimizes hyperparameters to achieve the best DILI prediction performance. It uses a skills-based architecture where:

- **The agent** can ONLY modify the `experiments/dili_optimization.ipynb` notebook
- **The user** controls the skills library and agent base code

## Architecture

```
DILI_Agent/
├── agent/
│   ├── base_agent.py      # Core agent loop (USER-CONTROLLED)
│   ├── config.yaml        # Agent configuration
│   └── state.json         # Persistent state
├── skills/                # Skills library (USER-CONTROLLED)
│   ├── data_finder.py     # Find and load DILI datasets
│   ├── feature_engineer.py # Generate molecular descriptors
│   ├── model_optimizer.py # Train and optimize ML models
│   ├── notebook_writer.py # Modify experiment notebook
│   └── validator.py       # Validate model performance
├── experiments/
│   └── dili_optimization.ipynb  # THE ONLY FILE AGENT MODIFIES
├── models/                # Saved trained models
└── data/                  # Local datasets
```

## Agentic Loop

The agent follows this cycle:

```
OBSERVE → PLAN → EXECUTE → EVALUATE → ITERATE
```

1. **OBSERVE**: Gather current state (data, features, model performance)
2. **PLAN**: Use LLM to decide next action based on observations
3. **EXECUTE**: Run the planned skill (e.g., generate features, train model)
4. **EVALUATE**: Check if target metrics are achieved
5. **ITERATE**: Continue until goal is met or max iterations reached

## Configuration

### Ollama Models

The agent uses Ollama for local LLM inference:

- **Primary model** (`qwen3.5:9b-q4`): Planning and reasoning
- **Secondary model** (`qwen3.5:4b`): Quick code generation tasks

Configure in `agent/config.yaml`:

```yaml
models:
  primary:
    name: "qwen3.5:9b-q4"
    endpoint: "http://localhost:11434"
    temperature: 0.7
  secondary:
    name: "qwen3.5:4b"
    endpoint: "http://localhost:11434"
    temperature: 0.5
```

### Target Metrics

```yaml
agent:
  targets:
    mcc_threshold: 0.6    # Target MCC score
    auc_threshold: 0.85   # Target AUC-ROC
```

## Usage

### Prerequisites

1. Install Ollama and pull models:
```bash
ollama pull qwen3.5:9b-q4
ollama pull qwen3.5:4b
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

### Running the Agent

```bash
# Start the agent
python -m agent.base_agent run

# Run with custom iteration limit
python -m agent.base_agent run -n 20

# Check status
python -m agent.base_agent status

# Reset state
python -m agent.base_agent reset

# List available skills
python -m agent.base_agent skills
```

### Interactive Mode

```python
from agent.base_agent import DILIOptimizationAgent

agent = DILIOptimizationAgent()
agent.run(max_iterations=10)
```

## Skills

### data_finder
Find and load DILI-relevant datasets:
- Local file search
- DILIrank loading
- PubChem/ChEMBL integration (stub)

### feature_engineer
Generate molecular features:
- Morgan fingerprints (ECFP)
- MACCS keys
- RDKit 2D descriptors
- Mordred descriptors
- Feature selection

### model_optimizer
Train and optimize models:
- Random Forest, XGBoost, SVM, etc.
- Cross-validation
- Hyperparameter search
- Ensemble methods

### notebook_writer
Modify the experiment notebook:
- Add code/markdown cells
- Generate standard code blocks
- Track experiment sections

### validator
Validate model performance:
- Prediction metrics (MCC, AUC, etc.)
- Data quality checks
- CV stability analysis
- External validation

## Data Sources

The agent can work with:
- DILIrank 2.0 dataset
- LiverTox annotations
- ToxCast hepatotoxicity assays
- HepG2 cytotoxicity data
- Custom local datasets

## Extending Skills

To add new skills, modify the skills library (user-controlled):

```python
# In skills/my_skill.py
from agent.base_agent import skills

@skills.register("my_skill", "Description of what it does")
def my_skill(param1, param2):
    # Implementation
    return {"status": "success", "result": ...}
```

## Safety

- Agent can ONLY modify `experiments/dili_optimization.ipynb`
- Skills library requires user modification
- Auto-execute disabled by default
- State checkpoints for recovery

## License

Part of the DEEPDILI project.

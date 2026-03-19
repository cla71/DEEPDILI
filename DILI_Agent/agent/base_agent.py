"""
DILI Optimization Agent - Base Agent Module
============================================
Core agentic loop for optimizing DILI prediction ML workflows.

Only the user can modify this file and the skills library.
The agent modifies ONLY the experiments/dili_optimization.ipynb file.

Uses Ollama with qwen3.5:9b-q4 (planner) and qwen3.5:4b (executor).
"""

import json
import yaml
import time
import logging
import requests
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Callable
from enum import Enum
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

AGENT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = AGENT_ROOT / "agent" / "config.yaml"
STATE_PATH = AGENT_ROOT / "agent" / "state.json"


def load_config() -> dict:
    """Load agent configuration from YAML."""
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)


# ============================================================================
# AGENT STATE
# ============================================================================

class Phase(Enum):
    """Agent execution phases."""
    INIT = "init"
    OBSERVE = "observe"
    PLAN = "plan"
    EXECUTE = "execute"
    EVALUATE = "evaluate"
    ITERATE = "iterate"
    COMPLETE = "complete"


@dataclass
class ModelMetrics:
    """Metrics for a trained model."""
    name: str
    mcc: float = 0.0
    auc: float = 0.0
    sensitivity: float = 0.0
    specificity: float = 0.0
    accuracy: float = 0.0
    features: List[str] = field(default_factory=list)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentState:
    """
    Persistent state for the DILI optimization agent.
    Tracks progress across iterations.
    """
    iteration: int = 0
    phase: str = "init"
    goal: str = "Optimize ML workflow for DILI prediction"

    # Best model tracking
    current_best: Dict[str, Any] = field(default_factory=lambda: {
        "model": None, "mcc": 0.0, "auc": 0.0, "features": []
    })

    # History and observations
    history: List[Dict[str, Any]] = field(default_factory=list)
    observations: List[str] = field(default_factory=list)
    pending_actions: List[Dict[str, Any]] = field(default_factory=list)

    # Data tracking
    data_sources: Dict[str, List[str]] = field(default_factory=lambda: {
        "loaded": [], "available": []
    })

    # Feature tracking
    features: Dict[str, List[str]] = field(default_factory=lambda: {
        "engineered": [], "selected": []
    })

    # Model tracking
    models: Dict[str, List[Dict]] = field(default_factory=lambda: {
        "trained": [], "evaluated": []
    })

    errors: List[str] = field(default_factory=list)
    checkpoints: List[str] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path = STATE_PATH) -> 'AgentState':
        """Load state from JSON file."""
        if path.exists():
            with open(path, 'r') as f:
                data = json.load(f)
            return cls(**data)
        return cls()

    def save(self, path: Path = STATE_PATH):
        """Save state to JSON file."""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2, default=str)

    def add_observation(self, obs: str):
        """Add an observation with timestamp."""
        self.observations.append(f"[{datetime.now().isoformat()}] {obs}")

    def add_to_history(self, action: str, result: Any, success: bool):
        """Record an action in history."""
        self.history.append({
            "iteration": self.iteration,
            "phase": self.phase,
            "action": action,
            "result": str(result)[:500],  # Truncate long results
            "success": success,
            "timestamp": datetime.now().isoformat()
        })

    def update_best(self, metrics: ModelMetrics):
        """Update best model if metrics are better."""
        if metrics.mcc > self.current_best.get("mcc", 0):
            self.current_best = {
                "model": metrics.name,
                "mcc": metrics.mcc,
                "auc": metrics.auc,
                "features": metrics.features
            }
            return True
        return False


# ============================================================================
# OLLAMA CLIENT
# ============================================================================

class OllamaClient:
    """Client for Ollama API interaction."""

    def __init__(self, endpoint: str = "http://localhost:11434"):
        self.endpoint = endpoint
        self.logger = logging.getLogger("OllamaClient")

    def generate(
        self,
        model: str,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4096
    ) -> str:
        """Generate text using Ollama API."""
        url = f"{self.endpoint}/api/generate"

        payload = {
            "model": model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }

        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            return response.json().get("response", "")
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Ollama API error: {e}")
            raise

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7
    ) -> str:
        """Chat completion using Ollama API."""
        url = f"{self.endpoint}/api/chat"

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature}
        }

        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            return response.json().get("message", {}).get("content", "")
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Ollama chat error: {e}")
            raise

    def is_available(self) -> bool:
        """Check if Ollama server is running."""
        try:
            response = requests.get(f"{self.endpoint}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False


# ============================================================================
# SKILL REGISTRY
# ============================================================================

class SkillRegistry:
    """Registry for agent skills."""

    def __init__(self):
        self._skills: Dict[str, Callable] = {}
        self._descriptions: Dict[str, str] = {}

    def register(self, name: str, description: str):
        """Decorator to register a skill."""
        def decorator(func: Callable) -> Callable:
            self._skills[name] = func
            self._descriptions[name] = description
            return func
        return decorator

    def get(self, name: str) -> Optional[Callable]:
        """Get a skill by name."""
        return self._skills.get(name)

    def list_skills(self) -> Dict[str, str]:
        """List all registered skills with descriptions."""
        return self._descriptions.copy()

    def execute(self, name: str, **kwargs) -> Any:
        """Execute a skill by name."""
        skill = self.get(name)
        if skill is None:
            raise ValueError(f"Skill not found: {name}")
        return skill(**kwargs)


# Global skill registry
skills = SkillRegistry()


# ============================================================================
# DILI OPTIMIZATION AGENT
# ============================================================================

class DILIOptimizationAgent:
    """
    Main agent for DILI prediction optimization.

    Implements the agentic loop:
    OBSERVE -> PLAN -> EXECUTE -> EVALUATE -> ITERATE

    The agent can ONLY modify: experiments/dili_optimization.ipynb
    """

    SYSTEM_PROMPT = """You are an expert ML scientist optimizing drug-induced liver injury (DILI) prediction models.

Your goal: Achieve the best possible MCC score for DILI classification.

Available data types:
- Molecular fingerprints (Morgan/ECFP, MACCS keys)
- RDKit 2D descriptors
- Mordred descriptors
- Binding data to liver off-targets
- HepG2 cytotoxicity assays
- Mitochondrial toxicity data
- Preclinical toxicity effects

Available ML algorithms:
- Random Forest, XGBoost, Gradient Boosting
- SVM, Logistic Regression, KNN
- MLP Neural Networks
- Ensemble meta-learners

You must output your response as JSON with these fields:
{
    "observation": "What you learned from current state",
    "reasoning": "Your analysis and reasoning",
    "action": "skill_name",
    "parameters": {"key": "value"},
    "expected_outcome": "What you expect to happen"
}
"""

    def __init__(self, config_path: Path = CONFIG_PATH):
        """Initialize the agent."""
        self.config = load_config()
        self.state = AgentState.load()
        self.ollama = OllamaClient(self.config['models']['primary']['endpoint'])
        self.logger = self._setup_logging()
        self.notebook_path = AGENT_ROOT / self.config['paths']['notebook']

        # Load skills
        self._load_skills()

    def _setup_logging(self) -> logging.Logger:
        """Set up logging."""
        logger = logging.getLogger("DILIAgent")
        logger.setLevel(logging.INFO)

        if self.config.get('logging', {}).get('console', True):
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            logger.addHandler(handler)

        return logger

    def _load_skills(self):
        """Load skills from the skills directory."""
        skills_dir = AGENT_ROOT / "skills"
        if skills_dir.exists():
            import sys
            sys.path.insert(0, str(AGENT_ROOT))

            # Import skill modules
            try:
                from skills import (
                    data_finder,
                    feature_engineer,
                    model_optimizer,
                    notebook_writer,
                    validator
                )
                self.logger.info("Skills loaded successfully")
            except ImportError as e:
                self.logger.warning(f"Could not load some skills: {e}")

    def _get_planner_response(self, context: str) -> Dict[str, Any]:
        """Get planning response from the primary model."""
        model = self.config['models']['primary']['name']

        prompt = f"""Current state:
{json.dumps(asdict(self.state), indent=2, default=str)}

Context:
{context}

Available skills:
{json.dumps(skills.list_skills(), indent=2)}

What is your next action? Respond with JSON only."""

        response = self.ollama.generate(
            model=model,
            prompt=prompt,
            system=self.SYSTEM_PROMPT,
            temperature=self.config['models']['primary']['temperature']
        )

        # Parse JSON from response
        try:
            # Extract JSON from response
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end > start:
                return json.loads(response[start:end])
        except json.JSONDecodeError:
            self.logger.error(f"Failed to parse planner response: {response[:200]}")

        return {"action": "none", "reasoning": "Failed to parse response"}

    def _execute_with_executor(self, code: str) -> str:
        """Use the smaller model for code execution tasks."""
        model = self.config['models']['secondary']['name']

        prompt = f"""Execute this code generation task:
{code}

Output only the Python code, no explanations."""

        return self.ollama.generate(
            model=model,
            prompt=prompt,
            temperature=self.config['models']['secondary']['temperature']
        )

    # -------------------------------------------------------------------------
    # AGENTIC LOOP PHASES
    # -------------------------------------------------------------------------

    def observe(self) -> Dict[str, Any]:
        """
        OBSERVE phase: Gather information about current state.
        """
        self.state.phase = Phase.OBSERVE.value
        self.logger.info("Phase: OBSERVE")

        observations = {
            "iteration": self.state.iteration,
            "current_best_mcc": self.state.current_best.get("mcc", 0),
            "data_loaded": self.state.data_sources.get("loaded", []),
            "features_available": len(self.state.features.get("engineered", [])),
            "models_trained": len(self.state.models.get("trained", [])),
            "recent_history": self.state.history[-5:] if self.state.history else []
        }

        # Check notebook state
        if self.notebook_path.exists():
            observations["notebook_exists"] = True
        else:
            observations["notebook_exists"] = False

        self.state.add_observation(f"Observation gathered: {json.dumps(observations)[:200]}")
        return observations

    def plan(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """
        PLAN phase: Determine next action based on observations.
        """
        self.state.phase = Phase.PLAN.value
        self.logger.info("Phase: PLAN")

        # Build context for planner
        context = f"""
Observations: {json.dumps(observations, indent=2)}
Target MCC: {self.config['agent']['targets']['mcc_threshold']}
Max iterations: {self.config['agent']['max_iterations']}
"""

        plan = self._get_planner_response(context)
        self.logger.info(f"Plan: {plan.get('action', 'unknown')}")

        return plan

    def execute(self, plan: Dict[str, Any]) -> Any:
        """
        EXECUTE phase: Perform the planned action.
        """
        self.state.phase = Phase.EXECUTE.value
        self.logger.info(f"Phase: EXECUTE - {plan.get('action', 'none')}")

        action = plan.get('action', 'none')
        params = plan.get('parameters', {})

        if action == 'none':
            return {"status": "skipped", "reason": "No action planned"}

        try:
            result = skills.execute(action, **params)
            self.state.add_to_history(action, result, success=True)
            return result
        except Exception as e:
            error_msg = f"Execution failed: {str(e)}"
            self.state.errors.append(error_msg)
            self.state.add_to_history(action, error_msg, success=False)
            return {"status": "error", "message": error_msg}

    def evaluate(self, result: Any) -> bool:
        """
        EVALUATE phase: Assess the result and determine if goal is met.
        """
        self.state.phase = Phase.EVALUATE.value
        self.logger.info("Phase: EVALUATE")

        # Check if we've reached target metrics
        target_mcc = self.config['agent']['targets']['mcc_threshold']
        current_mcc = self.state.current_best.get('mcc', 0)

        if current_mcc >= target_mcc:
            self.logger.info(f"Target MCC reached: {current_mcc:.4f} >= {target_mcc}")
            return True

        # Check iteration limit
        if self.state.iteration >= self.config['agent']['max_iterations']:
            self.logger.warning("Max iterations reached")
            return True

        return False

    def iterate(self):
        """
        ITERATE phase: Prepare for next iteration.
        """
        self.state.phase = Phase.ITERATE.value
        self.state.iteration += 1

        # Checkpoint if needed
        if self.state.iteration % self.config['agent']['checkpoint_frequency'] == 0:
            checkpoint_name = f"checkpoint_iter_{self.state.iteration}"
            self.state.checkpoints.append(checkpoint_name)
            self.state.save()
            self.logger.info(f"Checkpoint saved: {checkpoint_name}")

    # -------------------------------------------------------------------------
    # MAIN LOOP
    # -------------------------------------------------------------------------

    def run(self, max_iterations: Optional[int] = None):
        """
        Main agentic loop.

        OBSERVE -> PLAN -> EXECUTE -> EVALUATE -> ITERATE
        """
        if max_iterations is None:
            max_iterations = self.config['agent']['max_iterations']

        self.logger.info("="*60)
        self.logger.info("DILI Optimization Agent Started")
        self.logger.info(f"Goal: {self.state.goal}")
        self.logger.info(f"Max iterations: {max_iterations}")
        self.logger.info("="*60)

        # Check Ollama availability
        if not self.ollama.is_available():
            self.logger.error("Ollama server not available. Please start Ollama.")
            return

        while self.state.iteration < max_iterations:
            try:
                # OBSERVE
                observations = self.observe()

                # PLAN
                plan = self.plan(observations)

                # EXECUTE
                result = self.execute(plan)

                # EVALUATE
                goal_reached = self.evaluate(result)

                if goal_reached:
                    self.state.phase = Phase.COMPLETE.value
                    self.logger.info("Goal reached! Agent complete.")
                    break

                # ITERATE
                self.iterate()

                # Small delay to prevent overwhelming the system
                time.sleep(0.5)

            except KeyboardInterrupt:
                self.logger.info("Agent interrupted by user")
                break
            except Exception as e:
                self.logger.error(f"Agent error: {e}")
                self.state.errors.append(str(e))
                self.state.iteration += 1

        # Final save
        self.state.save()
        self.logger.info("Agent state saved.")
        self.logger.info(f"Best MCC achieved: {self.state.current_best.get('mcc', 0):.4f}")

    def reset(self):
        """Reset agent state."""
        self.state = AgentState()
        self.state.save()
        self.logger.info("Agent state reset.")


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """Command-line interface for the DILI agent."""
    import argparse

    parser = argparse.ArgumentParser(
        description="DILI Optimization Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Run command
    run_parser = subparsers.add_parser('run', help='Run the agent')
    run_parser.add_argument(
        '--max-iterations', '-n', type=int, default=None,
        help='Maximum iterations (overrides config)'
    )

    # Reset command
    subparsers.add_parser('reset', help='Reset agent state')

    # Status command
    subparsers.add_parser('status', help='Show current agent status')

    # Skills command
    subparsers.add_parser('skills', help='List available skills')

    args = parser.parse_args()

    if args.command == 'run':
        agent = DILIOptimizationAgent()
        agent.run(max_iterations=args.max_iterations)

    elif args.command == 'reset':
        agent = DILIOptimizationAgent()
        agent.reset()
        print("Agent state reset.")

    elif args.command == 'status':
        state = AgentState.load()
        print(f"Iteration: {state.iteration}")
        print(f"Phase: {state.phase}")
        print(f"Best MCC: {state.current_best.get('mcc', 0):.4f}")
        print(f"Models trained: {len(state.models.get('trained', []))}")
        print(f"Errors: {len(state.errors)}")

    elif args.command == 'skills':
        print("Available skills:")
        for name, desc in skills.list_skills().items():
            print(f"  - {name}: {desc}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()

import os
from typing import Any, TypedDict
from dotenv import load_dotenv

load_dotenv()  # Load .env file if present

# ==================== LLM ====================
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
MODEL_NAME = "deepseek-chat"

# Per-agent model configuration
PLANNER_MODEL = "deepseek-reasoner"
SIMULATOR_MODEL = "deepseek-chat"      # Reserved for Phase 2 (LLM-orchestrated tool calls)
EVALUATOR_MODEL = "deepseek-chat"

# ==================== RAG ====================
DB_DIR = "./chroma_db"
LOCAL_DATA_DIR = "rag_repo"    # Default paper corpus directory
EMBEDDING_MODEL = "BAAI/bge-m3"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
RETRIEVER_K = 6

# ==================== ML Potential ====================
ML_POTENTIAL = "mace"          # "mace" or "chgnet"
MACE_MODEL = "medium"
FMAX = 0.05                    # eV/Å

# ==================== Materials Project ====================
MP_API_KEY = os.environ.get("MP_API_KEY", "")

# ==================== Loop Control ====================
MAX_ITERATIONS = 20
EARLY_STOP_ROUNDS = 3


# ==================== Agent State ====================
class AgentState(TypedDict):
    goal: str
    target_properties: dict

    hypothesis: str
    composition: str
    space_group: int
    reasoning: str

    structure: Any
    relaxed_structure: Any
    computed_properties: dict
    simulation_error: str | None
    simulation_retry_count: int

    verdict: str
    physics_check: dict
    target_check: dict
    suggestion: str

    iteration: int
    max_iterations: int
    history: list[dict]
    no_improvement_count: int
    best_score: float


def compute_score(target_properties: dict, computed_properties: dict) -> float:
    """Composite score: sum of distance-from-target for each property. 0.0 = all met.

    Returns inf if no target properties could be evaluated (all missing/None).
    """
    score = 0.0
    evaluated = 0
    for prop, target in target_properties.items():
        if prop not in computed_properties:
            continue
        actual = computed_properties[prop]
        if actual is None:
            continue
        evaluated += 1
        if "min" in target:
            score += max(0, target["min"] - actual)
        if "max" in target:
            score += max(0, actual - target["max"])
    if evaluated == 0:
        return float("inf")
    return score

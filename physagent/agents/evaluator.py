"""Evaluator Agent: physics validation and target property checking.

Mostly rule-based. LLM only used for generating suggestion text.
"""
import numpy as np
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from physagent.config import API_KEY, BASE_URL, EVALUATOR_MODEL, FMAX, AgentState
from physagent.tools.database_query import query_material


# Physics hard constraint thresholds
MIN_ATOMIC_DISTANCE = 1.0       # Angstrom
FORMATION_ENERGY_MIN = -5.0     # eV/atom
FORMATION_ENERGY_MAX = 2.0      # eV/atom
BANDGAP_PROXY_TOLERANCE = 0.3   # eV — wider tolerance for proxy values


def check_charge_neutrality(structure) -> dict:
    """Check if the structure is charge-neutral based on common oxidation states."""
    try:
        oxi_structure = structure.copy()
        oxi_structure.add_oxidation_state_by_guess()
        total_charge = sum(site.specie.oxi_state for site in oxi_structure)
        passed = abs(total_charge) < 0.1
        return {"passed": passed, "total_charge": round(total_charge, 3)}
    except Exception:
        return {"passed": True, "total_charge": None, "note": "Could not determine oxidation states"}


def check_atomic_distances(structure) -> dict:
    """Check that no two atoms are unreasonably close."""
    if structure is None or len(structure) == 0:
        return {"passed": False, "min_distance": 0.0}
    dist_matrix = structure.distance_matrix
    np.fill_diagonal(dist_matrix, np.inf)
    min_dist = float(np.min(dist_matrix))
    return {"passed": min_dist > MIN_ATOMIC_DISTANCE, "min_distance": round(min_dist, 3)}


def check_formation_energy(computed_properties: dict) -> dict:
    """Check formation energy is in physically reasonable range."""
    e_form = computed_properties.get("formation_energy")
    if e_form is None:
        return {"passed": True, "value": None, "note": "Formation energy unavailable"}
    passed = FORMATION_ENERGY_MIN <= e_form <= FORMATION_ENERGY_MAX
    return {"passed": passed, "value": round(e_form, 4)}


def check_forces_converged(computed_properties: dict) -> dict:
    """Check that forces are converged (max force below threshold)."""
    max_force = computed_properties.get("max_force")
    if max_force is None:
        return {"passed": True, "max_force": None, "note": "Force data unavailable"}
    passed = max_force < FMAX
    return {"passed": passed, "max_force": round(max_force, 4)}


def check_energy_above_hull(composition: str) -> dict:
    """Soft constraint: query energy above hull from Materials Project.

    This is informational — it does NOT cause REJECT. Always returns passed=True.
    The value is included in physics_check for the Planner's reference.
    """
    mp_data = query_material(composition)
    if mp_data is None or mp_data.get("energy_above_hull") is None:
        return {"passed": True, "value": None, "note": "MP data unavailable"}
    e_hull = mp_data["energy_above_hull"]
    if e_hull < 0.025:
        confidence = "high"
    elif e_hull < 0.1:
        confidence = "medium (metastable)"
    else:
        confidence = "low (likely unstable)"
    return {"passed": True, "value": round(e_hull, 4), "confidence": confidence}


def check_targets(target_properties: dict, computed_properties: dict) -> dict:
    """Check each target property against computed values.

    For bandgap_proxy, applies wider tolerance (±0.3 eV).
    """
    results = {}
    for prop, target in target_properties.items():
        actual = computed_properties.get(prop)
        if actual is None:
            results[prop] = {"status": "SKIPPED", "reason": "not computed"}
            continue

        met = True
        tolerance = BANDGAP_PROXY_TOLERANCE if "proxy" in prop else 0.0

        if "min" in target and actual < target["min"] - tolerance:
            met = False
        if "max" in target and actual > target["max"] + tolerance:
            met = False

        results[prop] = {
            "status": "MET" if met else "NOT_MET",
            "required": target,
            "actual": round(actual, 4),
        }
    return results


def determine_verdict(physics_check: dict, target_check: dict) -> str:
    """Determine overall verdict from physics and target checks.

    Returns: "PASS", "REVISE", or "REJECT".
    """
    # Any hard constraint failure -> REJECT
    for check_name, result in physics_check.items():
        if not result.get("passed", True):
            return "REJECT"

    # All targets met -> PASS (SKIPPED counts as not met — we need actual data)
    target_statuses = [v["status"] for v in target_check.values()]
    if target_statuses and all(s == "MET" for s in target_statuses):
        return "PASS"

    # Some targets not met -> REVISE
    return "REVISE"


def generate_suggestion(verdict: str, physics_check: dict, target_check: dict,
                        composition: str) -> str:
    """Use LLM to generate a brief suggestion for the Planner.

    Falls back to rule-based suggestion if LLM unavailable.
    """
    issues = []
    for name, result in physics_check.items():
        if not result.get("passed", True):
            issues.append(f"Physics violation ({name}): {result}")
    for prop, result in target_check.items():
        if result.get("status") == "NOT_MET":
            issues.append(f"Target not met ({prop}): required {result['required']}, got {result['actual']}")

    if not issues:
        return f"All checks passed for {composition}."

    issues_text = "\n".join(issues)

    if not API_KEY:
        return f"Issues found for {composition}:\n{issues_text}"

    try:
        llm = ChatOpenAI(
            api_key=API_KEY, base_url=BASE_URL,
            model=EVALUATOR_MODEL, temperature=0.1
        )
        prompt = ChatPromptTemplate.from_template(
            "你是材料科学评估专家。当前候选材料 {composition} 的评估结果如下：\n"
            "{issues}\n"
            "请用1-2句话给出具体的改进建议（如调整组分、换元素等），直接输出建议内容。"
        )
        chain = prompt | llm | StrOutputParser()
        return chain.invoke({"composition": composition, "issues": issues_text})
    except Exception:
        return f"Issues found for {composition}:\n{issues_text}"


def evaluator_node(state: AgentState) -> dict:
    """LangGraph node function for the Evaluator Agent."""
    structure = state.get("relaxed_structure") or state.get("structure")
    props = state.get("computed_properties", {})
    targets = state.get("target_properties", {})
    composition = state.get("composition", "")

    # If there was a simulation error, skip evaluation
    if state.get("simulation_error"):
        return {
            "verdict": "REVISE",
            "physics_check": {},
            "target_check": {},
            "suggestion": f"Simulation failed: {state['simulation_error']}",
        }

    # Run physics checks
    physics_check = {
        "charge_neutrality": check_charge_neutrality(structure) if structure else {"passed": True},
        "atomic_distances": check_atomic_distances(structure) if structure else {"passed": True},
        "formation_energy": check_formation_energy(props),
        "forces_converged": check_forces_converged(props),
    }

    # Soft constraint: energy above hull (informational, does not cause REJECT)
    e_above_hull = check_energy_above_hull(composition)
    physics_check["energy_above_hull"] = e_above_hull

    # Run target checks
    target_check = check_targets(targets, props)

    # Determine verdict
    verdict = determine_verdict(physics_check, target_check)

    # Generate suggestion
    suggestion = generate_suggestion(verdict, physics_check, target_check, composition)

    return {
        "verdict": verdict,
        "physics_check": physics_check,
        "target_check": target_check,
        "suggestion": suggestion,
    }

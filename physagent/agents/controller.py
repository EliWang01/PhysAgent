"""Self-Correction Controller: pure conditional routing logic, no LLM."""
from physagent.config import AgentState, compute_score, EARLY_STOP_ROUNDS


def update_history(state: AgentState) -> AgentState:
    """Append current iteration to history and update improvement tracking.

    Call this before route() to ensure history and scores are up to date.
    Returns updated state.
    """
    props = state.get("computed_properties", {})
    targets = state.get("target_properties", {})

    score = compute_score(targets, props)

    entry = {
        "iteration": state.get("iteration", 0),
        "composition": state.get("composition", ""),
        "space_group": state.get("space_group", 0),
        "computed_properties": props,
        "verdict": state.get("verdict", ""),
        "score": score,
        "suggestion": state.get("suggestion", ""),
    }

    history = list(state.get("history", []))
    history.append(entry)

    best_score = state.get("best_score", float("inf"))
    no_improvement = state.get("no_improvement_count", 0)

    if score < best_score:
        best_score = score
        no_improvement = 0
    else:
        no_improvement += 1

    return {
        **state,
        "history": history,
        "best_score": best_score,
        "no_improvement_count": no_improvement,
        "iteration": state.get("iteration", 0) + 1,
    }


def route(state: AgentState) -> str:
    """Determine next node based on current state.

    Returns: "planner", "simulator", or "end".
    """
    verdict = state.get("verdict", "")
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", 20)
    no_improvement = state.get("no_improvement_count", 0)
    sim_error = state.get("simulation_error")
    retry_count = state.get("simulation_retry_count", 0)

    # Terminal conditions
    if verdict == "PASS":
        return "end"
    if iteration >= max_iter:
        return "end"
    if no_improvement >= EARLY_STOP_ROUNDS:
        return "end"

    # Simulation error handling
    if sim_error:
        if retry_count == 0:
            return "simulator"
        else:
            return "planner"

    # Evaluator verdicts
    if verdict in ("REJECT", "REVISE"):
        return "planner"

    # Default: continue to planner
    return "planner"


def controller_node(state: AgentState) -> dict:
    """LangGraph node function: update history, compute routing.

    The actual routing is done via conditional edges using route().
    This node just updates the state.
    """
    updated = update_history(state)

    # Reset retry count when routing to planner
    next_node = route(updated)
    if next_node == "planner":
        updated["simulation_retry_count"] = 0
    elif next_node == "simulator":
        updated["simulation_retry_count"] = updated.get("simulation_retry_count", 0) + 1

    return updated

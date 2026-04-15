from physagent.agents.controller import route, update_history, controller_node


def _base_state(**overrides):
    """Create a minimal valid state with overrides."""
    state = {
        "goal": "test",
        "target_properties": {"formation_energy": {"max": 0}},
        "hypothesis": "", "composition": "CsPbI3", "space_group": 221,
        "reasoning": "",
        "structure": None, "relaxed_structure": None,
        "computed_properties": {"formation_energy": -0.5},
        "simulation_error": None, "simulation_retry_count": 0,
        "verdict": "REVISE", "physics_check": {}, "target_check": {},
        "suggestion": "",
        "iteration": 0, "max_iterations": 20,
        "history": [], "no_improvement_count": 0, "best_score": float("inf"),
    }
    state.update(overrides)
    return state


def test_route_pass():
    assert route(_base_state(verdict="PASS")) == "end"


def test_route_max_iterations():
    assert route(_base_state(iteration=20, max_iterations=20)) == "end"


def test_route_early_stop():
    assert route(_base_state(no_improvement_count=3)) == "end"


def test_route_revise():
    assert route(_base_state(verdict="REVISE")) == "planner"


def test_route_reject():
    assert route(_base_state(verdict="REJECT")) == "planner"


def test_route_sim_error_first_retry():
    assert route(_base_state(simulation_error="boom", simulation_retry_count=0)) == "simulator"


def test_route_sim_error_after_retry():
    assert route(_base_state(simulation_error="boom", simulation_retry_count=1)) == "planner"


def test_update_history_appends():
    state = _base_state(computed_properties={"formation_energy": -0.5})
    updated = update_history(state)
    assert len(updated["history"]) == 1
    assert updated["history"][0]["composition"] == "CsPbI3"


def test_update_history_improves_score():
    state = _base_state(
        computed_properties={"formation_energy": -0.5},
        best_score=float("inf"),
    )
    updated = update_history(state)
    assert updated["best_score"] == 0.0  # -0.5 < max 0, so score = 0
    assert updated["no_improvement_count"] == 0


def test_update_history_no_improvement():
    state = _base_state(
        computed_properties={"formation_energy": 0.5},  # above target
        best_score=0.0,  # previous best was perfect
    )
    updated = update_history(state)
    assert updated["no_improvement_count"] == 1


def test_controller_node_resets_retry_on_planner():
    state = _base_state(
        verdict="REVISE",
        simulation_retry_count=1,
    )
    updated = controller_node(state)
    assert updated["simulation_retry_count"] == 0


def test_controller_node_increments_retry_on_simulator():
    state = _base_state(
        simulation_error="boom",
        simulation_retry_count=0,
    )
    updated = controller_node(state)
    assert updated["simulation_retry_count"] == 1

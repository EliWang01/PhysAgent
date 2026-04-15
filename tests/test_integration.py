"""Integration test: full LangGraph loop with mocked LLM and ML potential."""
from unittest.mock import patch
from pymatgen.core import Structure, Lattice

from physagent.config import AgentState


def _mock_planner_node(state):
    """Mock planner that cycles through compositions."""
    iteration = state.get("iteration", 0)
    compositions = ["CsPbI3", "CsSnI3", "CsSn0.5Ge0.5I3"]
    comp = compositions[min(iteration, len(compositions) - 1)]
    return {
        "hypothesis": f"Try {comp}",
        "composition": comp,
        "space_group": 221,
        "reasoning": "Mock reasoning",
        "simulation_error": None,
    }


def _mock_simulator_node(state):
    """Mock simulator that returns fake properties."""
    comp = state.get("composition", "")
    lattice = Lattice.cubic(6.2)
    struct = Structure(
        lattice,
        ["Cs", "Pb", "I", "I", "I"],
        [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
    )
    # Note: bandgap values chosen so only CsSn0.5Ge0.5I3 passes with ±0.3 tolerance
    # CsPbI3: 2.2 > 1.6+0.3=1.9 -> NOT_MET
    # CsSnI3: 0.5 < 1.3-0.3=1.0 -> NOT_MET
    # CsSn0.5Ge0.5I3: 1.45 in [1.0, 1.9] -> MET
    props_map = {
        "CsPbI3": {"formation_energy": -0.3, "bandgap_proxy": 2.2, "max_force": 0.01},
        "CsSnI3": {"formation_energy": -0.5, "bandgap_proxy": 0.5, "max_force": 0.01},
        "CsSn0.5Ge0.5I3": {"formation_energy": -0.4, "bandgap_proxy": 1.45, "max_force": 0.01},
    }
    props = props_map.get(comp, {"formation_energy": 0.0, "bandgap_proxy": 0.0, "max_force": 0.01})
    return {
        "structure": struct,
        "relaxed_structure": struct,
        "computed_properties": props,
        "simulation_error": None,
    }


def _make_initial_state(max_iterations=5):
    return {
        "goal": "Find perovskite",
        "target_properties": {
            "bandgap_proxy": {"min": 1.3, "max": 1.6},
            "formation_energy": {"max": 0},
        },
        "hypothesis": "", "composition": "", "space_group": 221, "reasoning": "",
        "structure": None, "relaxed_structure": None,
        "computed_properties": {}, "simulation_error": None, "simulation_retry_count": 0,
        "verdict": "", "physics_check": {}, "target_check": {}, "suggestion": "",
        "iteration": 0, "max_iterations": max_iterations,
        "history": [], "no_improvement_count": 0, "best_score": float("inf"),
    }


def test_full_loop_finds_candidate():
    """Test that the graph loop converges to a PASS verdict."""
    from langgraph.graph import StateGraph, END
    from physagent.agents.evaluator import evaluator_node
    from physagent.agents.controller import controller_node, route

    graph = StateGraph(AgentState)
    graph.add_node("planner", _mock_planner_node)
    graph.add_node("simulator", _mock_simulator_node)
    graph.add_node("evaluator", evaluator_node)
    graph.add_node("controller", controller_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "simulator")
    graph.add_edge("simulator", "evaluator")
    graph.add_edge("evaluator", "controller")
    graph.add_conditional_edges("controller", route, {
        "planner": "planner",
        "simulator": "simulator",
        "end": END,
    })

    app = graph.compile()

    with patch("physagent.agents.evaluator.generate_suggestion", return_value="Mock suggestion"):
        final = app.invoke(_make_initial_state())

    # CsSn0.5Ge0.5I3 should pass (bandgap 1.45 in [1.0, 1.9] with tolerance, e_form -0.4 < 0)
    assert final["verdict"] == "PASS"
    assert len(final["history"]) >= 1
    assert final["best_score"] == 0.0


def test_loop_terminates_on_max_iterations():
    """Test that the loop stops at max_iterations."""
    from langgraph.graph import StateGraph, END
    from physagent.agents.evaluator import evaluator_node
    from physagent.agents.controller import controller_node, route

    def _always_fail_planner(state):
        return {
            "hypothesis": "bad", "composition": "CsPbI3", "space_group": 221,
            "reasoning": "always same", "simulation_error": None,
        }

    def _always_miss_simulator(state):
        lattice = Lattice.cubic(6.2)
        struct = Structure(
            lattice, ["Cs", "Pb", "I", "I", "I"],
            [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
        )
        return {
            "structure": struct, "relaxed_structure": struct,
            "computed_properties": {"formation_energy": 1.0, "bandgap_proxy": 0.5, "max_force": 0.01},
            "simulation_error": None,
        }

    graph = StateGraph(AgentState)
    graph.add_node("planner", _always_fail_planner)
    graph.add_node("simulator", _always_miss_simulator)
    graph.add_node("evaluator", evaluator_node)
    graph.add_node("controller", controller_node)
    graph.set_entry_point("planner")
    graph.add_edge("planner", "simulator")
    graph.add_edge("simulator", "evaluator")
    graph.add_edge("evaluator", "controller")
    graph.add_conditional_edges("controller", route, {
        "planner": "planner", "simulator": "simulator", "end": END,
    })
    app = graph.compile()

    with patch("physagent.agents.evaluator.generate_suggestion", return_value="Mock"):
        final = app.invoke(_make_initial_state(max_iterations=3))

    # Should stop due to early_stop (3 rounds no improvement) or max_iterations
    assert final["iteration"] >= 3
    assert final["verdict"] != "PASS"

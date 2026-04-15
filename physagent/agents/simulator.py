"""Simulator Agent: orchestrates structure building, relaxation, and property calculation."""
from physagent.config import AgentState
from physagent.tools.structure_builder import build_structure
from physagent.tools.ml_potential import relax_structure, calculate_properties


def simulator_node(state: AgentState) -> dict:
    """LangGraph node function for the Simulator Agent.

    Sequential pipeline:
    1. Build structure from composition + space_group
    2. Relax structure with ML potential
    3. Calculate properties (energy, formation energy, bandgap proxy)

    On any error, captures it as simulation_error instead of crashing.
    """
    composition = state.get("composition", "")
    space_group = state.get("space_group", 221)

    try:
        # Step 1: Build structure
        print(f"  [Simulator] Building structure for {composition} (SG={space_group})...")
        structure = build_structure(composition, space_group)

        # Step 2: Relax
        print(f"  [Simulator] Relaxing structure with ML potential...")
        relax_result = relax_structure(structure)
        relaxed = relax_result["relaxed_structure"]
        converged = relax_result["converged"]

        if not converged:
            print(f"  [Simulator] Warning: relaxation did not converge")

        # Step 3: Calculate properties
        print(f"  [Simulator] Calculating properties...")
        props = calculate_properties(relaxed)

        print(f"  [Simulator] Done. E_form={props.get('formation_energy')}, "
              f"bandgap_proxy={props.get('bandgap_proxy')}")

        return {
            "structure": structure,
            "relaxed_structure": relaxed,
            "computed_properties": props,
            "simulation_error": None,
        }

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"  [Simulator] Error: {error_msg}")
        return {
            "structure": None,
            "relaxed_structure": None,
            "computed_properties": {},
            "simulation_error": error_msg,
        }

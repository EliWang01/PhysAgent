"""PhysAgent main entry point: LangGraph state graph definition + CLI."""
import argparse
import json
import os
import time
from datetime import datetime

from langgraph.graph import StateGraph, END

from physagent.config import AgentState, MAX_ITERATIONS
from physagent.agents.planner import planner_node
from physagent.agents.simulator import simulator_node
from physagent.agents.evaluator import evaluator_node
from physagent.agents.controller import controller_node, route


def enrich_knowledge_base(goal: str, max_papers: int = 5):
    """Search for papers relevant to the goal and add them to the knowledge base.

    Uses paper_fetcher to search arXiv/PubMed, downloads PDFs,
    then incrementally updates the ChromaDB vectorstore.
    """
    from physagent.paper_fetcher import search_papers, download_papers, sanitize_filename
    from physagent.memory.rag_store import build_vectorstore

    print("\n  [Enrich] Searching for papers related to the goal...")

    # Extract English keywords from goal for better search
    papers = search_papers(goal, max_results=max_papers)
    if not papers:
        print("  [Enrich] No papers found. Using existing knowledge base.")
        return

    downloadable = [p for p in papers if p.get("pdf_url")]
    print(f"  [Enrich] Found {len(papers)} papers, {len(downloadable)} downloadable.")

    if not downloadable:
        print("  [Enrich] No downloadable papers. Using existing knowledge base.")
        return

    # Download to a goal-specific subdirectory
    save_dir = os.path.join("papers", sanitize_filename(goal))
    downloaded = download_papers(downloadable, save_dir)

    if downloaded:
        print(f"  [Enrich] Downloaded {len(downloaded)} papers. Updating knowledge base...")
        build_vectorstore(save_dir)
        print("  [Enrich] Knowledge base updated.")
    else:
        print("  [Enrich] No papers downloaded. Using existing knowledge base.")


def build_graph() -> StateGraph:
    """Build the PhysAgent LangGraph state graph."""
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("planner", planner_node)
    graph.add_node("simulator", simulator_node)
    graph.add_node("evaluator", evaluator_node)
    graph.add_node("controller", controller_node)

    # Linear edges
    graph.set_entry_point("planner")
    graph.add_edge("planner", "simulator")
    graph.add_edge("simulator", "evaluator")
    graph.add_edge("evaluator", "controller")

    # Conditional edges from controller
    # Note: Controller is a separate node (not just a routing function on evaluator)
    # because it needs to mutate state (update history, scores, retry counts) before routing.
    graph.add_conditional_edges("controller", route, {
        "planner": "planner",
        "simulator": "simulator",
        "end": END,
    })

    return graph.compile()


def run(goal: str, target_properties: dict, max_iterations: int = None, enrich: bool = False):
    """Run the PhysAgent discovery loop.

    Args:
        goal: Natural language description of the materials discovery goal.
        target_properties: Dict of property targets, e.g.
            {"bandgap_proxy": {"min": 1.3, "max": 1.6}, "formation_energy": {"max": 0}}
        max_iterations: Override for max iteration count.
        enrich: If True, search and download relevant papers before running.

    Returns:
        Final AgentState dict.
    """
    # Goal-aware knowledge enrichment
    if enrich:
        enrich_knowledge_base(goal)

    app = build_graph()

    initial_state = {
        "goal": goal,
        "target_properties": target_properties,
        "hypothesis": "",
        "composition": "",
        "space_group": 221,
        "reasoning": "",
        "structure": None,
        "relaxed_structure": None,
        "computed_properties": {},
        "simulation_error": None,
        "simulation_retry_count": 0,
        "verdict": "",
        "physics_check": {},
        "target_check": {},
        "suggestion": "",
        "iteration": 0,
        "max_iterations": max_iterations or MAX_ITERATIONS,
        "history": [],
        "no_improvement_count": 0,
        "best_score": float("inf"),
    }

    print("=" * 70)
    print("  PhysAgent Phase 1 — Materials Discovery Loop")
    print("=" * 70)
    print(f"  Goal: {goal}")
    print(f"  Targets: {json.dumps(target_properties, ensure_ascii=False)}")
    print(f"  Max iterations: {initial_state['max_iterations']}")
    print("=" * 70)

    start_time = time.time()
    final_state = app.invoke(initial_state)
    elapsed = time.time() - start_time

    # Print summary
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)

    if final_state.get("verdict") == "PASS":
        print(f"  Candidate found: {final_state.get('composition')}")
    else:
        print(f"  No candidate met all targets.")
        print(f"  Best score: {final_state.get('best_score', 'N/A')}")

    print(f"  Iterations: {final_state.get('iteration', 0)}")
    print(f"  Time: {elapsed:.1f}s")

    # Print history
    history = final_state.get("history", [])
    if history:
        print(f"\n  Trajectory ({len(history)} iterations):")
        for entry in history:
            props = entry.get("computed_properties", {})
            print(f"    #{entry['iteration']}: {entry['composition']} | "
                  f"E_form={props.get('formation_energy', 'N/A')} | "
                  f"Eg_proxy={props.get('bandgap_proxy', 'N/A')} | "
                  f"{entry['verdict']} | score={entry['score']:.3f}")

    # Save trajectory
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"results/run_{timestamp}.json"

    # Remove non-serializable fields for JSON output
    serializable_state = {}
    for k, v in final_state.items():
        if k in ("structure", "relaxed_structure"):
            serializable_state[k] = str(v) if v else None
        elif k == "computed_properties":
            clean_props = {}
            for pk, pv in v.items():
                try:
                    import numpy as np
                    if isinstance(pv, np.ndarray):
                        clean_props[pk] = pv.tolist()
                    else:
                        clean_props[pk] = pv
                except ImportError:
                    clean_props[pk] = pv
            serializable_state[k] = clean_props
        else:
            serializable_state[k] = v

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable_state, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  Trajectory saved to: {output_path}")

    # Generate paper
    print("\n  Writing paper draft...")
    try:
        from physagent.agents.writer import generate_paper
        paper = generate_paper(final_state)
        paper_path = f"results/paper_{timestamp}.md"
        with open(paper_path, "w", encoding="utf-8") as f:
            f.write(paper)
        print(f"  Paper saved to: {paper_path}")
    except Exception as e:
        print(f"  Paper generation failed: {e}")

    return final_state


def main():
    """CLI entry point."""
    # Default target for perovskite demo
    DEFAULT_TARGET = {
        "bandgap_proxy": {"min": 1.3, "max": 1.6},
        "formation_energy": {"max": 0},
    }

    parser = argparse.ArgumentParser(description="PhysAgent: Materials Discovery Loop")
    parser.add_argument(
        "--goal", type=str,
        default="Find an ABX3 perovskite with bandgap 1.3-1.6 eV and formation energy < 0",
        help="Materials discovery goal (natural language)",
    )
    parser.add_argument(
        "--target", type=str, default=None,
        help='Target properties as JSON string (default: perovskite bandgap + formation energy)',
    )
    parser.add_argument(
        "--max-iterations", type=int, default=None,
        help=f"Max iterations (default: {MAX_ITERATIONS})",
    )
    parser.add_argument(
        "--enrich", action="store_true",
        help="Search and download relevant papers before running (goal-aware RAG enrichment)",
    )

    args = parser.parse_args()
    target_properties = json.loads(args.target) if args.target else DEFAULT_TARGET
    run(args.goal, target_properties, args.max_iterations, enrich=args.enrich)


if __name__ == "__main__":
    main()

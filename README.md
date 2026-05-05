# PhysAgent

A physics-constrained multi-agent system for materials discovery.

PhysAgent is a closed-loop automated materials discovery system. It uses LangGraph to orchestrate multiple collaborating AI agents and combines them with machine-learning interatomic potentials (MACE-MP-0/CHGNet) for real materials property calculations. The system automates the full loop from hypothesis generation to property validation and iterative optimization. After the discovery loop finishes, it can automatically generate an academic paper draft focused on the discovered material.

## Research Motivation

Current AI-for-materials work is often centered on single-point models such as property prediction or structure generation. Few systems can autonomously complete the full workflow of "propose hypothesis -> compute and validate -> analyze results -> improve iteratively". In particular, existing systems leave three gaps:

1. **Lack of closed-loop iteration**: Systems such as MatPilot and AtomAgents demonstrate one-shot discovery, but do not provide automatic iterative correction.
2. **Lack of physics constraints**: Many existing agents rely on LLM text-based self-reflection for correction instead of feedback driven by numerical physical quantities.
3. **Lack of deep computation**: Many existing agents mainly perform API calls and do not involve real structure relaxation or property calculation.

PhysAgent addresses these gaps by building a physics-constrained multi-agent closed loop for materials discovery. Its core idea is to drive agent self-correction with numerical physical quantities such as formation energy, bandgap, and force convergence rather than text reflection. By integrating ML potentials such as MACE-MP-0, the system performs real materials property calculations and iteratively approaches target materials.

## System Architecture

PhysAgent uses a LangGraph state machine to orchestrate four agent nodes in a closed-loop iteration:

```text
Planner Agent -> Simulator Agent -> Evaluator Agent -> Controller
  (DeepSeek-R1)   (MACE-MP-0)       (physics checks)    (numeric routing)
  RAG retrieval   Pymatgen builder   hard constraints    PASS -> output candidate
  hypothesis      ASE relaxation     distances/forces    REVISE -> Planner
  JSON output     property calc.     bandgap/energy      REJECT -> new proposal
       ^                                                     |
       +---------------- physics feedback -------------------+
```

### Agent Details

**Planner Agent**

- Uses the DeepSeek-R1 reasoning model with `temperature=0.7` to balance creativity and plausibility.
- Retrieves academic literature from ChromaDB through RAG and generates candidate material hypotheses based on literature evidence.
- Outputs structured JSON fields: chemical formula (`composition`), space group (`space_group`), and reasoning process (`reasoning`).
- Receives feedback from the Evaluator and history from prior iterations to avoid repeatedly exploring failed compositions.

**Simulator Agent**

- Uses Pymatgen to build ABX3 perovskite crystal structures, including both pure compositions and mixed-composition supercells.
- Performs structure relaxation through ASE + MACE-MP-0 using a BFGS optimizer with `fmax=0.05 eV/Angstrom`.
- Computes material properties such as total energy, formation energy based on elemental reference states, forces, and stress.
- Estimates the bandgap through interpolation from known endpoint values (`bandgap_proxy`).

**Evaluator Agent**

- Applies hard physics constraints. If any hard constraint fails, the verdict is `REJECT`:
  - Charge neutrality: oxidation-state sum approximately equals 0.
  - Atomic distance: minimum distance > 1.0 Angstrom.
  - Formation energy range: -5.0 to +2.0 eV/atom.
  - Force convergence: maximum force < 0.05 eV/Angstrom.
- Applies soft constraints by querying Materials Project for energy above hull. This is informational and does not directly trigger `REJECT`.
- Checks target properties against user-defined target ranges. The `bandgap_proxy` target uses a +/-0.3 eV tolerance.
- Uses an LLM to generate textual improvement suggestions for the Planner.

**Controller**

- Uses pure Python logic and does not call an LLM.
- Routes based on numerical physical quantities rather than text reflection:
  - `PASS` -> stop and output the candidate material.
  - `REVISE` -> return to the Planner to adjust the composition.
  - `REJECT` -> return to the Planner for a new proposal.
  - Simulation error -> retry the Simulator once, then return to the Planner if it fails again.
- Tracks a composite score, defined as the sum of distances from target properties.
- Stops early after 3 consecutive iterations without score improvement.
- Treats fully missing properties as `inf` score to prevent false-positive `PASS` verdicts.

**Writer Agent**

- Runs automatically after the closed loop finishes.
- Generates a full academic paper draft from the discovery trajectory and RAG literature.
- Focuses the paper on the discovered material, not on the system itself.
- Includes abstract, introduction, computational methods, results and discussion, conclusion, and references.
- Supports long-form paper generation with automatic continuation when output truncation is detected.

### Knowledge Augmentation (RAG)

The system supports two RAG modes:

- **Local mode** (default): Uses a pre-downloaded paper corpus in `rag_repo/` and retrieves from a ChromaDB vector database.
- **Cloud-enriched mode** (`--enrich`): Before running, automatically searches arXiv/PubMed for papers related to the goal, downloads them, and incrementally updates the knowledge base.

The RAG knowledge base uses the `BAAI/bge-m3` embedding model with `chunk_size=1000` and top-k retrieval set to 6.

## Quick Start

### Requirements

- Python 3.12+
- CUDA (optional, for accelerating ML-potential calculations)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Main dependencies include LangGraph, LangChain, Pymatgen, ASE, MACE-torch, CHGNet, ChromaDB, and mp-api.

### 2. Configure API Keys

Create a `.env` file. You can use `.env.example` as a reference:

```env
DEEPSEEK_API_KEY=sk-your-key-here
MP_API_KEY=your-mp-api-key-here
```

| Key | Required | Purpose |
|-----|----------|---------|
| `DEEPSEEK_API_KEY` | Yes | LLM calls for Planner (R1), Evaluator, and Writer |
| `MP_API_KEY` | No | Queries Materials Project data such as energy above hull and known bandgaps. You can register for a free key at https://next-gen.materialsproject.org/. If omitted, the system falls back to a built-in elemental reference energy table. |

### 3. Build the RAG Knowledge Base

```bash
# Download the predefined paper corpus to rag_repo/
# The corpus covers 5 research areas and contains about 50 papers.
python download.py
```

On the first PhysAgent run, the paper corpus is automatically indexed into the ChromaDB vector database at `chroma_db/`.

### 4. Run

```bash
# Default goal: find a perovskite with bandgap 1.3-1.6 eV and formation energy < 0
python -m physagent.main

# Enable cloud paper search to enrich the RAG knowledge base
python -m physagent.main --enrich

# Use a custom goal
python -m physagent.main \
    --goal "Find lead-free ABX3 perovskite with bandgap 1.3-1.6 eV" \
    --max-iterations 10
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--goal` | Find a perovskite with bandgap 1.3-1.6 eV | Natural-language materials discovery goal. English usually works best. |
| `--target` | `bandgap_proxy` [1.3, 1.6] + `formation_energy` < 0 | Target property JSON string. |
| `--max-iterations` | 20 | Maximum number of iterations. |
| `--enrich` | Disabled | Enable cloud paper search and RAG enrichment. |

### Example Scenarios

```bash
# Lead-free perovskite for photovoltaics
python -m physagent.main \
    --goal "Find a lead-free ABX3 perovskite with bandgap 1.3-1.6 eV using only Sn or Ge at B site"

# Wide-bandgap material for tandem solar-cell top cells
python -m physagent.main \
    --goal "Find ABX3 with wide bandgap 1.7-1.9 eV for tandem solar cell" \
    --target '{\"bandgap_proxy\": {\"min\": 1.7, \"max\": 1.9}, \"formation_energy\": {\"max\": 0}}'

# Blue LED material
python -m physagent.main \
    --goal "Find ABX3 perovskite with bandgap 2.5-3.0 eV for blue LED" \
    --target '{\"bandgap_proxy\": {\"min\": 2.5, \"max\": 3.0}, \"formation_energy\": {\"max\": 0}}' \
    --enrich

# Stability-first search
python -m physagent.main \
    --goal "Find the most stable ABX3 halide perovskite with bandgap in visible range" \
    --target '{\"bandgap_proxy\": {\"min\": 1.5, \"max\": 2.5}, \"formation_energy\": {\"max\": -0.5}}'
```

### Outputs

Each run generates files under `results/`:

- `run_<timestamp>.json`: Complete iteration trajectory, including composition, computed properties, evaluation results, and score for each iteration.
- `paper_<timestamp>.md`: Academic paper draft with abstract, introduction, computational methods, results and discussion, conclusion, and references.

## Supported Materials

PhysAgent currently supports halide perovskites with the ABX3 formula:

| Site | Allowed Elements | Description |
|------|------------------|-------------|
| A site | Cs, Rb, K | Inorganic cations |
| B site | Pb, Sn, Ge, Ti, Zr | Divalent metal cations |
| X site | I, Br, Cl, F | Halide anions |

Mixed compositions are supported, such as `CsSn0.5Pb0.5I3` and `CsPbBr1.5Cl1.5`. They are represented through ordered substitutions in a 2x1x1 supercell. Formulas must use a flat format such as `CsPbI3` or `CsSn0.5Ge0.5I3`; parenthesized formulas are not supported.

### Bandgap Estimation

ML potentials such as MACE and CHGNet do not directly predict bandgaps. PhysAgent estimates bandgaps through interpolation from known experimental endpoint values:

| Composition | Experimental Bandgap (eV) |
|-------------|---------------------------|
| CsPbI3 | 1.73 |
| CsPbBr3 | 2.30 |
| CsPbCl3 | 2.90 |
| CsSnI3 | 1.30 |
| CsSnBr3 | 1.75 |
| CsGeI3 | 1.60 |
| ... | ... |

For mixed compositions, the system computes a weighted interpolation using endpoint fractions. The expected accuracy is about +/-0.3 eV, and the Evaluator automatically applies this tolerance during target checks.

## Project Structure

```text
physagent/
|-- config.py                 # Configuration, AgentState TypedDict, compute_score
|-- main.py                   # LangGraph orchestration, CLI entry, enrich_knowledge_base
|-- agents/
|   |-- planner.py            # Hypothesis generation with DeepSeek-R1 + RAG retrieval
|   |-- simulator.py          # Structure building -> ML-potential relaxation -> property calculation
|   |-- evaluator.py          # Hard physics constraints, soft constraints, target checks
|   |-- controller.py         # Numeric routing, history tracking, early stop
|   `-- writer.py             # Automatic academic paper generation
|-- tools/
|   |-- structure_builder.py  # Pymatgen perovskite builder for pure and mixed compositions
|   |-- ml_potential.py       # MACE/CHGNet relaxation, properties, bandgap interpolation
|   `-- database_query.py     # Materials Project API queries and elemental reference fallback
|-- memory/
|   `-- rag_store.py          # ChromaDB build/load/update and RAG chain factory
`-- paper_fetcher.py          # Multi-source paper search: arXiv, Semantic Scholar, PubMed

tests/                        # 58 tests: unit tests and integration tests
download.py                   # Standalone batch paper downloader
```

## Testing

```bash
# Run all 58 tests
python -m pytest tests/ -v

# Run one test file
python -m pytest tests/test_controller.py -v

# Run one test function
python -m pytest tests/test_evaluator.py::test_verdict_pass -v
```

Test coverage includes:

- `test_state.py`: `compute_score` boundary cases (7 tests).
- `test_controller.py`: Routing logic, history tracking, and retry behavior (12 tests).
- `test_evaluator.py`: Physics constraint checks, target checks, and verdict logic (14 tests).
- `test_structure_builder.py`: Composition parsing, structure building, and mixed compositions (10 tests).
- `test_ml_potential.py`: Bandgap interpolation estimates (4 tests).
- `test_database_query.py`: Elemental reference energies and formation-energy calculation (4 tests).
- `test_planner.py`: JSON parsing and history formatting (5 tests).
- `test_integration.py`: Full LangGraph closed loop with mocked agents (2 tests).

## Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Orchestration | LangGraph | State-machine graph definition, conditional routing, closed-loop iteration |
| LLM | DeepSeek-R1 | Planner hypothesis generation with reasoning |
| LLM | DeepSeek-chat | Evaluator suggestion generation and Writer paper drafting |
| ML potential | MACE-MP-0 | Structure relaxation and energy/force/stress calculation |
| ML potential | CHGNet | Alternative calculation backend with magnetic-moment support |
| Structure handling | Pymatgen | Crystal structure construction, composition analysis, oxidation-state inference |
| Atomistic simulation | ASE | BFGS optimizer and calculator interface |
| Vector retrieval | ChromaDB + bge-m3 | Paper knowledge-base storage and semantic retrieval |
| Materials data | Materials Project API | Known material properties and energy-above-hull references |
| Paper search | arXiv + PubMed | Cloud paper search and download in `--enrich` mode |

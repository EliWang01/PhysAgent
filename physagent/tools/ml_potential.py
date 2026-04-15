"""ML potential calculations: structure relaxation, property computation, bandgap proxy."""
import math
import numpy as np
from pymatgen.core import Structure, Element
from pymatgen.io.ase import AseAtomsAdaptor

from physagent.config import ML_POTENTIAL, MACE_MODEL, FMAX
from physagent.tools.database_query import get_elemental_references
from physagent.tools.structure_builder import (
    parse_perovskite_composition, IONIC_RADII, goldschmidt_tolerance, _get_avg_radius,
)


def compute_formation_energy(total_energy: float, structure, calculator_refs: dict | None = None) -> float | None:
    """Compute formation energy per atom from total energy and elemental references.

    Args:
        total_energy: Total energy of the compound (eV)
        structure: Pymatgen Structure object
        calculator_refs: Optional pre-computed elemental references

    Returns:
        Formation energy per atom (eV/atom), or None if references unavailable.
    """
    comp = structure.composition
    elements = [str(el) for el in comp.elements]
    refs = calculator_refs or get_elemental_references(elements)

    for el in elements:
        if el not in refs:
            return None

    n_atoms = comp.num_atoms
    ref_energy = sum(comp[el] * refs[el] for el in elements)
    return (total_energy - ref_energy) / n_atoms


def _get_calculator(calculator: str = None):
    """Load ML potential calculator."""
    calc_name = calculator or ML_POTENTIAL
    if calc_name == "mace":
        from mace.calculators import mace_mp
        return mace_mp(model=MACE_MODEL, default_dtype="float64")
    elif calc_name == "chgnet":
        from chgnet.model import CHGNetCalculator
        return CHGNetCalculator()
    else:
        raise ValueError(f"Unknown calculator: {calc_name}")


def relax_structure(structure: Structure, calculator: str = None) -> dict:
    """Relax a structure using ML potential.

    Returns:
        {"relaxed_structure": Structure, "energy": float, "forces": ndarray, "converged": bool}
    """
    from ase.optimize import BFGS

    adaptor = AseAtomsAdaptor()
    atoms = adaptor.get_atoms(structure)
    atoms.calc = _get_calculator(calculator)

    optimizer = BFGS(atoms, logfile=None)
    try:
        converged = optimizer.run(fmax=FMAX, steps=500)
    except Exception:
        converged = False

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    relaxed_structure = adaptor.get_structure(atoms)

    return {
        "relaxed_structure": relaxed_structure,
        "energy": float(energy),
        "forces": forces,
        "converged": bool(converged),
    }


def calculate_properties(structure: Structure, calculator: str = None) -> dict:
    """Calculate properties of a structure using ML potential.

    Returns dict with: total_energy, energy_per_atom, formation_energy,
    forces, stress, max_force.
    """
    adaptor = AseAtomsAdaptor()
    atoms = adaptor.get_atoms(structure)
    atoms.calc = _get_calculator(calculator)

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    stress = atoms.get_stress()
    n_atoms = len(atoms)

    # Formation energy
    elements = [str(el) for el in structure.composition.elements]
    refs = get_elemental_references(elements)
    form_energy = compute_formation_energy(energy, structure, calculator_refs=refs)

    # Bandgap proxy
    bandgap_result = estimate_bandgap_proxy(structure)

    props = {
        "total_energy": float(energy),
        "energy_per_atom": float(energy / n_atoms),
        "formation_energy": form_energy,
        "forces": forces,
        "stress": stress,
        "max_force": float(np.max(np.abs(forces))),
    }
    # Merge bandgap proxy results
    props.update(bandgap_result)
    return props


# Known experimental bandgaps (eV) for ABX3 halide perovskites
# Used for interpolation-based bandgap proxy
KNOWN_BANDGAPS = {
    ("Cs", "Pb", "I"): 1.73,
    ("Cs", "Pb", "Br"): 2.30,
    ("Cs", "Pb", "Cl"): 2.90,
    ("Cs", "Sn", "I"): 1.30,
    ("Cs", "Sn", "Br"): 1.75,
    ("Cs", "Sn", "Cl"): 2.40,
    ("Cs", "Ge", "I"): 1.60,
    ("Cs", "Ge", "Br"): 2.10,
    ("Rb", "Pb", "I"): 1.53,
    ("Rb", "Sn", "I"): 1.25,
    ("K", "Pb", "I"): 1.60,
}


def _parse_composition_as_perovskite(structure: Structure) -> dict | None:
    """Try to interpret a structure's composition as ABX3 perovskite.

    Handles both simple formulas (CsPbI3) and supercell formulas (Cs2SnPbI6)
    by normalizing to ABX3 stoichiometry.
    """
    A_ions = {"Cs", "Rb", "K"}
    B_ions = {"Pb", "Sn", "Ge", "Ti", "Zr"}
    X_ions = {"I", "Br", "Cl", "F"}

    comp = structure.composition
    sites = {"A": {}, "B": {}, "X": {}}

    for el in comp.elements:
        el_str = str(el)
        amount = comp[el]
        if el_str in A_ions:
            sites["A"][el_str] = amount
        elif el_str in B_ions:
            sites["B"][el_str] = amount
        elif el_str in X_ions:
            sites["X"][el_str] = amount
        else:
            return None

    a_sum = sum(sites["A"].values())
    b_sum = sum(sites["B"].values())
    x_sum = sum(sites["X"].values())

    if a_sum == 0 or b_sum == 0 or x_sum == 0:
        return None

    # Check ABX3 ratio: A:B:X should be 1:1:3
    ratio_check = abs(a_sum / b_sum - 1.0) < 0.1 and abs(x_sum / b_sum - 3.0) < 0.1
    if not ratio_check:
        return None

    # Normalize to fractions (A=1, B=1, X=3)
    for site_key, factor in [("A", a_sum), ("B", b_sum), ("X", x_sum / 3)]:
        sites[site_key] = {el: amt / factor for el, amt in sites[site_key].items()}

    return sites


def estimate_bandgap_proxy(structure: Structure) -> dict:
    """Estimate bandgap for perovskite structures using interpolation from known values.

    For mixed compositions (e.g. CsPb0.5Sn0.5I3), linearly interpolates between
    known endpoint bandgaps weighted by site fractions.

    Returns:
        {"bandgap_proxy": float|None, "tolerance_factor": float|None, "is_proxy": True}
    """
    # Try to parse as perovskite from structure composition (handles supercells)
    sites = _parse_composition_as_perovskite(structure)

    if sites is None:
        # Fallback: try reduced formula string parsing
        comp_str = structure.composition.reduced_formula
        sites = parse_perovskite_composition(comp_str)

    if sites is None:
        # Not a perovskite — try Materials Project lookup
        from physagent.tools.database_query import query_material
        comp_str = structure.composition.reduced_formula
        mp_data = query_material(comp_str)
        if mp_data and mp_data.get("bandgap") is not None:
            return {
                "bandgap_proxy": mp_data["bandgap"],
                "tolerance_factor": None,
                "is_proxy": True,
            }
        return {"bandgap_proxy": None, "tolerance_factor": None, "is_proxy": True}

    r_A = _get_avg_radius(sites["A"])
    r_B = _get_avg_radius(sites["B"])
    r_X = _get_avg_radius(sites["X"])
    t = goldschmidt_tolerance(r_A, r_B, r_X)

    # Interpolation-based bandgap estimation
    # For each combination of (A, B, X) endpoints, look up known bandgap
    # then weight by fractional occupancy
    bandgap = 0.0
    total_weight = 0.0
    for a_el, a_frac in sites["A"].items():
        for b_el, b_frac in sites["B"].items():
            for x_el, x_frac in sites["X"].items():
                key = (a_el, b_el, x_el)
                if key in KNOWN_BANDGAPS:
                    # Weight = product of site fractions (x_frac normalized to sum=1)
                    weight = a_frac * b_frac * (x_frac / 3.0)
                    bandgap += KNOWN_BANDGAPS[key] * weight
                    total_weight += weight

    if total_weight > 0:
        bandgap = bandgap / total_weight
    else:
        # No known endpoints found, return None
        return {"bandgap_proxy": None, "tolerance_factor": round(t, 4), "is_proxy": True}

    return {
        "bandgap_proxy": round(bandgap, 3),
        "tolerance_factor": round(t, 4),
        "is_proxy": True,
    }

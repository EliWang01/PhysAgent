"""Crystal structure construction for perovskite ABX3 and general compositions."""
import re
import math
from pymatgen.core import Structure, Lattice, Element


# Shannon ionic radii (coordination VI) for common perovskite ions
IONIC_RADII = {
    "Cs": 1.67, "Rb": 1.52, "K": 1.38,
    "Pb": 1.19, "Sn": 1.12, "Ge": 0.73, "Ti": 0.605, "Zr": 0.72,
    "I": 2.20, "Br": 1.96, "Cl": 1.81, "F": 1.33,
    # Organic cation effective radii
    "MA": 2.17, "FA": 2.53,
}

# Perovskite prototype: cubic Pm-3m (#221)
# Fractional coords: A at (0,0,0), B at (0.5,0.5,0.5), X at (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
PEROVSKITE_COORDS = {
    "A": [[0.0, 0.0, 0.0]],
    "B": [[0.5, 0.5, 0.5]],
    "X": [[0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]],
}


def estimate_lattice_param(r_B: float, r_X: float) -> float:
    """Estimate cubic perovskite lattice parameter from ionic radii: a = 2*(r_B + r_X)."""
    return 2.0 * (r_B + r_X)


def goldschmidt_tolerance(r_A: float, r_B: float, r_X: float) -> float:
    """Goldschmidt tolerance factor: t = (r_A + r_X) / (sqrt(2) * (r_B + r_X))."""
    return (r_A + r_X) / (math.sqrt(2) * (r_B + r_X))


def parse_perovskite_composition(composition: str) -> dict | None:
    """Parse ABX3 composition string into {A: {el: frac}, B: {el: frac}, X: {el: frac}}.

    Supports: CsPbI3, CsSn0.5Ge0.5I3, CsPbBr1.5Cl1.5
    Returns None if not a recognizable perovskite.
    """
    A_ions = {"Cs", "Rb", "K", "MA", "FA"}
    B_ions = {"Pb", "Sn", "Ge", "Ti", "Zr", "Ca", "Sr", "Ba"}
    X_ions = {"I", "Br", "Cl", "F"}

    # Tokenize: split into (element, fraction) pairs
    tokens = re.findall(r'([A-Z][a-z]*)(\d*\.?\d*)', composition)
    if not tokens:
        return None

    sites = {"A": {}, "B": {}, "X": {}}
    for el, frac_str in tokens:
        if not el:
            continue
        frac = float(frac_str) if frac_str else 1.0
        if el in A_ions:
            sites["A"][el] = frac
        elif el in B_ions:
            sites["B"][el] = frac
        elif el in X_ions:
            sites["X"][el] = frac
        else:
            return None  # Unknown element for perovskite

    # Validate stoichiometry: A=1, B=1, X=3
    a_sum = sum(sites["A"].values())
    b_sum = sum(sites["B"].values())
    x_sum = sum(sites["X"].values())
    if abs(a_sum - 1.0) > 0.01 or abs(b_sum - 1.0) > 0.01 or abs(x_sum - 3.0) > 0.01:
        return None

    return sites


def _get_avg_radius(site_dict: dict) -> float:
    """Weighted average ionic radius for a site."""
    total = sum(site_dict.values())
    return sum(IONIC_RADII.get(el, 1.0) * frac / total for el, frac in site_dict.items())


def build_perovskite(composition: str, space_group: int = 221) -> Structure:
    """Build a perovskite ABX3 structure.

    For pure compositions: returns a 5-atom unit cell.
    For mixed compositions: returns a 2x1x1 supercell (10 atoms) with ordered substitution.
    """
    sites = parse_perovskite_composition(composition)
    if sites is None:
        raise ValueError(f"Cannot parse '{composition}' as perovskite ABX3")

    r_A = _get_avg_radius(sites["A"])
    r_B = _get_avg_radius(sites["B"])
    r_X = _get_avg_radius(sites["X"])
    a = estimate_lattice_param(r_B, r_X)

    is_mixed = any(len(v) > 1 for v in sites.values())

    if not is_mixed:
        # Pure ABX3: simple 5-atom cell
        A_el = list(sites["A"].keys())[0]
        B_el = list(sites["B"].keys())[0]
        X_el = list(sites["X"].keys())[0]
        lattice = Lattice.cubic(a)
        species = [A_el, B_el, X_el, X_el, X_el]
        coords = (PEROVSKITE_COORDS["A"] + PEROVSKITE_COORDS["B"]
                  + PEROVSKITE_COORDS["X"])
        return Structure(lattice, species, coords)
    else:
        # Mixed: build 2x1x1 supercell with ordered substitution
        # First build with the majority element, then substitute
        A_el = max(sites["A"], key=sites["A"].get)
        B_el = max(sites["B"], key=sites["B"].get)
        X_el = max(sites["X"], key=sites["X"].get)

        lattice = Lattice.cubic(a)
        species = [A_el, B_el, X_el, X_el, X_el]
        coords = (PEROVSKITE_COORDS["A"] + PEROVSKITE_COORDS["B"]
                  + PEROVSKITE_COORDS["X"])
        unit_cell = Structure(lattice, species, coords)
        supercell = unit_cell * [2, 1, 1]  # 10 atoms

        # Substitute minority elements on their sites
        for site_key, site_dict in sites.items():
            if len(site_dict) <= 1:
                continue
            majority_el = max(site_dict, key=site_dict.get)
            # Find indices of this site type in supercell
            site_indices = [i for i, s in enumerate(supercell)
                           if str(s.specie) == majority_el
                           and _is_site_type(s.frac_coords, site_key)]
            # Replace half with minority element
            for minority_el, frac in site_dict.items():
                if minority_el == majority_el:
                    continue
                n_replace = round(frac * len(site_indices))
                for idx in site_indices[:n_replace]:
                    supercell.replace(idx, minority_el)

        return supercell


def _is_site_type(frac_coords, site_key: str) -> bool:
    """Check if fractional coordinates match a perovskite site type.
    # TODO Phase 2: implement proper crystallographic site matching for complex substitutions
    """
    return True


def build_structure(composition: str, space_group: int = 221) -> Structure:
    """Build a crystal structure from composition and space group.

    For perovskite ABX3 compositions: uses prototype-based construction.
    For other compositions: raises ValueError (extend in Phase 2).
    """
    sites = parse_perovskite_composition(composition)
    if sites is not None:
        return build_perovskite(composition, space_group)
    raise ValueError(
        f"Cannot build structure for '{composition}' with space group {space_group}. "
        f"Only perovskite ABX3 compositions are supported in Phase 1."
    )

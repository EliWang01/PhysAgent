"""Materials Project API queries with offline fallback."""
from pymatgen.core import Composition
from physagent.config import MP_API_KEY

# Hardcoded elemental reference energies (eV/atom) from Materials Project
# These are ground-state energies for common perovskite elements
ELEMENTAL_REFERENCES = {
    "Cs": -0.856,   # mp-1  bcc Cs
    "Pb": -3.704,   # mp-20483
    "Sn": -3.836,   # mp-117
    "Ge": -4.623,   # mp-32
    "I":  -1.343,   # mp-23155
    "Br": -1.318,   # mp-23154
    "Cl": -1.748,   # mp-22848
    "C":  -9.227,   # mp-66  diamond
    "N":  -8.336,   # mp-154
    "H":  -3.393,   # mp-24504
}


def query_material(composition: str) -> dict | None:
    """Query Materials Project for known properties of a composition.

    Returns dict with formation_energy, bandgap, energy_above_hull, space_group,
    or None if API unavailable or composition not found.
    """
    if not MP_API_KEY:
        return None
    try:
        from mp_api.client import MPRester
        with MPRester(MP_API_KEY) as mpr:
            docs = mpr.materials.summary.search(
                formula=composition,
                fields=["material_id", "formation_energy_per_atom",
                         "band_gap", "energy_above_hull", "symmetry"]
            )
            if not docs:
                return None
            # Pick the most stable entry (lowest energy above hull)
            doc = min(docs, key=lambda d: d.energy_above_hull)
            return {
                "material_id": str(doc.material_id),
                "formation_energy": doc.formation_energy_per_atom,
                "bandgap": doc.band_gap,
                "energy_above_hull": doc.energy_above_hull,
                "space_group": doc.symmetry.symbol if doc.symmetry else None,
            }
    except Exception:
        return None


def get_elemental_references(elements: list[str]) -> dict:
    """Get ground-state energies for elements. Uses MP API with hardcoded fallback.

    Returns dict mapping element symbol -> energy_per_atom (eV).
    """
    refs = {}
    missing = []
    # First try hardcoded values
    for el in elements:
        if el in ELEMENTAL_REFERENCES:
            refs[el] = ELEMENTAL_REFERENCES[el]
        else:
            missing.append(el)

    # Try MP API for any missing elements
    if missing and MP_API_KEY:
        try:
            from mp_api.client import MPRester
            with MPRester(MP_API_KEY) as mpr:
                for el in missing:
                    docs = mpr.materials.summary.search(
                        formula=el,
                        fields=["formation_energy_per_atom", "energy_above_hull"]
                    )
                    if docs:
                        stable = min(docs, key=lambda d: d.energy_above_hull)
                        refs[el] = stable.formation_energy_per_atom
        except Exception:
            pass

    return refs

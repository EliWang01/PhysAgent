from unittest.mock import patch
from physagent.tools.database_query import (
    get_elemental_references,
    ELEMENTAL_REFERENCES,
)
from physagent.tools.ml_potential import compute_formation_energy


def test_elemental_references_hardcoded():
    refs = get_elemental_references(["Cs", "Pb", "I"])
    assert "Cs" in refs
    assert "Pb" in refs
    assert "I" in refs
    assert all(isinstance(v, float) for v in refs.values())


def test_elemental_references_missing_element_no_api():
    """Unknown element without API key returns partial dict."""
    with patch("physagent.tools.database_query.MP_API_KEY", ""):
        refs = get_elemental_references(["Cs", "Unobtanium"])
        assert "Cs" in refs
        assert "Unobtanium" not in refs


def test_compute_formation_energy():
    """Test formation energy calculation with known references."""
    from pymatgen.core import Structure, Lattice

    # Simple cubic CsPbI3 (5 atoms)
    lattice = Lattice.cubic(6.2)
    structure = Structure(
        lattice,
        ["Cs", "Pb", "I", "I", "I"],
        [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]
    )
    refs = {"Cs": -0.856, "Pb": -3.704, "I": -1.343}
    # total_energy chosen so formation energy = -0.5 eV/atom
    ref_sum = 1 * (-0.856) + 1 * (-3.704) + 3 * (-1.343)  # = -8.589
    total_energy = ref_sum + (-0.5 * 5)  # -8.589 + -2.5 = -11.089
    result = compute_formation_energy(total_energy, structure, calculator_refs=refs)
    assert result is not None
    assert abs(result - (-0.5)) < 1e-6


def test_compute_formation_energy_missing_ref():
    """Returns None if element reference is missing."""
    from pymatgen.core import Structure, Lattice

    lattice = Lattice.cubic(6.0)
    structure = Structure(lattice, ["Xx"], [[0, 0, 0]])
    result = compute_formation_energy(-5.0, structure, calculator_refs={})
    assert result is None

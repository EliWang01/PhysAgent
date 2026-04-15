"""Tests for ML potential tool. Uses bandgap proxy (no ML model needed)."""
from pymatgen.core import Structure, Lattice
from physagent.tools.ml_potential import estimate_bandgap_proxy


def _make_cspbi3():
    """Helper: build a simple CsPbI3 structure."""
    lattice = Lattice.cubic(6.2)
    return Structure(
        lattice,
        ["Cs", "Pb", "I", "I", "I"],
        [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
    )


def test_bandgap_proxy_returns_dict():
    struct = _make_cspbi3()
    result = estimate_bandgap_proxy(struct)
    assert "bandgap_proxy" in result
    assert "tolerance_factor" in result
    assert result["is_proxy"] is True


def test_bandgap_proxy_positive():
    struct = _make_cspbi3()
    result = estimate_bandgap_proxy(struct)
    assert result["bandgap_proxy"] is not None
    assert result["bandgap_proxy"] >= 0.0


def test_tolerance_factor_reasonable():
    struct = _make_cspbi3()
    result = estimate_bandgap_proxy(struct)
    t = result["tolerance_factor"]
    assert t is not None
    assert 0.7 < t < 1.1  # CsPbI3 should be ~0.81


def test_bandgap_proxy_non_perovskite():
    """Non-perovskite: returns None if no MP API, or a real value if MP API is available."""
    lattice = Lattice.cubic(5.64)
    struct = Structure(lattice, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    result = estimate_bandgap_proxy(struct)
    assert result["tolerance_factor"] is None
    assert result["is_proxy"] is True
    # bandgap_proxy is None (no API) or a float (from MP lookup)
    assert result["bandgap_proxy"] is None or isinstance(result["bandgap_proxy"], float)

import math
import pytest
from physagent.tools.structure_builder import (
    parse_perovskite_composition,
    goldschmidt_tolerance,
    estimate_lattice_param,
    build_structure,
)


def test_parse_pure_perovskite():
    result = parse_perovskite_composition("CsPbI3")
    assert result is not None
    assert result["A"] == {"Cs": 1.0}
    assert result["B"] == {"Pb": 1.0}
    assert result["X"] == {"I": 3.0}


def test_parse_mixed_B_site():
    result = parse_perovskite_composition("CsSn0.5Ge0.5I3")
    assert result is not None
    assert result["B"] == {"Sn": 0.5, "Ge": 0.5}


def test_parse_mixed_X_site():
    result = parse_perovskite_composition("CsPbBr1.5Cl1.5")
    assert result is not None
    assert result["X"] == {"Br": 1.5, "Cl": 1.5}


def test_parse_invalid_stoichiometry():
    result = parse_perovskite_composition("CsPbI2")
    assert result is None


def test_parse_unknown_element():
    result = parse_perovskite_composition("CsZnI3")
    assert result is None


def test_goldschmidt():
    t = goldschmidt_tolerance(1.67, 1.19, 2.20)  # CsPbI3
    assert 0.8 < t < 1.1  # Should be around 0.81


def test_lattice_param():
    a = estimate_lattice_param(1.19, 2.20)  # Pb + I
    assert 6.0 < a < 7.0  # ~6.78


def test_build_pure_perovskite():
    struct = build_structure("CsPbI3")
    assert len(struct) == 5  # 1 Cs + 1 Pb + 3 I
    elements = sorted(set(str(s.specie) for s in struct))
    assert elements == ["Cs", "I", "Pb"]


def test_build_mixed_perovskite():
    struct = build_structure("CsSn0.5Ge0.5I3")
    assert len(struct) == 10  # 2x1x1 supercell
    elements = set(str(s.specie) for s in struct)
    assert "Sn" in elements
    assert "Ge" in elements


def test_build_unsupported_raises():
    with pytest.raises(ValueError, match="Only perovskite"):
        build_structure("NaCl")

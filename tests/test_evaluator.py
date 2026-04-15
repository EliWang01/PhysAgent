from pymatgen.core import Structure, Lattice
from physagent.agents.evaluator import (
    check_atomic_distances,
    check_formation_energy,
    check_forces_converged,
    check_targets,
    determine_verdict,
)


def _make_structure():
    lattice = Lattice.cubic(6.2)
    return Structure(
        lattice,
        ["Cs", "Pb", "I", "I", "I"],
        [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
    )


def test_atomic_distances_pass():
    result = check_atomic_distances(_make_structure())
    assert result["passed"] is True
    assert result["min_distance"] > 1.0


def test_formation_energy_pass():
    result = check_formation_energy({"formation_energy": -0.5})
    assert result["passed"] is True


def test_formation_energy_too_low():
    result = check_formation_energy({"formation_energy": -6.0})
    assert result["passed"] is False


def test_formation_energy_too_high():
    result = check_formation_energy({"formation_energy": 3.0})
    assert result["passed"] is False


def test_forces_converged_pass():
    result = check_forces_converged({"max_force": 0.01})
    assert result["passed"] is True


def test_forces_not_converged():
    result = check_forces_converged({"max_force": 0.1})
    assert result["passed"] is False


def test_targets_met():
    targets = {"formation_energy": {"max": 0}}
    props = {"formation_energy": -0.5}
    result = check_targets(targets, props)
    assert result["formation_energy"]["status"] == "MET"


def test_targets_not_met():
    targets = {"formation_energy": {"max": 0}}
    props = {"formation_energy": 0.5}
    result = check_targets(targets, props)
    assert result["formation_energy"]["status"] == "NOT_MET"


def test_targets_bandgap_proxy_tolerance():
    """Bandgap proxy gets ±0.3 eV tolerance."""
    targets = {"bandgap_proxy": {"min": 1.3, "max": 1.6}}
    props = {"bandgap_proxy": 1.05}  # 1.3 - 0.3 = 1.0, so 1.05 should pass
    result = check_targets(targets, props)
    assert result["bandgap_proxy"]["status"] == "MET"


def test_targets_skipped():
    targets = {"bandgap_proxy": {"min": 1.3}}
    props = {}
    result = check_targets(targets, props)
    assert result["bandgap_proxy"]["status"] == "SKIPPED"


def test_verdict_pass():
    physics = {"e": {"passed": True}}
    targets = {"f": {"status": "MET"}}
    assert determine_verdict(physics, targets) == "PASS"


def test_verdict_reject():
    physics = {"e": {"passed": False}}
    targets = {"f": {"status": "MET"}}
    assert determine_verdict(physics, targets) == "REJECT"


def test_verdict_revise():
    physics = {"e": {"passed": True}}
    targets = {"f": {"status": "NOT_MET"}}
    assert determine_verdict(physics, targets) == "REVISE"


def test_verdict_skipped_is_not_pass():
    """SKIPPED targets should not count as PASS."""
    physics = {"e": {"passed": True}}
    targets = {"f": {"status": "MET"}, "g": {"status": "SKIPPED"}}
    assert determine_verdict(physics, targets) == "REVISE"

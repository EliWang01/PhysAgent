from physagent.config import compute_score


def test_score_all_targets_met():
    targets = {"bandgap_proxy": {"min": 1.3, "max": 1.6}, "formation_energy": {"max": 0}}
    props = {"bandgap_proxy": 1.4, "formation_energy": -0.5}
    assert compute_score(targets, props) == 0.0


def test_score_bandgap_too_low():
    targets = {"bandgap_proxy": {"min": 1.3, "max": 1.6}}
    props = {"bandgap_proxy": 1.0}
    assert abs(compute_score(targets, props) - 0.3) < 1e-9  # 1.3 - 1.0


def test_score_bandgap_too_high():
    targets = {"bandgap_proxy": {"min": 1.3, "max": 1.6}}
    props = {"bandgap_proxy": 2.0}
    assert abs(compute_score(targets, props) - 0.4) < 1e-9  # 2.0 - 1.6


def test_score_multiple_misses():
    targets = {"bandgap_proxy": {"min": 1.3, "max": 1.6}, "formation_energy": {"max": 0}}
    props = {"bandgap_proxy": 1.0, "formation_energy": 0.5}
    assert abs(compute_score(targets, props) - 0.8) < 1e-9  # 0.3 + 0.5


def test_score_missing_property_ignored():
    targets = {"bandgap_proxy": {"min": 1.3, "max": 1.6}}
    props = {"formation_energy": -0.5}
    assert compute_score(targets, props) == float("inf")  # No target props evaluated


def test_score_none_property_ignored():
    targets = {"bandgap_proxy": {"min": 1.3, "max": 1.6}}
    props = {"bandgap_proxy": None}
    assert compute_score(targets, props) == float("inf")  # None counts as not evaluated


def test_score_empty_properties():
    targets = {"bandgap_proxy": {"min": 1.3}, "formation_energy": {"max": 0}}
    props = {}
    assert compute_score(targets, props) == float("inf")

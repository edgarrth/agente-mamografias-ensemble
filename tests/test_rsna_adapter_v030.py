from mammography_agent.datasets.rsna import (
    POLICY_ID,
    SELECTION_POLICY,
    canonical_view,
    deterministic_selection_key,
)


def test_rsna_canonical_view_mapping():
    assert canonical_view("L", "CC") == "L_CC"
    assert canonical_view("LEFT", "MLO") == "L_MLO"
    assert canonical_view("R", "CC") == "R_CC"
    assert canonical_view("RIGHT", "MLO") == "R_MLO"
    assert canonical_view("R", "AT") == ""


def test_duplicate_selection_key_is_deterministic_and_view_scoped():
    a = deterministic_selection_key("25", "R_CC", "822390278")
    b = deterministic_selection_key("25", "R_CC", "822390278")
    c = deterministic_selection_key("25", "R_CC", "1997933901")
    d = deterministic_selection_key("25", "R_MLO", "822390278")
    assert a == b
    assert a != c
    assert a != d
    assert len(a) == 64


def test_policy_ids_are_explicit():
    assert POLICY_ID == "RSNA_REQUIRED_FOUR_VIEWS_V1"
    assert SELECTION_POLICY == "DETERMINISTIC_LABEL_BLIND_SHA256_V1"

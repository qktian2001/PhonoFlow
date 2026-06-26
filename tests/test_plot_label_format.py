from phonoflow.plotting.plot_band import collapse_tick_labels, format_band_label


def test_gamma_label_formatting():
    assert format_band_label("G") == "Γ"
    assert format_band_label("GAMMA") == "Γ"
    assert format_band_label("Gamma") == "Γ"


def test_duplicate_labels_are_simplified():
    assert format_band_label("G|G") == "Γ"
    assert format_band_label("L|L") == "L"


def test_tick_collapse_keeps_distinct_boundary_labels():
    ticks, labels = collapse_tick_labels([0.0, 1.0, 1.0, 2.0], ["G", "X", "U", "K"])
    assert ticks == [0.0, 1.0, 2.0]
    assert labels == ["Γ", "X|U", "K"]

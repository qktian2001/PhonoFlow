from phonoflow.band.labels import collapse_tick_labels, format_band_label


def test_band_label_gamma_aliases_and_boundaries():
    cases = {
        "G": "Γ",
        "GAMMA": "Γ",
        "Gamma": "Γ",
        "\\Gamma": "Γ",
        "Γ": "Γ",
        "G|G": "Γ",
        "Γ|Γ": "Γ",
        "L|L": "L",
        "W|W": "W",
        "X|X": "X",
        "U|K": "U|K",
        "X|U": "X|U",
        "K|G": "K|Γ",
        "K|Γ": "K|Γ",
    }
    for raw, expected in cases.items():
        assert format_band_label(raw) == expected


def test_same_position_labels_merge_and_deduplicate():
    ticks, labels = collapse_tick_labels(
        [0.0, 1.0, 1.0, 2.0, 2.0],
        ["G", "X", "X", "U", "K"],
    )

    assert ticks == [0.0, 1.0, 2.0]
    assert labels == ["Γ", "X", "U|K"]

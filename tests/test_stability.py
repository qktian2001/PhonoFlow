from phonoflow.analysis.stability import analyze_stability


def test_stable_non_negative_frequencies():
    result = analyze_stability([0.0, 1.0, 2.0], imag_threshold=-0.1)
    assert result["dynamically_stable"] is True
    assert result["imaginary_mode_count"] == 0


def test_small_negative_frequency_is_tolerated():
    result = analyze_stability([-0.02, 1.0, 2.0], imag_threshold=-0.1)
    assert result["dynamically_stable"] is True
    assert result["imaginary_mode_count"] == 0


def test_large_negative_frequency_is_unstable():
    result = analyze_stability([-0.5, 1.0, 2.0], imag_threshold=-0.1)
    assert result["dynamically_stable"] is False
    assert result["imaginary_mode_count"] == 1

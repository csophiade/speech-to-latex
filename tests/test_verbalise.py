import pytest

from src.verbalise import flat_reading


@pytest.mark.parametrize("a,b", [
    (r"(x + 3)^2", r"x + 3^2"),
    (r"\sqrt{5-x}", r"\sqrt{5}-x"),
    (r"\frac{-1}{x-1}", r"-\frac{1}{x}-1"),
    (r"\frac{1}{y+4}", r"\frac{1}{y}+4"),
])
def test_grouping_variants_share_reading(a, b):
    assert flat_reading(a) == flat_reading(b) != ""


@pytest.mark.parametrize("latex,want", [
    ("x^2 + 1", "x squared plus 1"),
    ("17.124", "17.124"),
    (r"\bar{x}_{i}", "x bar sub i"),
    (r"y = \sqrt{5 - x}", "y equals square root of 5 minus x"),
])
def test_readings(latex, want):
    assert flat_reading(latex) == want


def test_lexical_differences_stay_distinct():
    assert flat_reading("x = 15") != flat_reading("x = 50")
    assert flat_reading(r"\bar{x}") != flat_reading("x")

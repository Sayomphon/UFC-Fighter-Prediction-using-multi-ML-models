import numpy as np

from ufc_prediction.evaluation.metrics import (
    american_odds_to_prob,
    expected_calibration_error,
    odds_implied_red_prob,
)


def test_american_odds_conversion():
    converted = american_odds_to_prob([-200, 200])
    np.testing.assert_allclose(converted, [2 / 3, 1 / 3])


def test_vig_removed_probabilities_are_normalized():
    red = odds_implied_red_prob([-150], [130])
    blue = odds_implied_red_prob([130], [-150])
    np.testing.assert_allclose(red + blue, [1.0])


def test_perfect_calibration_has_zero_error():
    assert expected_calibration_error([0, 1], [0, 1], n_bins=2) == 0.0

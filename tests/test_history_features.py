import pandas as pd

from ufc_prediction.features.history import ELO_START, build_engineered


def sample_fights():
    return pd.DataFrame(
        [
            {
                "Date": pd.Timestamp("2024-01-01"),
                "RedFighter": "A",
                "BlueFighter": "B",
                "Winner": "Red",
                "Finish": "KO/TKO",
            },
            {
                "Date": pd.Timestamp("2024-02-01"),
                "RedFighter": "A",
                "BlueFighter": "C",
                "Winner": "Blue",
                "Finish": "U-DEC",
            },
            {
                "Date": pd.Timestamp("2024-03-01"),
                "RedFighter": "B",
                "BlueFighter": "C",
                "Winner": "Red",
                "Finish": "SUB",
            },
        ]
    )


def test_debutants_start_with_neutral_pre_fight_state():
    engineered = build_engineered(sample_fights())
    first = engineered.iloc[0]
    assert first["RedElo"] == ELO_START
    assert first["BlueElo"] == ELO_START
    assert first["RedNFights"] == 0
    assert first["BlueNFights"] == 0


def test_future_fight_does_not_change_prior_features():
    fights = sample_fights()
    first_two = build_engineered(fights.iloc[:2]).iloc[:2]
    with_future = build_engineered(fights).iloc[:2]
    pd.testing.assert_frame_equal(first_two, with_future)

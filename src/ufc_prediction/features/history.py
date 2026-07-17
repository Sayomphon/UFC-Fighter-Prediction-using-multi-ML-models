# -*- coding: utf-8 -*-
"""UFC Feature Engineering — leakage-safe pre-fight features.

Adds history-derived features to the fight table by iterating fights in chronological
order. For every fight, the engineered values are read from each fighter's current
state BEFORE the fight, then the state is updated with the fight result. This guarantees
no future information leaks into a fight's own features.

Engineered columns
------------------
- RedElo / BlueElo / EloDif            : pre-fight Elo rating (start 1500, K=32) and difference
- Red/BlueRecentWinRate, RecentWinRateDif   : win rate over the last 5 fights (0.5 prior for debutants)
- Red/BlueRecentFinishRate             : share of last-5 wins that ended by KO/TKO or SUB
- Red/BlueDaysSinceLast, DaysSinceLastDif   : layoff in days since the fighter's previous bout
- Red/BlueNFights, NFightsDif          : number of prior fights in the dataset (experience)

NOTE ON VALUE: an ablation study (see the companion notebook / README) found these features
do NOT improve winner prediction once betting odds are present — odds already price in form
and matchup. They are kept as a reusable, leakage-safe module for no-odds scenarios, ensembles,
or future data without market lines. They are intentionally NOT wired into the main pipeline.
"""
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from ufc_prediction.paths import ENGINEERED_FIGHTS_PATH, RAW_FIGHTS_PATH

K_ELO = 32.0
ELO_START = 1500.0
RECENT_N = 5
FINISH_METHODS = {'KO/TKO', 'SUB'}

ENGINEERED_COLUMNS = [
    'RedElo', 'BlueElo', 'EloDif',
    'RedRecentWinRate', 'BlueRecentWinRate', 'RecentWinRateDif',
    'RedRecentFinishRate', 'BlueRecentFinishRate',
    'RedDaysSinceLast', 'BlueDaysSinceLast', 'DaysSinceLastDif',
    'RedNFights', 'BlueNFights', 'NFightsDif',
]


def build_engineered(df):
    """Return df sorted by Date with the engineered pre-fight columns appended.

    Requires columns: Date, RedFighter, BlueFighter, Winner ('Red'/'Blue'), Finish.
    """
    df = df.sort_values('Date').reset_index(drop=True)
    elo = defaultdict(lambda: ELO_START)
    results = defaultdict(list)    # fighter -> chronological [1 win / 0 loss]
    finishes = defaultdict(list)   # fighter -> [1 if won by finish else 0]
    last_date = {}
    cols = {c: [] for c in ENGINEERED_COLUMNS}

    def recent_rate(lst):
        window = lst[-RECENT_N:]
        return float(np.mean(window)) if window else 0.5  # neutral prior for debutants

    for _, r in df.iterrows():
        red, blue, date = r['RedFighter'], r['BlueFighter'], r['Date']
        red_elo, blue_elo = elo[red], elo[blue]

        # ----- pre-fight features (must not see this fight's outcome) -----
        cols['RedElo'].append(red_elo)
        cols['BlueElo'].append(blue_elo)
        cols['EloDif'].append(red_elo - blue_elo)

        rwr, bwr = recent_rate(results[red]), recent_rate(results[blue])
        cols['RedRecentWinRate'].append(rwr)
        cols['BlueRecentWinRate'].append(bwr)
        cols['RecentWinRateDif'].append(rwr - bwr)
        cols['RedRecentFinishRate'].append(recent_rate(finishes[red]))
        cols['BlueRecentFinishRate'].append(recent_rate(finishes[blue]))

        rl = (date - last_date[red]).days if red in last_date else np.nan
        bl = (date - last_date[blue]).days if blue in last_date else np.nan
        cols['RedDaysSinceLast'].append(rl)
        cols['BlueDaysSinceLast'].append(bl)
        cols['DaysSinceLastDif'].append((rl if rl == rl else 0) - (bl if bl == bl else 0))

        rn, bn = len(results[red]), len(results[blue])
        cols['RedNFights'].append(rn)
        cols['BlueNFights'].append(bn)
        cols['NFightsDif'].append(rn - bn)

        # ----- update state AFTER reading (uses this fight's outcome) -----
        red_won = 1 if r['Winner'] == 'Red' else 0
        exp_red = 1.0 / (1.0 + 10 ** ((blue_elo - red_elo) / 400.0))
        elo[red] = red_elo + K_ELO * (red_won - exp_red)
        elo[blue] = blue_elo + K_ELO * ((1 - red_won) - (1 - exp_red))

        finished = 1 if r.get('Finish') in FINISH_METHODS else 0
        results[red].append(red_won)
        results[blue].append(1 - red_won)
        finishes[red].append(finished if red_won else 0)
        finishes[blue].append(finished if not red_won else 0)
        last_date[red] = date
        last_date[blue] = date

    eng = pd.DataFrame(cols, index=df.index)
    return pd.concat([df, eng], axis=1)


def generate_engineered_dataset(
    source_path: Path = RAW_FIGHTS_PATH,
    output_path: Path = ENGINEERED_FIGHTS_PATH,
) -> Path:
    """Generate the canonical processed dataset and return its output path."""

    src = pd.read_csv(source_path)
    src['Date'] = pd.to_datetime(src['Date'])
    src = src[src['Finish'] != 'Overturned'].reset_index(drop=True)
    out = build_engineered(src)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(
        f'Wrote {output_path} with {len(ENGINEERED_COLUMNS)} engineered columns, '
        f'{len(out)} rows.'
    )
    return output_path


if __name__ == '__main__':
    generate_engineered_dataset()

# Feature Engineering — Leakage-Safe Pre-Fight Features

This module adds history-derived features to the fight table and reports, honestly,
whether they improve winner prediction.

## What it builds

[`history.py`](../src/ufc_prediction/features/history.py) → `build_engineered(df)` iterates fights in
chronological order and, for each fight, reads each fighter's state **before** the fight, then updates
that state with the result. No future information can leak into a fight's own features.

| Column(s) | Meaning |
|---|---|
| `RedElo`, `BlueElo`, `EloDif` | Pre-fight Elo rating (start 1500, K=32) and difference |
| `Red/BlueRecentWinRate`, `RecentWinRateDif` | Win rate over the last 5 fights (0.5 prior for debutants) |
| `Red/BlueRecentFinishRate` | Share of recent wins ending by KO/TKO or SUB |
| `Red/BlueDaysSinceLast`, `DaysSinceLastDif` | Layoff in days since the previous bout |
| `Red/BlueNFights`, `NFightsDif` | Number of prior fights (experience) |

Run `python scripts/build_features.py` from the repo root to regenerate
`data/processed/ufc-master-fe.csv`. Sanity check: elite fighters end well above 1500
(e.g. Jon Jones ≈ 1720).

## Does it help? — Ablation result (CatBoost winner model)

**No, not once betting odds are present.** Tested with the project's evaluation methodology
(train ≤ 2024-03-31, leakage-safe, train-only imputation).

Single holdout (test = 2024-04+, 326 fights):

| Feature set | Accuracy | Log loss |
|---|---|---|
| baseline 40 features | 0.696 | 0.595 |
| baseline 40 + engineered | 0.669 | 0.604 |
| engineered only (12) | 0.549 | 0.690 |

Rolling-origin CV (2020–2024), baseline-40 vs +8 RFE-kept engineered:

| | Accuracy | Log loss | Beats baseline |
|---|---|---|---|
| baseline 40 | 0.6549 | 0.6112 | — |
| + engineered | 0.6584 | 0.6124 | acc +0.0035 (2/5 yrs), **log loss worse** |
| favorite-pick baseline | 0.6694 | — | beats both |

## Why odds dominate

The Elo/form/layoff signal is **already priced into the betting odds** (`RedOdds`, `BlueOdds`, …),
which the main 40-feature set includes. Recomputing it from fight history is redundant and adds
noise — the accuracy gain is within run-to-run variance and **log loss (the metric that matters for
probability outputs) gets worse**. Even with odds removed, the existing per-fighter stats (significant
strikes, takedowns, streaks, reach) already cover most of what Elo captures.

## Why it's kept anyway

The module is intentionally **not** wired into the main pipeline (that would degrade the cleaned-up
models). It is retained because it is reusable and leakage-safe, and is valuable for:

- **No-odds scenarios** — predicting fights before betting lines open, or datasets without market lines.
- **Ensembling / research** — Elo as an independent signal to blend or analyze.
- **Evidence** — it empirically establishes that betting odds are the ceiling for history-based features,
  so future effort should target signals the market misses (e.g. stylistic matchup interactions), not
  more history aggregates.

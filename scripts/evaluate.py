# -*- coding: utf-8 -*-
"""UFC_Fighter_prediction_evaluation.ipynb

#**UFC Fight Outcome Prediction — Evaluation Framework**

This notebook is the standard evaluation harness for every model in this project. It answers three questions that plain accuracy cannot:

1. **Is the model better than doing nothing?** — compares against naive baselines (always pick Red, pick the betting favorite) and against the **sportsbook's own implied probabilities** (vig removed).
2. **Are the predicted probabilities trustworthy?** — log loss, Brier score, ROC-AUC, Expected Calibration Error (ECE) and reliability diagrams, since the project outputs *probabilities*, not just labels.
3. **Is performance stable over time?** — rolling-origin cross-validation (train up to year X, test on year X+1) instead of trusting a single train/test split.

All data preparation here is **leakage-free**: post-fight columns are never used as features, and imputation statistics are computed from the training period only (re-computed per fold in the rolling CV).
"""

"""##**Step 1: Install and Import Libraries**
"""

# Install the project first with: pip install -e .
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import accuracy_score, classification_report, f1_score, log_loss, roc_auc_score
from catboost import CatBoostClassifier

from ufc_prediction.evaluation.metrics import (
    evaluate_binary,
    multiclass_metrics,
    odds_implied_red_prob,
    red_probability,
)
from ufc_prediction.paths import (
    METRICS_DIR,
    PLOTS_DIR,
    RAW_FIGHTS_PATH,
    SELECTED_FEATURES_PATH,
    ensure_artifact_directories,
)

try:
    from IPython.display import display
except ImportError:
    display = print

ensure_artifact_directories()

DATA_PATH = RAW_FIGHTS_PATH
FEATURES_PATH = SELECTED_FEATURES_PATH

HOLDOUT_TRAIN_END = '2024-03-31'  # same split as the rest of the project

# Columns only known AFTER the fight ends — never used as features
POST_FIGHT_COLS = ['Winner', 'Finish', 'FinishDetails', 'FinishRound', 'FinishRoundTime', 'TotalFightTimeSecs']
# Non-feature columns dropped from every method matrix (fighter names don't generalize to unseen fighters;
# HasFinish is an internal flag; Date is the split key)
NON_FEATURE_COLS = ['Date', 'RedFighter', 'BlueFighter', 'HasFinish'] + POST_FIGHT_COLS

CATEGORICAL_FEATURES = ['Location', 'Country', 'WeightClass',
                        'TitleBout', 'Gender', 'Winner', 'BlueStance', 'RedStance', 'BetterRank', 'Finish']

"""##**Step 2: Leakage-Free Data Preparation**

`prepare_data(train_cutoff)` returns an imputed and encoded copy of the dataset where **all imputation statistics come from fights on/before the cutoff date**. The rolling CV in Step 8 calls it once per fold, so no fold ever sees statistics from its own future.
"""

raw_df = pd.read_csv(DATA_PATH)
raw_df['Date'] = pd.to_datetime(raw_df['Date'])
features_to_use = pd.read_csv(FEATURES_PATH)['Selected Features'].tolist()

def prepare_data(train_cutoff):
    # Impute with training-period statistics only, then label-encode categoricals
    df = raw_df.copy()
    df = df[df['Finish'] != 'Overturned']            # post-hoc reversals (e.g. failed tests), not normal outcomes
    df['HasFinish'] = df['Finish'].notna()           # method target excludes No Contest / missing finishes
    # Reframed targets from RAW labels (built before imputation/encoding; excluded from impute below)
    df['Method3'] = df['Finish'].map({'KO/TKO': 'KO/TKO', 'SUB': 'SUB', 'U-DEC': 'DEC', 'S-DEC': 'DEC', 'M-DEC': 'DEC'})
    df['IsFinishTgt'] = df['Finish'].isin(['KO/TKO', 'SUB'])
    df['RoundBin'] = df['FinishRound'].map(lambda r: 'R1' if r == 1 else 'R2' if r == 2 else 'R3+' if (pd.notna(r) and r >= 3) else None)
    # NOTE: index is intentionally NOT reset so raw_df.loc[test_idx] still aligns (test period has no Overturned)
    train_mask = df['Date'] <= pd.to_datetime(train_cutoff)

    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    # Rank columns: NaN means UNRANKED, not missing -> sentinel just past the worst ranked slot (ranks 0..15)
    rank_cols = [c for c in df.columns if 'Rank' in c and df[c].dtype != object]
    df[rank_cols] = df[rank_cols].fillna(16)
    df[numerical_cols] = df[numerical_cols].fillna(df.loc[train_mask, numerical_cols].mean())
    for col in categorical_cols:
        if col in ('Method3', 'RoundBin'):
            continue
        df[col] = df[col].fillna(df.loc[train_mask, col].mode()[0])

    encoders = {}
    for col in CATEGORICAL_FEATURES:
        encoders[col] = LabelEncoder()
        df[col] = encoders[col].fit_transform(df[col])
    return df, encoders

df, encoders = prepare_data(HOLDOUT_TRAIN_END)
RED = list(encoders['Winner'].classes_).index('Red')  # encoded label for a Red-corner win
print('Winner classes:', list(encoders['Winner'].classes_))
print('Finish classes:', list(encoders['Finish'].classes_))

"""##**Step 3: Baselines and Odds-Implied Probabilities**

A model is only useful if it beats what you get for free:

- **Always Red** — the red corner wins ~58% of historical fights (the UFC usually gives the favorite the red corner).
- **Favorite pick** — predict whoever has the better (more negative) American odds.
- **Sportsbook implied probability** — convert both fighters' American odds to probabilities and normalize them so they sum to 1 (this removes the bookmaker margin, the *vig*). This is the strongest available benchmark for both accuracy and log loss.
"""

"""##**Step 4: Metric Helpers**

For winner prediction (binary, probability output) we report:

- **Accuracy** — fraction of correct picks at the 0.5 threshold.
- **Log loss** — penalizes confident wrong probabilities; the primary metric for probability quality (lower is better).
- **Brier score** — mean squared error of the probabilities (lower is better).
- **ROC-AUC** — ranking quality independent of threshold.
- **ECE (Expected Calibration Error)** — average gap between predicted probability and observed win frequency: when the model says 70%, does Red win ~70% of those fights?
"""

"""##**Step 5: Winner — Holdout Evaluation**

Train on fights up to 2024-03-31, test on fights from 2024-04-01. Models compared:

- Logistic Regression (simple reference model — if boosting barely beats it, most of the signal is already in the odds)
- Random Forest
- CatBoost with the Main-notebook hyperparameters
- CatBoost + sigmoid calibration (`CalibratedClassifierCV`) — same model, calibrated probabilities

Two views: the **full test set**, and the subset of fights that have real (non-imputed) betting odds, where the sportsbook itself can be scored as a competitor.
"""

df_train = df[df['Date'] <= pd.to_datetime(HOLDOUT_TRAIN_END)]
df_test = df[df['Date'] > pd.to_datetime(HOLDOUT_TRAIN_END)]

X_train, X_test = df_train[features_to_use], df_test[features_to_use]
y_train, y_test = df_train['Winner'].values, df_test['Winner'].values
y_test_red = (y_test == RED).astype(int)

scaler = StandardScaler().fit(X_train)
X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

winner_models = {
    'Logistic Regression': LogisticRegression(max_iter=2000),
    'Random Forest': RandomForestClassifier(max_depth=10, random_state=42),
    'CatBoost (Main params)': CatBoostClassifier(depth=10, learning_rate=0.01, iterations=1000, verbose=0),
    'CatBoost + sigmoid calibration': CalibratedClassifierCV(
        CatBoostClassifier(depth=10, learning_rate=0.01, iterations=1000, verbose=0),
        method='sigmoid', cv=3),
}

test_probs = {}
for name, model in winner_models.items():
    model.fit(X_train_s, y_train)
    test_probs[name] = red_probability(model, X_test_s, RED)
    print(f'trained: {name}')

# Constant baseline: predict Red with the training-period Red win rate as the probability
p_red_const = np.full(len(y_test), (y_train == RED).mean())

rows = [evaluate_binary('Baseline: always Red (train rate)', y_test_red, p_red_const)]
for name, p in test_probs.items():
    rows.append(evaluate_binary(name, y_test_red, p))
results_full = pd.DataFrame(rows).set_index('Model').round(4)

print('=== Winner — full test set ===')
display(results_full)
results_full.to_csv(METRICS_DIR / 'evaluation_winner_full_test.csv')

# Apples-to-apples vs the sportsbook on fights that have real odds
test_idx = df_test.index
odds_mask = raw_df.loc[test_idx, ['RedOdds', 'BlueOdds']].notna().all(axis=1).values
y_odds = y_test_red[odds_mask]

p_book = odds_implied_red_prob(raw_df.loc[test_idx[odds_mask], 'RedOdds'],
                               raw_df.loc[test_idx[odds_mask], 'BlueOdds'])

rows = [evaluate_binary('Sportsbook odds (vig removed)', y_odds, p_book)]
for name, p in test_probs.items():
    rows.append(evaluate_binary(name, y_odds, p[odds_mask]))
results_odds = pd.DataFrame(rows).set_index('Model').round(4)

print('=== Winner — fights with real betting odds ===')
display(results_odds)
results_odds.to_csv(METRICS_DIR / 'evaluation_winner_vs_odds.csv')

"""##**Step 6: Calibration Curve (Reliability Diagram)**

Each point bins fights by predicted probability and plots the observed Red win rate in that bin. A trustworthy model lies on the diagonal. Points above the diagonal = underconfident; below = overconfident.
"""

plt.figure(figsize=(7, 7))
plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')

curves = {'Sportsbook odds (vig removed)': p_book,
          'CatBoost (Main params)': test_probs['CatBoost (Main params)'][odds_mask],
          'CatBoost + sigmoid calibration': test_probs['CatBoost + sigmoid calibration'][odds_mask]}
for name, pp in curves.items():
    frac_pos, mean_pred = calibration_curve(y_odds, pp, n_bins=10, strategy='quantile')
    plt.plot(mean_pred, frac_pos, marker='o', label=name)

plt.xlabel('Predicted probability (Red wins)')
plt.ylabel('Observed frequency (Red wins)')
plt.title('Winner calibration — test set (fights with real odds)')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(PLOTS_DIR / 'calibration_winner.png', dpi=150, bbox_inches='tight')
plt.show()

"""##**Step 7: Method & Round — Holdout Evaluation**

Both targets are **reframed** for meaning and balance, and compared against a class-prior baseline (`DummyClassifier`). Accuracy sits next to **macro-F1** — the metric that matters under imbalance.

- **Method → 3 classes**: KO/TKO, SUB, DEC (U/S/M-DEC merged); DQ and missing-finish bouts are dropped. With balanced class weights every class becomes predictable, unlike the old 7-class target where S-DEC/M-DEC/DQ scored 0.
- **Round → two stages**: stage 1 predicts *finish (KO/SUB) vs decision* (a balanced, meaningful target — the old single-stage round was dominated by decisions landing on "round 3"); stage 2 predicts *which round (R1/R2/R3+)* among finishes only.
"""

# Reframed targets built from RAW labels (df_train/df_test come from prepare_data, which keeps Finish text)
def _method3(f):
    return 'KO/TKO' if f == 'KO/TKO' else 'SUB' if f == 'SUB' else 'DEC' if f in ('U-DEC', 'S-DEC', 'M-DEC') else None
def _roundbin(r):
    return 'R1' if r == 1 else 'R2' if r == 2 else 'R3+' if (pd.notna(r) and r >= 3) else None

# All tasks share the 40 selected features; tasks differ only in which rows they use
Xtr_all = StandardScaler().fit(df_train[features_to_use])
Xtr_s = Xtr_all.transform(df_train[features_to_use])
Xte_s = Xtr_all.transform(df_test[features_to_use])

# ---- Method (3-class: KO/TKO, SUB, DEC) ----
m_tr = df_train['HasFinish'].values & df_train['Method3'].notna().values
m_te = df_test['HasFinish'].values & df_test['Method3'].notna().values
ytr_m = df_train['Method3'].values[m_tr]
yte_m = df_test['Method3'].values[m_te]

method_models = {
    'Baseline: class prior': DummyClassifier(strategy='prior'),
    'CatBoost (+balanced weights)': CatBoostClassifier(depth=8, learning_rate=0.03, iterations=600,
                                                       verbose=0, auto_class_weights='Balanced'),
}
rows = []
for name, model in method_models.items():
    model.fit(Xtr_s[m_tr], ytr_m)
    rows.append(multiclass_metrics(name, model, Xte_s[m_te], yte_m))
results_method = pd.DataFrame(rows).set_index('Model').round(4)
print('=== Method of victory (3-class) ===')
display(results_method)
results_method.to_csv(METRICS_DIR / 'evaluation_method.csv')

best_m = method_models['CatBoost (+balanced weights)']
yp_m = best_m.classes_[np.argmax(best_m.predict_proba(Xte_s[m_te]), axis=1)]
print(classification_report(yte_m, yp_m, zero_division=0))

# ---- Round stage 1: finish (KO/SUB) vs decision ----
ytr_f = df_train['IsFinishTgt'].astype(int).values[m_tr]
yte_f = df_test['IsFinishTgt'].astype(int).values[m_te]
fm = CatBoostClassifier(depth=8, learning_rate=0.03, iterations=600, verbose=0, auto_class_weights='Balanced')
fm.fit(Xtr_s[m_tr], ytr_f)
pf = fm.predict_proba(Xte_s[m_te])[:, 1]; ypf = (pf >= 0.5).astype(int)
prior_f = DummyClassifier(strategy='prior').fit(Xtr_s[m_tr], ytr_f)
results_round1 = pd.DataFrame([
    {'Model': 'Baseline: class prior', 'Accuracy': accuracy_score(yte_f, prior_f.predict(Xte_s[m_te])),
     'Macro F1': f1_score(yte_f, prior_f.predict(Xte_s[m_te]), average='macro', zero_division=0), 'ROC-AUC': np.nan},
    {'Model': 'CatBoost (+balanced weights)', 'Accuracy': accuracy_score(yte_f, ypf),
     'Macro F1': f1_score(yte_f, ypf, average='macro'), 'ROC-AUC': roc_auc_score(yte_f, pf)},
]).set_index('Model').round(4)
print('=== Round stage 1: finish vs decision ===')
display(results_round1)
results_round1.to_csv(METRICS_DIR / 'evaluation_round_stage1.csv')

# ---- Round stage 2: which round (R1/R2/R3+) among finished bouts with a known round ----
r_tr = df_train['IsFinishTgt'].values & df_train['RoundBin'].notna().values
r_te = df_test['IsFinishTgt'].values & df_test['RoundBin'].notna().values
ytr_r = df_train['RoundBin'].values[r_tr]
yte_r = df_test['RoundBin'].values[r_te]
rmodels = {
    'Baseline: class prior': DummyClassifier(strategy='prior'),
    'CatBoost (+balanced weights)': CatBoostClassifier(depth=8, learning_rate=0.03, iterations=600,
                                                       verbose=0, auto_class_weights='Balanced'),
}
rows = []
for name, model in rmodels.items():
    model.fit(Xtr_s[r_tr], ytr_r)
    rows.append(multiclass_metrics(name, model, Xte_s[r_te], yte_r))
results_round2 = pd.DataFrame(rows).set_index('Model').round(4)
print('=== Round stage 2: round | finish (R1/R2/R3+) ===')
display(results_round2)
results_round2.to_csv(METRICS_DIR / 'evaluation_round_stage2.csv')

"""##**Step 8: Rolling-Origin Cross-Validation (Winner)**

A single train/test split can flatter (or punish) a model by luck. Here the model is retrained once per fold — train on everything up to Dec 31 of year X-1, test on year X — for 2020 through 2024. Imputation is recomputed inside every fold, so no fold uses future statistics.

Read the result as: *does the model beat the favorite-pick baseline consistently, or only in one lucky year?*
"""

cv_rows = []
for year in [2020, 2021, 2022, 2023, 2024]:
    cutoff = pd.to_datetime(f'{year - 1}-12-31')
    df_f, enc_f = prepare_data(cutoff)
    red_f = list(enc_f['Winner'].classes_).index('Red')

    tr = df_f[df_f['Date'] <= cutoff]
    te = df_f[(df_f['Date'] > cutoff) & (df_f['Date'] <= pd.to_datetime(f'{year}-12-31'))]

    sc = StandardScaler().fit(tr[features_to_use])
    Xtr, Xte = sc.transform(tr[features_to_use]), sc.transform(te[features_to_use])
    ytr = tr['Winner'].values
    yte_red = (te['Winner'].values == red_f).astype(int)

    model = CatBoostClassifier(depth=10, learning_rate=0.01, iterations=1000, verbose=0)
    model.fit(Xtr, ytr)
    p = red_probability(model, Xte, red_f)

    om = raw_df.loc[te.index, ['RedOdds', 'BlueOdds']].notna().all(axis=1).values
    p_bk = odds_implied_red_prob(raw_df.loc[te.index[om], 'RedOdds'],
                                 raw_df.loc[te.index[om], 'BlueOdds'])

    cv_rows.append({
        'Test year': year,
        'n': len(yte_red),
        'n (odds)': int(om.sum()),
        'Model Acc': accuracy_score(yte_red, (p >= 0.5).astype(int)),
        'Model LogLoss': log_loss(yte_red, np.column_stack([1 - p, p]), labels=[0, 1]),
        'Model Acc (odds subset)': accuracy_score(yte_red[om], (p[om] >= 0.5).astype(int)),
        'Favorite Acc': accuracy_score(yte_red[om], (p_bk >= 0.5).astype(int)),
        'Odds LogLoss': log_loss(yte_red[om], np.column_stack([1 - p_bk, p_bk]), labels=[0, 1]),
    })
    print(f'fold {year}: done (n={len(yte_red)})')

cv_df = pd.DataFrame(cv_rows).set_index('Test year').round(4)
cv_df.loc['Mean'] = cv_df.mean(numeric_only=True).round(4)

print('=== Rolling-origin CV — winner (CatBoost vs sportsbook) ===')
display(cv_df)
cv_df.to_csv(METRICS_DIR / 'evaluation_rolling_cv_winner.csv')

"""##**Step 9: Summary**

Artifacts written to `artifacts/metrics` and `artifacts/plots`:

- `evaluation_winner_full_test.csv` — winner metrics on the full holdout test set
- `evaluation_winner_vs_odds.csv` — winner metrics vs the sportsbook (odds subset)
- `evaluation_method.csv` — method metrics vs the class-prior baseline
- `evaluation_round_stage1.csv`, `evaluation_round_stage2.csv` — two-stage round metrics
- `evaluation_rolling_cv_winner.csv` — per-year stability check
- `calibration_winner.png` — reliability diagram

How to read the results:

- A winner model is **useful** only if it beats *Favorite Acc* on accuracy and *Odds LogLoss* on log loss — consistently across folds, not just in one year.
- A method/round model is **useful** only if it beats the class-prior baseline on **macro-F1**, not just accuracy.
- Low **ECE** (and a curve near the diagonal) means the predicted probabilities can be taken at face value.
"""

print('Evaluation complete. Artifacts saved:')
for artifact in [
    METRICS_DIR / 'evaluation_winner_full_test.csv',
    METRICS_DIR / 'evaluation_winner_vs_odds.csv',
    METRICS_DIR / 'evaluation_method.csv',
    METRICS_DIR / 'evaluation_round_stage1.csv',
    METRICS_DIR / 'evaluation_round_stage2.csv',
    METRICS_DIR / 'evaluation_rolling_cv_winner.csv',
    PLOTS_DIR / 'calibration_winner.png',
]:
    print(' -', artifact.relative_to(PROJECT_ROOT), '(ok)' if artifact.exists() else '(MISSING)')

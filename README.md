# How Much Is One Audit Worth? Verifying What Federated Clients Report

Reproduction code and data for the paper. Every number, table, and figure in
the experiments section (Section VII), and the experimental statements in
Sections I, IV, V, VI, VIII and the appendices, is recomputed from cached
per-record evaluation scores shipped in `data/`, and compared automatically
with the value printed in the paper. No model training and no GPU are needed.

## Quick start

Requires Python 3.9 or newer.

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt          # numpy, scipy
PYTHONPATH=src python -m auditworth.verify_data   # check data integrity
python reproduce.py                      # all results; prints PASS/FAIL per paper number
```

`python reproduce.py --list` lists the scripts; `python reproduce.py table2 fig4`
runs only scripts whose names contain the given strings. Each script writes its
recomputed values to `results/<script>.json` (figure data to `results/fig*/`)
and compares every value the paper reports against `expected/<script>.json`,
which records the printed value, the tolerance implied by its printed
precision, and where it appears in the paper.

Runtime: the full run takes about 30-40 minutes on a 10-core laptop, dominated
by the numerical best-response replays (`vii_b_decision_stake`,
`vii_b_settings_stake`, `vii_b_separation_stake`); everything else takes seconds.
`AUDITWORTH_WORKERS` sets the number of processes used by the decision-stake
replays (default: all cores), and `AUDITWORTH_QUICK=1` restricts them to two
groups for a smoke test (their checks then fail, as expected).

## What each script reproduces

| Script | Paper item |
|---|---|
| `table1_curation` | Table I: credit gain, flip rate, and target effect of credit-seeking curation in ten studies |
| `vii_a_credit_curation` | Section VII-A and Section I: credit rose in all ten primary contrasts; panel-over-target excess grew in all eight evaluable ones |
| `iv_payment_slope` | Section IV: payment change for distorting a report to the accuracy of 16 direct reads |
| `v_dimension_bound` | Section V: the dimension bound is 9.5 to 37 times (median 24) the random-panel error |
| `vi_federation_size` | Section VI: halving the regret of a random choice needs N·eps_dec of about 256 |
| `vii_b_panel_fallback` | Section VII-B: payment-seeking panels versus direct reads and random panels |
| `table2_decision` | Table II: reports within 16 direct draws under the decision stake; regret from reports or one audit per client |
| `vii_b_decision_stake` | Section VII-B, "The decision stake" (and Appendix D) |
| `vii_b_decision_choice` | Section VII-B, "Does report accuracy improve the choice?" |
| `vii_b_settings_leverage` | Section VII-B, "Two other task settings": UCI HAR and Stack Overflow leverage and choice gain |
| `vii_b_settings_stake` | Section VII-B: the decision stake on UCI HAR and Stack Overflow |
| `vii_b_separation_value` | Section VII-B, "Separation within FEMNIST": decision gain as candidate updates are scaled apart |
| `vii_b_separation_stake` | Section VII-B, "Separation within FEMNIST": own-update decision stake |
| `vii_b_strong_textmodel` | Section VII-B, "A stronger text model" |
| `fig2_separation_data` | Figure 2 data (decision gain vs candidate range) |
| `vii_c_credit_from_submission` | Section VII-C and Figure 3 data (credit computed from the submission) |
| `fig4_staleness` | Figure 4 and Section VII-D (panel staleness over training rounds) |
| `vii_d_kernel_reuse` | Section VII-D: richer elicitation kernels at rounds 120 and 160 |
| `appb_participation_ratio` | Appendix B: participation-ratio prediction |
| `viii_payment_exposure` | Section VIII: expected net payment to a truthful FEMNIST client |
| `viii_exposure_crosssetting` | Section VIII: expected net payment in UCI HAR, Stack Overflow, and scaled FEMNIST |

## Repository layout

```
data/            cached per-record scores and derived per-state rows (see data/README.md)
src/auditworth/  loaders (data.py), result checking (check.py), and the replay code:
                 leverage.py (panels, credit stake), decision.py (private choice, best
                 responses, reports vs audits), crosssetting.py (UCI HAR, Stack Overflow,
                 update scaling), staleness.py (panel decay, kernels), curation.py (Table I)
scripts/         one script per paper item (table above)
expected/        values printed in the paper, with tolerances
training/        optional: code that regenerates the caches in data/ from the public datasets
```

## Data and training

`data/README.md` describes every file. The caches contain only per-record
correctness bits of candidate models, per-record Shapley values, and record
indices into the public datasets; no raw records are redistributed, except
7x7 average-pooled FEMNIST images and model activations used for one kernel
comparison. The replays use the evaluation seeds, groups, and optimizer
settings of the original analyses; groups reserved for debugging (FEMNIST
groups 0-3) are included in the data but excluded from all reported results.

`training/README.md` explains how to regenerate the caches from the public
datasets (FEMNIST via LEAF/Flower, UCI HAR, the Stack Overflow federated
dataset) and how to compare regenerated files with `data/`. This step is
optional and needs substantially more compute.

## Datasets and licenses

The code is released under the MIT license (`LICENSE`). The derived data in
`data/` inherit the terms of their source datasets: FEMNIST (LEAF benchmark,
derived from NIST Special Database 19), UCI Human Activity Recognition Using
Smartphones (CC BY 4.0), WISDM (used only in Table I rows), and the Stack
Overflow federated dataset (CC BY-SA 3.0).

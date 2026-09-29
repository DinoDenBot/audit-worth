# How Much Is One Audit Worth? Verifying What Federated Clients Report

Reproduction code and data for the paper. Every number, table, and figure in
the experiments section (Section VII), and the experimental statements in
Sections I, IV, VI, VIII and Appendices B, D, and E, is recomputed from cached
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

`python reproduce.py --list` lists the scripts; `python reproduce.py decision_table fig4`
runs only scripts whose names contain the given strings. Each script writes its
recomputed values to `results/<script>.json` (figure data to `results/fig*/`)
and compares every value the paper reports against `expected/<script>.json`,
which records the printed value, the tolerance implied by its printed
precision, and where it appears in the paper.

Runtime: the full run takes about 30-40 minutes on a 10-core laptop, dominated
by the numerical best-response replays (`decision_stake`,
`settings_stake`, `separation_stake`); everything else takes seconds.
`AUDITWORTH_WORKERS` sets the number of processes used by the decision-stake
replays (default: all cores), and `AUDITWORTH_QUICK=1` restricts them to two
groups for a smoke test (their checks then fail, as expected).

## What each script reproduces

| Script | Paper item |
|---|---|
| `curation_summary` | Section VII-A and Section I: scores rose in all ten primary contrasts; panel-over-target excess grew in all eight evaluable ones. |
| `curation_table` | Table II: contribution score gain, flip rate, and target effect (95% interval) of score-seeking curation in ten studies. |
| `payment_slope` | Section IV and Section I: payment change for degrading a report to the 16-draw reference, and the cost of preparing a report. |
| `federation_size` | Section VI, 'Federation size', and Appendix D: halving the regret of a random choice needs N*eps_dec of about 256. |
| `decision_stake` | Section VII-B, 'What payment does the decision stake require?', and Appendix D (replicated federations; box responses outside M_dec). |
| `decision_table` | Table III: reports within the 16-draw reference under the decision stake (left) and regret of the choice from reports, one audit per client, or at random (right). |
| `settings_stake` | Section VII-B: the decision stake on UCI HAR and Stack Overflow. |
| `separation_stake` | Section VII-B: own-update decision stake in the scaled FEMNIST setup. |
| `decision_choice` | Section VII-C, 'Does report accuracy change the private choice?': FEMNIST choice from exact reports versus 1, 4, or 16 audits per client. |
| `settings_har_stackoverflow` | Section VII-C and VII-D: UCI HAR and Stack Overflow choice gains and panel leverage. |
| `separation_value` | Section VII-C: candidate ranges and the gain of exact reports as FEMNIST candidate updates are scaled apart. |
| `top_gap` | Table I and Section VI: mean top gap (best minus second-best action) and range of each replay setting, and the size of theta*Delta at eps_dec = 4. |
| `fig2_separation_data` | Figure 2 and Section VII-C: data for the separation figure (decision gain of exact reports vs candidate range, eps_dec = 1 and 4). |
| `contribution score_from_submission` | Section VII-D and Figure 3: contribution score computed from the submission (report versus panel as the payment ratio falls). |
| `panel_leverage` | Section VII-D, Appendix D, and Appendix E: payment-seeking panels versus random draws and random panels (FEMNIST). |
| `payment_exposure_femnist` | Section VIII: expected net payment to a truthful FEMNIST client at beta/U = 300 (median and 90th percentile, in units of U). |
| `payment_exposure_crosssetting` | Section VIII: expected net payment to a truthful client at beta/U = 300 in UCI HAR, Stack Overflow, and scaled FEMNIST. |
| `panel_dimension_bound` | Appendix B: the dimension bound is 9.5 to 37 times (median 24) the random-panel error. |
| `participation_ratio` | Appendix B: the frozen participation-ratio prediction failed in the opposite direction (Spearman -0.29 [-0.47, -0.09]). |
| `stronger_text_model` | Appendix D, 'Stronger text-model sensitivity': Stack Overflow replay with a 120-round sparse character-feature model. |
| `fig4_staleness` | Figure 4 and Appendix E: how long a round-80 payment-seeking 16-record panel stays valid. |
| `staleness_kernels` | Appendix E: richer elicitation kernels do not keep round-80 panels accurate at rounds 120 and 160. |

## Repository layout

```
data/            cached per-record scores and derived per-state rows (see data/README.md)
src/auditworth/  loaders (data.py), result checking (check.py), and the replay code:
                 leverage.py (panels, contribution score stake), decision.py (private choice, best
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

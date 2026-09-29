# Regenerating the cached scores (optional)

The paper's numbers are reproduced by `python reproduce.py` from the cached
per-record scores shipped in `data/`; nothing in this directory is needed for
that. This directory contains the training and scoring code that produced
those caches from the public datasets, so that the caches themselves can be
checked. Every generator writes into an output directory (default
`training/output/`, which is git-ignored) and refuses to write inside `data/`.
`training/compare.py` then compares the regenerated files with `data/`, array
by array.

| Cache in `data/` | Generator | Dataset | Framework |
|---|---|---|---|
| `femnist30/state/` | `femnist30/prepare.py`, `train.py`, `score.py` | FEMNIST (Flower Parquet) | PyTorch |
| `femnist30/decay/` | `femnist30/decay.py` (after `train.py`) | FEMNIST | PyTorch |
| `femnist_scaled/state/` | `femnist_scaled/prepare.py`, `train.py`, `score.py` | FEMNIST | PyTorch |
| `har/state/` | `har/har.py` | UCI HAR | scikit-learn |
| `stackoverflow/` | `stackoverflow/prepare.py` (scikit-learn), `train.py`, `score.py` (PyTorch) | Stack Overflow (FedJAX SQLite) | both |
| `stackoverflow_strong/` | `stackoverflow_strong/run.py` | Stack Overflow | scikit-learn |

The writer and user assignments are fixed in `femnist30/writer-manifest.json`,
`femnist_scaled/client-manifest.json` and `stackoverflow/client-manifest.json`
(public dataset identifiers only). The preparation scripts check the record
and role counts against these files. The quoted string constants in the code
(for example the arguments of `seed_for` and `intseed`) are fixed labels for
the random streams: changing them changes every result.

## Environments

Two environments were used, and both are CPU-only:

```bash
# PyTorch generators (Python 3.12)
python3.12 -m venv .venv-torch && .venv-torch/bin/pip install -r training/requirements.txt
# scikit-learn generators (Python 3.11)
python3.11 -m venv .venv-sk && .venv-sk/bin/pip install -r training/requirements-sklearn.txt
```

The HAR and stronger Stack Overflow generators and the Stack Overflow feature
extraction need `.venv-sk`. The FEMNIST generators and Stack Overflow
training/scoring need `.venv-torch`. The recorded versions are pinned in the
two requirements files. The FEMNIST models set `torch.set_num_threads(6)` and
the Stack Overflow MLP sets 4 threads; the original FEMNIST runs also set
`OPENBLAS_NUM_THREADS=1`.

## Datasets

Download into `training/output/downloads/`, which is the scripts' default
`--source` location. Alternatively, pass `--source PATH`. Every script checks
the SHA-256 of the file that the caches were built from. If a host has
repackaged a file, `--skip-digest` bypasses the check, but the regenerated
files may then differ.

| Dataset | File (save as) | Source | SHA-256 |
|---|---|---|---|
| FEMNIST (LEAF), Flower Hugging Face copy, revision `df739a2b09df2b5cc1ec93107659b9a9a8566487` | `femnist-train.parquet` | https://huggingface.co/datasets/flwrlabs/femnist/resolve/df739a2b09df2b5cc1ec93107659b9a9a8566487/data/train-00000-of-00001.parquet | `cdb389c8d41d5facab79a4a8d452724c30d9c629a1c1e0b1e9ae9de2c633522b` |
| UCI HAR (Human Activity Recognition Using Smartphones), CC BY 4.0 | `uci-har.zip` | https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones (download button; direct link https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip). The file is the outer archive that contains `UCI HAR Dataset.zip`. | `c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031` |
| Stack Overflow, FedJAX training SQLite (5,506,732,032 bytes) | `stackoverflow_train.sqlite` | https://storage.googleapis.com/gresearch/fedjax/stackoverflow/stackoverflow_train.sqlite | `3efde71d939fd7cb58d357dd3c060c3a8c8babec7607df9c9cc549034bafc646` |

```bash
mkdir -p training/output/downloads && cd training/output/downloads
curl -L -o femnist-train.parquet https://huggingface.co/datasets/flwrlabs/femnist/resolve/df739a2b09df2b5cc1ec93107659b9a9a8566487/data/train-00000-of-00001.parquet
curl -L -o uci-har.zip https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip
curl -L -o stackoverflow_train.sqlite https://storage.googleapis.com/gresearch/fedjax/stackoverflow/stackoverflow_train.sqlite
shasum -a 256 *
```

## Commands

Run from the repository root. `--out DIR` changes the output root (default
`training/output`), and the caches appear under `DIR/<name as in data/>`.
Every script accepts `--help`, and most accept `--groups`, `--seeds` or
`--states` to regenerate a subset. Existing cache files in the output
directory are skipped, so an interrupted run can be resumed.

Wall-clock times below were measured on one Apple-silicon laptop (arm64, 10
cores, CPU only). Times on other machines will differ.

```bash
# UCI HAR: 80 files, about 15 s
.venv-sk/bin/python training/har/har.py

# Stack Overflow, trained MLP: 640 files (groups 1-8, seeds 0-19, states 0-3), about 2 min
.venv-sk/bin/python    training/stackoverflow/prepare.py       # hashed word features of the 128 users
.venv-torch/bin/python training/stackoverflow/train.py         # 80 federated rounds per group
.venv-torch/bin/python training/stackoverflow/score.py

# Stack Overflow, stronger sparse character model: 512 files (groups 1-8, seeds 4-19), about 2.5 min
.venv-sk/bin/python training/stackoverflow_strong/run.py

# FEMNIST 30-group study: 1224 files (groups 0-33, rounds 80/120/160, development seeds 0-1,
# confirmation seeds 2-5, one or three candidate epochs)
export OPENBLAS_NUM_THREADS=1
.venv-torch/bin/python training/femnist30/prepare.py           # under 1 min
.venv-torch/bin/python training/femnist30/train.py             # 1-3 min per group, 160 rounds
.venv-torch/bin/python training/femnist30/score.py             # about 2 min per group (36 states)
# Deterministic replay of rounds 80-120: 1224 files (rounds 80, 81, 82, 84, 88, 96, 104, 112, 120)
.venv-torch/bin/python training/femnist30/decay.py             # about 2 min per group; needs train.py output

# FEMNIST update-scaling study: 384 files (groups 1-4, scales 0.5/1/2, seeds 4-11, states 0-3)
.venv-torch/bin/python training/femnist_scaled/prepare.py
.venv-torch/bin/python training/femnist_scaled/train.py        # about 5 min per group, 160 rounds
.venv-torch/bin/python training/femnist_scaled/score.py        # about 8 s per state, about 50 min in total
```

A complete regeneration takes roughly 4 to 5 hours on this hardware, almost
all of it for the two FEMNIST studies. The cheapest spot checks are HAR, one
Stack Overflow group, and one FEMNIST group:

```bash
.venv-torch/bin/python training/femnist30/prepare.py --groups 4
.venv-torch/bin/python training/femnist30/train.py --groups 4
.venv-torch/bin/python training/femnist30/score.py --groups 4 --rounds 80 --epochs 1 --phase confirmation --seeds 2
```

## Comparing with `data/`

```bash
python training/compare.py training/output                      # every regenerated cache with a counterpart in data/
python training/compare.py training/output/har/state data/har/state
python training/compare.py training/output --atol 1e-12 --quiet  # tolerate float round-off, print mismatches only
```

The comparison covers every array in every file. Integer arrays (the 0/1
correctness `bits`, record indices) must be identical. Floating-point arrays
(Shapley values, accuracies) must agree within `--atol`, which defaults to 0.
Files without a counterpart in `data/`, such as prepared inputs, model
checkpoints and the replay checkpoints, are ignored. The exit status is 0 only
when every compared file matches.

The training is deterministic: all sampling uses hash-derived seeds and no
dropout is used. On the machine and library versions above, the regenerated
files are bit-identical to `data/`. On other CPUs, operating systems or
PyTorch/BLAS builds, floating-point summation order can differ. This can
change a small number of predictions and therefore some 0/1 correctness
entries and the Shapley values derived from them. `compare.py` reports how
many entries differ in each array.

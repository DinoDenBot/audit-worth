# Cached per-record scores

All replays in this repository read only these files. Each `.npz` holds, for the
four evaluation providers (clients) of one federation state, the 0/1
correctness of every candidate action on each record of the provider's complete
working inventory and per-record Shapley values of the contributors. No raw
records are included; row indices refer to the public source datasets.

| Directory | Source | Contents |
|---|---|---|
| `femnist30/state/` | FEMNIST, 30 disjoint 8-writer federations (groups 4-33 analysed, 0-3 debugging), two-convolution CNN, federated averaging | `g{group}-r{round}-s{seed}[-e3]-{confirmation,development}.npz` at rounds 80/120/160; five actions (keep-current and four contributor updates); `bits_p` (5, n), `psi_player{j}_provider{p}` (n,), `rows_p` |
| `femnist30/decay/` | Same federations, training deterministically replayed from round 80 | rounds 80, 81, 82, 84, 88, 96, 104, 112, 120, same format |
| `femnist_scaled/state/` | FEMNIST, four 64-writer groups, round-160 CNN, candidate updates scaled by 0.5/1/2 | `g{group}-x{scale}-z{seed}-a{state}.npz`: `candidate_bits_p` (4, n), `psi_p`, `target_acc` |
| `har/state/` | UCI HAR, one 30-subject federation, logistic classifier | `s{seed}-a{state}.npz`: `bits_p` (4, n), own Shapley `psi_p` |
| `stackoverflow/` | Stack Overflow, eight 16-user federations, trained MLP | `group-{g}/score-s{seed}-a{state}.npz`: `bits_p` (4, n), `psi_p` (4, n) |
| `stackoverflow_strong/` | Same federations, stronger 120-round sparse character-feature model | `group-{g}/score-s{seed}-a{state}.npz` |
| `stackoverflow_strong/quality.json` | Same | per-group target accuracy of the stronger model, the earlier MLP, and the best constant predictor (from training over all 16 users; not derivable from the caches) |
| `femnist30/kernel_inputs/` | FEMNIST 30-group study, groups 4-33 | `g{group}.npz`: per provider p, `rows_p` (source row), `y_p` (label), `pooled_p` (n, 49; 7x7 average-pooled image in [0,1]), `h_p` (n, 128; round-80 model penultimate activations), in the row order of the score caches; used only by the kernel comparison in Section VII-D |
| `curation_studies/` | Ten credit-seeking curation studies (Table I) | one CSV per study's primary contrast: `group, seed, state, client, weight, choice_committed, choice_curated, flip, regret_committed, regret_curated, credit_committed, credit_curated, credit_target` (see below) |

`SHA256SUMS` lists a digest for every file; `python -m auditworth.verify_data` checks them.

## Curation-study rows (`curation_studies/`)

One row per curating client and training state. `group`, `seed`, `state` index
the study's federation, training seed, and training state (for the 30-group
study, `state` is the training round). `client` is the curating client's index
in its group (original writer and user identifiers are not included). `weight`
is the study's averaging weight (the client's selection probability, or 1/12 in
the 30-group study). `choice_*` is the adopted candidate with a committed or a
curated panel (five actions with 0 = keep-current in the 30-group study);
`flip` is 1 if it changed (in the two bounded-search studies, the fraction of
random inspection orders that changed it, and `choice_curated` is the modal
choice). `regret_*` is target regret in accuracy units. `credit_*` is the
curating client's panel credit (four-player record-averaged Shapley credit,
except the 30-group primary file, which uses the proxy described in the Table I
caption; the `_exact_shapley` file holds that study's exact-Shapley arm).
`credit_target` is the same credit rule on the target reference, empty where
it was not computable.

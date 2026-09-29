"""Two-layer MLP on hashed word features, deterministic local Adam updates, and federated averaging.

The string arguments to ``intseed`` are fixed domain-separation labels for the
random streams; changing them changes every trained model.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
torch.set_num_threads(4)


class TextMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1024, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, x):
        return self.net(x)


def intseed(*parts):
    return int.from_bytes(hashlib.sha256(":".join(map(str, parts)).encode()).digest()[:8], "big") % (2**32)


def load_group(out: Path, group: int):
    """Return (users, players, data); data[user][role] = (features, labels, source indices)."""
    manifest = json.loads((HERE / "client-manifest.json").read_text())
    result = {}
    for idx, uid in enumerate(sum(manifest["groups"], [])):
        if uid not in manifest["groups"][group]:
            continue
        with np.load(Path(out) / "stackoverflow" / "prepared" / f"user-{idx:02d}.npz") as z:
            result[uid] = [(torch.from_numpy(z[f"x{r}"].copy()), torch.from_numpy(z[f"y{r}"].copy()),
                            z[f"id{r}"].copy()) for r in range(4)]
    return manifest["groups"][group], manifest["players"][group], result


def make_model(params=None, seed=0):
    torch.manual_seed(seed)
    model = TextMLP()
    if params is not None:
        model.load_state_dict(params)
    return model


def clone_params(model):
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def local_update(params, block, seed, epochs=1, cap=128, lr=.003):
    """Adam on at most ``cap`` of a client's training records, minibatches of 64, hash-seeded order."""
    x, y, _ = block
    model = make_model(params)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        order = np.random.default_rng(intseed(seed, epoch)).permutation(len(y))[:cap]
        for start in range(0, len(order), 64):
            idx = order[start:start + 64]
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model(x[idx]), y[idx])
            loss.backward()
            opt.step()
    return clone_params(model)


def average_params(params_list):
    return {name: torch.stack([p[name] for p in params_list]).mean(0) for name in params_list[0]}


def coalition_params(base, updates, mask):
    """Coalition model: the average of the chosen contributors' updated weights (mask 0 keeps the state)."""
    if not mask:
        return base
    chosen = [updates[i] for i in range(4) if mask & (1 << i)]
    return average_params(chosen)


def predict(params, x, batch=2048):
    model = make_model(params)
    model.eval()
    with torch.no_grad():
        return torch.cat([model(x[i:i + batch]).argmax(1) for i in range(0, len(x), batch)]).numpy()


def target_accuracy(pred, y, slices):
    return float(np.mean([np.mean(pred[lo:hi] == y[lo:hi]) for lo, hi in slices]))


def concat_role(writers, data, role):
    arrays, labels, slices, offset = [], [], [], 0
    for uid in writers:
        x, y, _ = data[uid][role]
        arrays.append(x)
        labels.append(y.numpy())
        slices.append((offset, offset + len(y)))
        offset += len(y)
    return torch.cat(arrays), np.concatenate(labels), slices


def train_group(out: Path, group: int, rounds: int = 80):
    """80 rounds of federated averaging, 8 of 16 users per round; saves round-80.pt."""
    writers, _, data = load_group(out, group)
    model = make_model(seed=intseed("init-v1"))
    params = clone_params(model)
    history = []
    for round_no in range(1, rounds + 1):
        rng = np.random.default_rng(intseed("fedavg-v1", group, round_no))
        chosen = rng.choice(len(writers), size=8, replace=False)
        updates = [local_update(params, data[writers[i]][0], intseed("train-v1", group, round_no, writers[i]))
                   for i in chosen]
        params = average_params(updates)
        if round_no in (1, 20, 40, 60, 80):
            x, y, slices = concat_role(writers, data, 1)
            acc = target_accuracy(predict(params, x), y, slices)
            history.append({"round": round_no, "target_accuracy": acc})
            print(f"group {group} round {round_no} accuracy {acc:.5f}", flush=True)
    target = Path(out) / "stackoverflow" / "checkpoints" / f"group-{group}"
    target.mkdir(parents=True, exist_ok=True)
    torch.save(params, target / "round-80.pt")
    quality = {"rounds": rounds, "history": history}
    (target / "quality.json").write_text(json.dumps(quality, indent=2) + "\n")
    return quality

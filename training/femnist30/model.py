"""Two-convolution FEMNIST CNN, deterministic local training, and equal-client federated averaging.

The string arguments to ``seed_for`` are fixed domain-separation labels for the
random streams (client sampling, minibatch order); changing them changes every
trained model.
"""
import copy
import hashlib

import numpy as np
import torch
from torch import nn

torch.set_num_threads(6)


def seed_for(prefix, value):
    return int.from_bytes(hashlib.sha256((prefix + value).encode()).digest()[:8], "big") % (2**32)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(1, 32, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
                                 nn.Conv2d(32, 64, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
                                 nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Linear(128, 62))

    def forward(self, x):
        return self.net(x)


def tensor_block(block):
    x, y, _ = block
    return torch.from_numpy(x.astype(np.float32)[:, None, :, :] / 255), torch.from_numpy(y.astype(np.int64))


def new_model(seed):
    torch.manual_seed(seed)
    return CNN()


def local_fit(model, block, seed, epochs=2, lr=.03, batch_size=32, optimizer="sgd"):
    """Minibatch SGD or Adam on one client's block, with a hash-seeded record order per epoch."""
    x, y = tensor_block(block)
    if optimizer == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr)
    elif optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError(optimizer)
    model.train()
    for epoch in range(epochs):
        order = np.random.default_rng(seed_for("cnn-local-order-v1:", f"{seed}:{epoch}")).permutation(len(y))
        for start in range(0, len(y), batch_size):
            ix = order[start:start + batch_size]
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(x[ix]), y[ix])
            loss.backward()
            opt.step()
    return model


def predict(model, block, batch_size=512):
    x, _, _ = block
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            t = torch.from_numpy(x[start:start + batch_size].astype(np.float32)[:, None, :, :] / 255)
            out.append(model(t).argmax(dim=1).numpy())
    return np.concatenate(out)


def equal_user_accuracy(model, parts, writers, role):
    return float(np.mean([np.mean(predict(model, parts[w][role]) == parts[w][role][1]) for w in writers]))


def fedavg_round(global_model, parts, writers, round_index, group_tag, clients_per_round=16):
    """One round: sample clients, one Adam epoch each on their training role, average the weights."""
    rng = np.random.default_rng(seed_for("cnn-fedavg-clients-v1:", f"{group_tag}:{round_index}"))
    chosen = rng.choice(np.asarray(writers), size=clients_per_round, replace=False)
    states = []
    for w in chosen:
        m = copy.deepcopy(global_model)
        local_fit(m, parts[str(w)][0], f"fedavg:{group_tag}:{round_index}:{w}", epochs=1, lr=.001, optimizer="adam")
        states.append(m.state_dict())
    with torch.no_grad():
        global_model.load_state_dict({k: torch.stack([s[k] for s in states]).mean(dim=0) for k in states[0]})
    return [str(w) for w in chosen]


def terminal_models(state, parts, players, seed, state_index):
    """Branch the state with one contributor's update, then form each contributor's candidate update."""
    branch = players[state_index]
    branch_model = local_fit(copy.deepcopy(state), parts[branch][0],
                             f"branch:{seed}:{state_index}:{branch}", epochs=1, lr=.01)
    out = []
    for j, w in enumerate(players):
        out.append(local_fit(copy.deepcopy(branch_model), parts[w][0],
                             f"candidate:{seed}:{state_index}:{j}:{w}", epochs=1, lr=.01))
    return branch_model, out


def coalition_model(state, candidates, mask):
    """Apply the mean update of the contributors in ``mask`` to ``state`` (mask 0 keeps the state)."""
    if mask == 0:
        return state
    chosen = [j for j in range(4) if mask & (1 << j)]
    sm = state.state_dict()
    cms = [candidates[j].state_dict() for j in chosen]
    m = copy.deepcopy(state)
    with torch.no_grad():
        m.load_state_dict({k: sm[k] + torch.stack([c[k] - sm[k] for c in cms]).mean(dim=0) for k in sm})
    return m

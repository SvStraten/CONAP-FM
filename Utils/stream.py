"""Stream helpers used by CONAP-FM.

Evaluation protocol for every method
  * the log is cut into the first 15% of the cases (validation, used only for tuning) and the rest (evaluation)
  * the evaluated stream is the evaluation part in its original event order
  * every window, including a short last one, is first tested and then used for training
"""
import random

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _first(x):
    """test_inputs may be a dict of tasks, a list of tasks or an array. Return the concatenated array."""
    if isinstance(x, dict):
        return np.concatenate([np.asarray(x[k]) for k in sorted(x.keys())])
    if isinstance(x, list) and len(x) > 0 and isinstance(x[0], (list, np.ndarray)):
        return np.concatenate([np.asarray(v) for v in x])
    return np.asarray(x)


def infer_n_classes(sampler):
    return int(np.max(_first(sampler.test_labels))) + 1


def apply_split(sampler, split):
    """Cut the sampler in place. Returns the boolean mask over the full event stream that was kept.
    validation = first 15% of the cases, evaluation = the remaining 85%, full = everything."""
    case_ids = sampler.case_ids
    _, idx = np.unique(case_ids, return_index=True)
    cases = case_ids[np.sort(idx)]
    cut = int(len(cases) * 0.15)
    if split == "validation":
        mask = np.isin(case_ids, cases[:cut])
    elif split == "evaluation":
        mask = np.isin(case_ids, cases[cut:])
    else:
        mask = np.ones(len(case_ids), dtype=bool)

    ti = sampler.test_inputs
    if isinstance(ti, dict):
        k = sorted(ti.keys())[0]
        sampler.test_inputs = {k: ti[k][mask]}
        sampler.test_labels = {k: sampler.test_labels[k][mask]}
    elif isinstance(ti, list) and len(ti) > 0 and isinstance(ti[0], (list, np.ndarray)):
        sampler.test_inputs = [ti[0][mask]]
        sampler.test_labels = [sampler.test_labels[0][mask]]
    else:
        sampler.test_inputs = ti[mask]
        sampler.test_labels = sampler.test_labels[mask]
    return mask


def flatten(sampler):
    """Return the stream as arrays X (N x L_max, activity ids, most recent first, zero padded) and y (N)."""
    X = _first(sampler.test_inputs)
    y = _first(sampler.test_labels)
    if X.ndim == 1:
        X = X[:, None]
    return X, y.astype(np.int64)


def to_eval_coordinates(indices, mask):
    """Translate event indices of the FULL log into indices of the evaluated stream.
    Indices that fall into the removed validation part are dropped."""
    kept = np.flatnonzero(mask)
    out = []
    for d in indices:
        e = int(np.searchsorted(kept, d))
        if 0 < e < len(kept):
            out.append(e)
    return sorted(set(out))


def load_stream(dataset, split="evaluation", max_events=0):
    """Preprocess (cached) and cut a log. Returns data_name, sampler, X, y, mask."""
    from Utils.preprocess import preprocess
    data_name, sampler = preprocess(dataset)
    mask = apply_split(sampler, split)
    truncate(sampler, max_events)
    X, y = flatten(sampler)
    return data_name, sampler, X, y, mask


def truncate(sampler, n):
    """Keep only the first n events of the evaluated stream (for quick tests)."""
    if not n or n <= 0:
        return
    ti, tl = sampler.test_inputs, sampler.test_labels
    if isinstance(ti, dict):
        k = sorted(ti.keys())[0]
        sampler.test_inputs, sampler.test_labels = {k: ti[k][:n]}, {k: tl[k][:n]}
    elif isinstance(ti, list) and len(ti) > 0 and isinstance(ti[0], (list, np.ndarray)):
        sampler.test_inputs, sampler.test_labels = [ti[0][:n]], [tl[0][:n]]
    else:
        sampler.test_inputs, sampler.test_labels = ti[:n], tl[:n]

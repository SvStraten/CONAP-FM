import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from Methods.CONAP_FM.calora_handler import CaLoraHandler
from Methods.CONAP_FM.run import build_model, infer_n_classes, set_seed
from Utils.preprocess import preprocess


def make_loader(sampler, batch_size, device):
    ti, tl = sampler.test_inputs, sampler.test_labels
    if isinstance(ti, dict):
        x, y = np.concatenate(list(ti.values())), np.concatenate(list(tl.values()))
    elif isinstance(ti, list) and len(ti) > 0 and isinstance(ti[0], (list, np.ndarray)):
        x, y = np.concatenate(ti), np.concatenate(tl)
    else:
        x, y = np.asarray(ti), np.asarray(tl)
    x = torch.tensor(x, dtype=torch.long)
    if x.ndim == 1:
        x = x.unsqueeze(1)
    y = torch.tensor(y, dtype=torch.long)
    return DataLoader(TensorDataset(x.to(device), y.to(device)), batch_size=batch_size, shuffle=True)


def pretrain(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    _, sampler = preprocess(args.dataset)
    loader = make_loader(sampler, args.batch_size, device)
    model = build_model(args.model_name, infer_n_classes(sampler), "lora", args.lora_r, args.lora_alpha,
                        args.lora_dropout, args.lora_targets, device)

    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    crit = nn.CrossEntropyLoss()
    model.train()
    for ep in range(args.epochs):
        tot, n = 0.0, 0
        for bx, by in loader:
            opt.zero_grad()
            loss = crit(model(bx), by)
            loss.backward()
            opt.step()
            tot += loss.item()
            n += 1
        print(f"Epoch {ep + 1}/{args.epochs} | loss {tot / max(1, n):.4f}")

    handler = CaLoraHandler(device=device, projection_threshold=args.calora_energy)
    model.eval()
    model.zero_grad()
    acc = {}
    for b, (bx, by) in enumerate(loader):
        if b >= args.calora_batches:
            break
        nn.CrossEntropyLoss(reduction="sum")(model(bx), by).backward()
        for name, p in model.named_parameters():
            if "lora" in name and p.grad is not None:
                acc[name] = p.grad.detach().clone() if name not in acc else acc[name] + p.grad.detach()
        model.zero_grad()
    for name, g in acc.items():
        handler.update_memory(name, g)

    torch.save(model.backbone.state_dict(), os.path.join(args.output_dir, "offline_lora.pt"))
    torch.save(model.projection.state_dict(), os.path.join(args.output_dir, "offline_head_not_reused.pt"))
    handler.save_subspace(os.path.join(args.output_dir, "offline_subspace.pt"))
    print(f"Saved to {args.output_dir}. Memory dims (max) {max(handler.memory_dims().values())}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--output-dir", default="checkpoints")
    p.add_argument("--model-name", default="arnir0/Tiny-LLM")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-targets", choices=["all", "attn"], default="all")
    p.add_argument("--calora-energy", type=float, default=0.97)
    p.add_argument("--calora-batches", type=int, default=50)
    p.add_argument("--seed", type=int, default=40)
    pretrain(p.parse_args())

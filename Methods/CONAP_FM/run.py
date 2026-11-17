"""Run CONAP-FM (or one of its ablation variants) on one log and one seed.

Examples
  python -m Methods.CONAP_FM.run --dataset Data/DomesticDeclarations.csv --data-split evaluation
  python -m Methods.CONAP_FM.run --dataset Data/DomesticDeclarations.csv --paca --tag with_mask       (adds the importance mask, off by default)
  python -m Methods.CONAP_FM.run --dataset Data/DomesticDeclarations.csv --model-name distilbert/distilgpt2 --plots-dir runs/distilgpt2
  python -m Methods.CONAP_FM.run --dataset Data/DomesticDeclarations.csv --variant random_full --tag random_full
  python -m Methods.CONAP_FM.run --dataset Data/DomesticDeclarations.csv \
      --pretrained-lora-path checkpoints/RequestForPayment_from_PrepaidTravelCost/offline_lora.pt --pretrained-subspace-path checkpoints/RequestForPayment_from_PrepaidTravelCost/offline_subspace.pt

Model variants (--variant)
  lora         pretrained backbone (Tiny-LLM by default, or distilbert/distilgpt2), frozen, LoRA adapters trained online (CONAP-FM)
  full_ft      pretrained backbone, all weights trained online, no LoRA
  random_full  same architecture with random initial weights, all weights trained online
  random_lora  same architecture with random frozen weights, LoRA adapters trained online

Input to the model. Each prefix of a case is a vector of integer activity ids (most recent event
first, zero padded up to L_max = longest case in the log). The ids index the frozen token embedding
table of the backbone. The hidden state of the last position goes through a new linear head that is
trained online from scratch. Pretraining transfers only the LoRA weights and the memory subspace,
not the head, because the activity sets of two logs differ.
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoConfig, AutoModel

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from Methods.CONAP_FM.calora_handler import CaLoraHandler
from Methods.CONAP_FM.engine import OnlineCaLoraEngine
from Setup import drifts
from Utils.metrics import save_granular_accuracy
from Utils.preprocess import preprocess
from Utils.stream import apply_split, infer_n_classes, set_seed, truncate

ATTN = ["q_proj", "k_proj", "v_proj", "o_proj"]
MLP = ["gate_proj", "up_proj", "down_proj"]
# GPT-2 family (distilgpt2). Its attention and MLP projections are Conv1D layers, so LoRA needs fan_in_fan_out=True.
GPT2_ATTN = r".*attn\.(c_attn|c_proj)"
GPT2_ALL = ["c_attn", "c_proj", "c_fc"]
HIDDEN = {"arnir0/Tiny-LLM": 192, "HuggingFaceTB/SmolLM2-135M": 576,
          "Qwen/Qwen2.5-0.5B": 896, "meta-llama/Llama-3.2-1B": 2048,
          "distilbert/distilgpt2": 768, "distilgpt2": 768}


class LLMWithProjection(nn.Module):
    def __init__(self, backbone, hidden_size, num_classes):
        super().__init__()
        self.backbone = backbone
        self.projection = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids):
        out = self.backbone(input_ids, output_hidden_states=True)
        if hasattr(out, "last_hidden_state"):
            h = out.last_hidden_state
        elif isinstance(out, tuple):
            h = out[0]
        else:
            h = out.hidden_states[-1]
        return self.projection(h[:, -1, :])


def build_model(model_name, n_classes, variant="lora", lora_r=16, lora_alpha=32,
                lora_dropout=0.05, lora_targets="all", device="cpu"):
    config = AutoConfig.from_pretrained(model_name)
    if getattr(config, "pad_token_id", None) is None:
        config.pad_token_id = config.eos_token_id
    random_init = variant in ("random_full", "random_lora")
    base = AutoModel.from_config(config) if random_init else AutoModel.from_pretrained(model_name, config=config)
    hidden = HIDDEN.get(model_name, getattr(config, "hidden_size", 576))

    if variant in ("lora", "random_lora"):
        is_gpt2 = getattr(config, "model_type", "") == "gpt2"
        if is_gpt2:
            targets = GPT2_ATTN if lora_targets == "attn" else GPT2_ALL
        else:
            targets = ATTN if lora_targets == "attn" else ATTN + MLP
        peft_cfg = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, inference_mode=False, r=lora_r,
                              lora_alpha=lora_alpha, lora_dropout=lora_dropout, target_modules=targets,
                              fan_in_fan_out=is_gpt2)
        backbone = get_peft_model(base, peft_cfg)
    else:
        backbone = base            # every weight stays trainable
    return LLMWithProjection(backbone, hidden, n_classes).to(device)


def run(args):
    set_seed(args.seed)
    data_name, sampler = preprocess(args.dataset)
    mask = apply_split(sampler, args.data_split)
    truncate(sampler, args.max_events)
    n_classes = infer_n_classes(sampler)

    drift_idx = drifts.drift_points(data_name)
    drift_idx = [d - args.oracle_offset for d in drift_idx]

    use_adapt_modules = args.variant in ("lora", "random_lora")
    # the importance mask is off in the final method (it hurt on several logs), --paca switches it on for the ablation
    use_paca = args.paca and not args.no_paca and not args.no_backward_transfer and use_adapt_modules
    use_caga = (not args.no_caga) and not args.no_backward_transfer and use_adapt_modules
    use_svd = (not args.no_svd) and not args.no_backward_transfer and use_adapt_modules
    trigger = "oracle" if args.oracle_drift else args.trigger

    print(f"{data_name} | seed {args.seed} | split {args.data_split} | variant {args.variant} | tag {args.tag}")
    print(f"model {args.model_name}")
    print(f"PACA={use_paca}({args.paca_mode}) CAGA={use_caga}({args.caga_mode}) SVD={use_svd} trigger={trigger}")

    model = build_model(args.model_name, n_classes, args.variant, args.lora_r, args.lora_alpha,
                        args.lora_dropout, args.lora_targets, args.device)

    if args.pretrained_lora_path and os.path.exists(args.pretrained_lora_path) and use_adapt_modules:
        state = torch.load(args.pretrained_lora_path, map_location=args.device, weights_only=True)
        missing, unexpected = model.backbone.load_state_dict(state, strict=False)
        print(f"[Pretrained] loaded LoRA weights, unexpected keys {len(unexpected)}")

    subspace_path = None if args.no_pretrained_subspace else args.pretrained_subspace_path
    handler = CaLoraHandler(old_task_subspace_path=subspace_path, device=args.device,
                            projection_threshold=args.calora_energy, epsilon=args.epsilon,
                            paca_mode=args.paca_mode, caga_mode=args.caga_mode,
                            max_memory_fraction=args.max_memory_fraction)

    eng = OnlineCaLoraEngine(
        model=model, calora_handler=handler, device=args.device, window_size=args.window_size,
        epochs_per_window=args.epochs_per_window, learning_rate=args.lr, input_dtype=args.input_dtype,
        verbose=not args.quiet, batch_size=args.batch_size, svd_limit_batches=args.calora_batches,
        loss_window_length=args.loss_window_length, loss_variance_threshold=args.loss_variance_threshold,
        svd_cooldown_windows=args.svd_cooldown_windows, use_paca=use_paca, use_caga=use_caga,
        use_svd=use_svd, trigger_mode=trigger, drift_indices=drift_idx,
        fixed_interval=args.fixed_interval, adwin_delta=args.adwin_delta, ph_delta=args.ph_delta,
        ph_threshold=args.ph_threshold, adapt_order=args.adapt_order)

    start = time.perf_counter()
    res = eng.run_stream(sampler)
    runtime_h = (time.perf_counter() - start) / 3600.0

    method_name = f"CONAP_FM_{args.tag}"
    save_granular_accuracy(data_name, method_name, args.seed, res["prediction_results"]["actual_labels"],
                           res["prediction_results"]["prediction_labels"])

    out_dir = os.path.join(args.plots_dir, "results")
    os.makedirs(out_dir, exist_ok=True)
    record = {k: v for k, v in res.items() if k != "prediction_results"}
    record.update(dataset=data_name, seed=args.seed, tag=args.tag, variant=args.variant, split=args.data_split,
                  trigger=trigger, use_paca=use_paca, use_caga=use_caga, use_svd=use_svd,
                  paca_mode=args.paca_mode, caga_mode=args.caga_mode, adapt_order=args.adapt_order, window_size=args.window_size,
                  lora_r=args.lora_r, lora_alpha=args.lora_alpha, lr=args.lr, epsilon=args.epsilon, cooldown=args.svd_cooldown_windows, steps=args.epochs_per_window, tau=args.loss_variance_threshold,
                  drift_indices=drift_idx, pretrained=bool(args.pretrained_lora_path), runtime_hours=runtime_h,
                  model_name=args.model_name)
    with open(os.path.join(out_dir, f"{data_name}__{args.tag}__seed{args.seed}.json"), "w") as f:
        json.dump(record, f)

    log_path = os.path.join(args.plots_dir, "ablation_results.csv")
    row = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "dataset": data_name, "tag": args.tag,
           "seed": args.seed, "overall_accuracy": f"{res['overall_accuracy']:.4f}",
           "macro_f1": f"{res['macro_f1']:.4f}", "triggers": res["plateau_triggers"],
           "trainable_params": res["trainable_parameters"], "runtime_hours": f"{runtime_h:.4f}"}
    new = not os.path.isfile(log_path)
    with open(log_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)
    print(f"Accuracy {res['overall_accuracy']:.4f} | macro F1 {res['macro_f1']:.4f} | triggers {res['plateau_triggers']}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--model-name", default="arnir0/Tiny-LLM")
    p.add_argument("--variant", choices=["lora", "full_ft", "random_full", "random_lora"], default="lora")
    p.add_argument("--tag", default="full", help="name of this configuration in result files")
    p.add_argument("--max-events", type=int, default=0, help="only use the first n events (quick tests)")
    p.add_argument("--window-size", type=int, default=100)
    p.add_argument("--epochs-per-window", type=int, default=10, help="Adam steps per window")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--device", default="auto")
    p.add_argument("--input-dtype", choices=["long", "float"], default="long")
    p.add_argument("--plots-dir", default="runs")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-targets", choices=["all", "attn"], default="all")
    p.add_argument("--calora-energy", type=float, default=0.95)
    p.add_argument("--calora-batches", type=int, default=30)
    p.add_argument("--epsilon", type=float, default=0.05)
    p.add_argument("--max-memory-fraction", type=float, default=0.5)
    p.add_argument("--paca-mode", choices=["legacy", "scaled"], default="scaled")
    p.add_argument("--adapt-order", choices=["caga_paca", "paca_caga"], default="caga_paca",
                   help="caga_paca = project the gradient first and weight it with the mask after, paca_caga = old order")
    p.add_argument("--caga-mode", choices=["orthogonal", "affinity"], default="orthogonal")
    p.add_argument("--loss-window-length", type=int, default=5)
    p.add_argument("--loss-variance-threshold", type=float, default=5e-4)
    p.add_argument("--svd-cooldown-windows", type=int, default=10)
    p.add_argument("--data-split", choices=["full", "validation", "evaluation"], default="evaluation")
    p.add_argument("--trigger", choices=["variance", "oracle", "fixed", "adwin", "page_hinkley"], default="variance")
    p.add_argument("--fixed-interval", type=int, default=20)
    p.add_argument("--adwin-delta", type=float, default=0.002)
    p.add_argument("--ph-delta", type=float, default=0.005)
    p.add_argument("--ph-threshold", type=float, default=0.5)
    p.add_argument("--oracle-drift", action="store_true", help="same as --trigger oracle")
    p.add_argument("--oracle-offset", type=int, default=0)
    p.add_argument("--no-backward-transfer", action="store_true", help="plain sequential LoRA (Forward), no memory and no CaGA")
    p.add_argument("--paca", action="store_true", help="use the parameter importance mask (off by default)")
    p.add_argument("--no-paca", action="store_true", help="kept for old command lines, the mask is off anyway")
    p.add_argument("--no-caga", action="store_true")
    p.add_argument("--no-svd", action="store_true")
    p.add_argument("--pretrained-lora-path", default=None)
    p.add_argument("--pretrained-subspace-path", default=None)
    p.add_argument("--no-pretrained-subspace", action="store_true")
    args = p.parse_args()
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args


if __name__ == "__main__":
    run(parse_args())

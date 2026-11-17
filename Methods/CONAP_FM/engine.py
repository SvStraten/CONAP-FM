import collections
import resource
import time
from typing import List, Optional

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.optim import Adam


class PageHinkley:
    def __init__(self, delta=0.005, threshold=0.5, alpha=0.9999, min_instances=5):
        self.delta, self.threshold, self.alpha, self.min_instances = delta, threshold, alpha, min_instances
        self.reset()

    def reset(self):
        self.n, self.mean, self.m, self.m_min = 0, 0.0, 0.0, 0.0

    def update(self, x):
        self.n += 1
        self.mean += (x - self.mean) / self.n
        self.m = self.alpha * self.m + (x - self.mean - self.delta)
        self.m_min = min(self.m_min, self.m)
        if self.n >= self.min_instances and (self.m - self.m_min) > self.threshold:
            self.reset()
            return True
        return False


def _sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


class OnlineCaLoraEngine:
    def __init__(self,
                 model: torch.nn.Module,
                 calora_handler,
                 device: str = "cpu",
                 window_size: int = 100,
                 epochs_per_window: int = 10,
                 learning_rate: float = 2e-4,
                 input_dtype: str = "long",
                 verbose: bool = True,
                 batch_size: int = 32,
                 svd_limit_batches: int = 30,
                 loss_window_length: int = 5,
                 loss_variance_threshold: float = 5e-4,
                 svd_cooldown_windows: int = 10,
                 use_paca: bool = True,
                 use_caga: bool = True,
                 use_svd: bool = True,
                 trigger_mode: str = "variance",
                 drift_indices: Optional[List[int]] = None,
                 fixed_interval: int = 20,
                 adwin_delta: float = 0.002,
                 ph_delta: float = 0.005,
                 ph_threshold: float = 0.5,
                 adapt_order: str = "caga_paca",
                 adapt_filter: str = "lora",
                 latency_probe: int = 5):
        assert trigger_mode in ("variance", "oracle", "fixed", "adwin", "page_hinkley")
        self.model = model
        self.handler = calora_handler
        self.device = device
        self.model.to(self.device)
        self.W = window_size
        self.gradient_steps = epochs_per_window
        self.input_dtype = input_dtype
        self.verbose = verbose
        self.batch_size = batch_size
        self.svd_limit_batches = svd_limit_batches
        self.use_paca, self.use_caga, self.use_svd = use_paca, use_caga, use_svd
        assert adapt_order in ("caga_paca", "paca_caga")
        self.adapt_order = adapt_order
        self.trigger_mode = trigger_mode
        self.drift_indices = sorted(drift_indices or [])
        self.fixed_interval = fixed_interval
        self.adapt_filter = adapt_filter
        self.latency_probe = latency_probe

        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = Adam(params, lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

        self.loss_queue = collections.deque(maxlen=loss_window_length)
        self.loss_queue_len = loss_window_length
        self.tau = loss_variance_threshold
        self.cooldown_windows = svd_cooldown_windows
        self.cooldown = 0

        self.adwin = None
        if trigger_mode == "adwin":
            from river.drift import ADWIN
            self.adwin = ADWIN(delta=adwin_delta)
        self.ph = PageHinkley(delta=ph_delta, threshold=ph_threshold) if trigger_mode == "page_hinkley" else None

        self.trigger_windows: List[int] = []

    # ------------------------------------------------------------ helpers
    def _to_tensor(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(np.asarray(x))
        if x.ndim == 1:
            x = x.unsqueeze(1)
        x = x.long() if self.input_dtype == "long" else x.float()
        return x.to(self.device)

    def _to_labels(self, y):
        yt = torch.from_numpy(np.asarray(y)) if not torch.is_tensor(y) else y
        return yt.long().to(self.device)

    def _peak_memory_mb(self):
        if str(self.device).startswith("cuda"):
            return torch.cuda.max_memory_allocated() / 2 ** 20
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    # ------------------------------------------------- memory consolidation
    def _extract_subspace(self, x_win, y_win, window_idx, reason):
        if not self.use_svd:
            return
        self.trigger_windows.append(window_idx)
        if self.verbose:
            print(f"[Trigger {reason}] window {window_idx}, consolidating memory (#{len(self.trigger_windows)})")

        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(self._to_tensor(x_win), self._to_labels(y_win)),
            batch_size=self.batch_size, shuffle=False)

        was_training = self.model.training
        self.model.eval()                      
        self.model.zero_grad()
        acc = {}
        for b, (bx, by) in enumerate(loader):
            if b >= self.svd_limit_batches:
                break
            self.criterion(self.model(bx), by).backward()
            for name, p in self.model.named_parameters():
                if self.adapt_filter in name and p.grad is not None:
                    acc[name] = p.grad.detach().clone() if name not in acc else acc[name] + p.grad.detach()
            self.model.zero_grad()
        for name, g in acc.items():
            try:
                self.handler.update_memory(name, g)
            except Exception as e:  
                print(f"[Memory] update failed for {name}: {e}")
        self.model.train(was_training)

    def _decide_trigger(self, i, loss, x_win, y_win, prev_win, next_end, win_end):
        """Return (x, y, reason) to extract from, or None."""
        if not self.use_svd:
            return None
        if self.trigger_mode == "oracle":
            if any(win_end < dp <= next_end for dp in self.drift_indices):
                return x_win, y_win, "oracle"
            return None
        if self.trigger_mode == "fixed":
            if (i + 1) % self.fixed_interval == 0:
                return x_win, y_win, "fixed"
            return None
        if self.trigger_mode in ("adwin", "page_hinkley"):
            if self.trigger_mode == "adwin":
                self.adwin.update(loss)
                fired = self.adwin.drift_detected
            else:
                fired = self.ph.update(loss)
            if fired and prev_win is not None:
                return prev_win[0], prev_win[1], self.trigger_mode
            return None
        # variance (plateau) trigger
        self.loss_queue.append(loss)
        if self.cooldown > 0:
            self.cooldown -= 1
            return None
        if len(self.loss_queue) < self.loss_queue_len:
            return None
        if np.var(np.array(self.loss_queue)) < self.tau:
            self.loss_queue.clear()
            self.cooldown = self.cooldown_windows
            return x_win, y_win, "plateau"
        return None

    # --------------------------------------------------------------- stream
    @staticmethod
    def _flatten(data):
        ti, tl = data.test_inputs, data.test_labels
        if isinstance(ti, dict):
            keys = sorted(ti.keys())
            return np.concatenate([ti[k] for k in keys]), np.concatenate([tl[k] for k in keys])
        if isinstance(ti, list):
            if len(ti) > 0 and isinstance(ti[0], (list, np.ndarray)):
                return np.concatenate(ti), np.concatenate(tl)
            return np.array(ti), np.array(tl)
        return ti, tl

    def run_stream(self, data):
        X, Y = self._flatten(data)
        N, W = len(X), self.W
        n_win = (N + W - 1) // W

        if str(self.device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()

        all_pred, all_true = [], []
        win_acc, win_loss, t_pred, t_upd, t_single = [], [], [], [], []
        prev_win = None
        correct = seen = 0
        self.model.train()

        for i in range(n_win):
            s, e = i * W, min((i + 1) * W, N)
            x_win, y_win = X[s:e], Y[s:e]
            next_end = min(e + W, N)

            # 1. test (prequential)
            self.model.eval()
            _sync(self.device)
            t0 = time.perf_counter()
            with torch.no_grad():
                logits = self.model(self._to_tensor(x_win))
                y_hat = torch.argmax(logits, dim=1).cpu().numpy()
            _sync(self.device)
            t_pred.append(time.perf_counter() - t0)

            # single-event latency probe (what a deployed predictor would see per new event)
            probe = []
            with torch.no_grad():
                for j in range(min(self.latency_probe, len(x_win))):
                    xj = self._to_tensor(x_win[j:j + 1])
                    _sync(self.device)
                    t1 = time.perf_counter()
                    self.model(xj)
                    _sync(self.device)
                    probe.append(time.perf_counter() - t1)
            t_single.append(float(np.mean(probe)) if probe else 0.0)

            correct += int((y_hat == y_win).sum())
            seen += len(y_win)
            all_pred.extend(y_hat.tolist())
            all_true.extend(np.asarray(y_win).tolist())
            win_acc.append(float((y_hat == y_win).mean()))

            # 2. train
            self.model.train()
            _sync(self.device)
            t0 = time.perf_counter()
            loss_acc = self._train_window(x_win, y_win)
            _sync(self.device)
            t_upd.append(time.perf_counter() - t0)
            avg_loss = loss_acc / max(1, self.gradient_steps)
            win_loss.append(avg_loss)

            # 3. consolidation trigger
            trig = self._decide_trigger(i, avg_loss, x_win, y_win, prev_win, next_end, e)
            if trig is not None:
                self._extract_subspace(trig[0], trig[1], i, trig[2])
            prev_win = (x_win, y_win)

            if self.verbose:
                print(f"  Window {i + 1}/{n_win} | loss {avg_loss:.4f} | acc {win_acc[-1]:.4f}")

        acc = correct / seen if seen else 0.0
        macro_f1 = f1_score(all_true, all_pred, average="macro")
        n_train = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        n_all = sum(p.numel() for p in self.model.parameters())
        tp = np.array(t_pred)
        per_event = tp / np.array([min(W, N - k * W) for k in range(n_win)])
        return {
            "prediction_results": {"actual_labels": all_true, "prediction_labels": all_pred},
            "window_accuracies": win_acc,
            "window_losses": win_loss,
            "window_predict_seconds": t_pred,
            "window_update_seconds": t_upd,
            "single_event_latency_seconds": t_single,
            "overall_accuracy": acc,
            "macro_f1": macro_f1,
            "plateau_triggers": len(self.trigger_windows),
            "trigger_windows": self.trigger_windows,
            "trainable_parameters": n_train,
            "total_parameters": n_all,
            "peak_memory_mb": self._peak_memory_mb(),
            "predict_seconds_per_event_batched_mean": float(per_event.mean()),
            "predict_seconds_per_event_batched_p95": float(np.percentile(per_event, 95)),
            "single_event_latency_mean": float(np.mean(t_single)),
            "single_event_latency_p95": float(np.percentile(t_single, 95)),
            "update_seconds_per_window_mean": float(np.mean(t_upd)),
            "update_seconds_per_window_p95": float(np.percentile(t_upd, 95)),
            "handler_stats": self.handler.summary(),
        }

    def _train_window(self, x_win, y_win):
        n = len(y_win)
        total_loss = 0.0
        use_adapt = self.use_paca or self.use_caga
        for _ in range(self.gradient_steps):
            self.optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            bs = self.batch_size or n
            for st in range(0, n, bs):
                bx = self._to_tensor(x_win[st:st + bs])
                by = self._to_labels(y_win[st:st + bs])
                loss_sum = self.criterion(self.model(bx), by)
                step_loss += float(loss_sum.detach())
                (loss_sum / float(n)).backward()
            total_loss += step_loss / n

            if use_adapt:
                with torch.no_grad():
                    for name, p in self.model.named_parameters():
                        if p.requires_grad and p.grad is not None and self.adapt_filter in name:
                            if self.adapt_order == "caga_paca":
                                # project the true gradient first, then weight what is left (descent direction is kept)
                                if self.use_caga and self.handler.old_u_matrices:
                                    p.grad = self.handler.apply_caga_correction(name, p)
                                if self.use_paca:
                                    p.grad.mul_(self.handler.paca_mask(p))
                            else:
                                # old order, the projection of an already rescaled gradient can point uphill
                                if self.use_paca:
                                    p.grad.mul_(self.handler.paca_mask(p))
                                if self.use_caga and self.handler.old_u_matrices:
                                    p.grad = self.handler.apply_caga_correction(name, p)
            self.optimizer.step()
        return total_loss

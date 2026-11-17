"""Memory subspace, parameter importance mask (PACA) and gradient adaptation (CaGA).

Changes with respect to the first release
-----------------------------------------
* The secondary SVD that merges old and new subspaces is truncated by singular
  value. Before, every trigger added k columns, so the memory could grow until it
  spanned the whole space and the projected gradient became zero.
* The importance mask has two modes.
    legacy  1 + softmax(|g * theta|). Mean is about 1 + 1/n, so it is almost a no-op.
    scaled  n * softmax(standardised |g * theta|). Mean is exactly 1 and the spread
            is large enough to matter. This is the default.
* CaGA has two modes.
    orthogonal  subtract the projection whenever rho > epsilon (the released behaviour).
    affinity    subtract only if rho > epsilon AND the cosine between the projected
                gradient and the stored mean gradient of consolidated tasks is negative.
* The memory dimension is capped at `max_memory_fraction` of the matrix input dimension
  (default 0.5), keeping the directions with the largest singular values.
* epsilon is a constructor argument (it was hard-coded to 0.05).
* Optional statistics (mask spread, number of CaGA corrections, memory dimension)
  for the paper and for debugging.
"""
import os
from collections import defaultdict

import torch
import torch.nn.functional as F


class CaLoraHandler:
    def __init__(self,
                 old_task_subspace_path=None,
                 device="cuda",
                 projection_threshold=0.95,
                 epsilon=0.05,
                 paca_mode="scaled",
                 caga_mode="orthogonal",
                 truncate_memory=True,
                 svd_tol=1e-2,
                 paca_max=10.0,
                 max_memory_fraction=0.5,
                 track_stats=True):
        assert paca_mode in ("legacy", "scaled")
        assert caga_mode in ("orthogonal", "affinity")
        self.device = device
        self.projection_threshold = projection_threshold
        self.epsilon = epsilon
        self.paca_mode = paca_mode
        self.caga_mode = caga_mode
        self.truncate_memory = truncate_memory
        self.svd_tol = svd_tol
        self.paca_max = paca_max
        self.max_memory_fraction = max_memory_fraction
        self.track_stats = track_stats

        self.old_u_matrices = {}   # name -> (in_dim x k) orthonormal basis
        self.mem_grads = {}        # name -> unit-norm mean gradient of consolidated tasks
        self.stats = defaultdict(float)
        self._warned_shape = set()

        self.old_task_subspace_path = old_task_subspace_path
        if old_task_subspace_path:
            self.load_subspace()

    # ------------------------------------------------------------------ io
    def load_subspace(self, path=None):
        if path is not None:
            self.old_task_subspace_path = path
        p = self.old_task_subspace_path
        if p and os.path.exists(p):
            data = torch.load(p, map_location=self.device, weights_only=False)
            self.old_u_matrices = data["u_matrices"]
            self.mem_grads = data.get("mem_grads", {})
        elif p:
            print(f"Warning, subspace path {p} not found.")

    def save_subspace(self, path):
        torch.save({"u_matrices": self.old_u_matrices, "mem_grads": self.mem_grads}, path)

    # -------------------------------------------------------------- memory
    @torch.no_grad()
    def update_memory(self, name, grad):
        """Add the dominant right singular directions of `grad` to the memory of matrix `name`."""
        G = grad.detach().float()
        total = torch.sum(G ** 2)
        if not torch.isfinite(total) or total <= 0:
            return
        _, S, Vh = torch.linalg.svd(G, full_matrices=False)
        energy = torch.cumsum(S ** 2, dim=0) / torch.sum(S ** 2)
        idx = (energy >= self.projection_threshold).nonzero(as_tuple=True)[0]
        k = int(idx[0].item()) + 1 if len(idx) > 0 else len(S)
        V_new = Vh[:k, :].T                                   # in_dim x k

        if name not in self.old_u_matrices:
            self.old_u_matrices[name] = V_new
        else:
            combined = torch.cat([self.old_u_matrices[name].to(V_new.device), V_new], dim=1)
            Uc, Sc, _ = torch.linalg.svd(combined, full_matrices=False)
            if self.truncate_memory:
                keep = Sc > self.svd_tol * Sc[0]
                Uc = Uc[:, keep]
            # never let the memory span the whole space, otherwise CaGA removes the entire gradient
            max_k = max(1, int(self.max_memory_fraction * Uc.shape[0]))
            Uc = Uc[:, :max_k]
            self.old_u_matrices[name] = Uc

        gn = G / (G.norm() + 1e-12)
        if name in self.mem_grads:
            m = self.mem_grads[name].to(gn.device) + gn
            gn = m / (m.norm() + 1e-12)
        self.mem_grads[name] = gn

    def memory_dims(self):
        return {n: int(u.shape[1]) for n, u in self.old_u_matrices.items()}

    # ---------------------------------------------------------------- PACA
    @torch.no_grad()
    def paca_mask(self, param):
        s = torch.abs(param.grad * param) + 1e-10
        flat = s.reshape(-1)
        if self.paca_mode == "legacy":
            mask = 1.0 + F.softmax(flat, dim=0)
        else:
            z = (flat - flat.mean()) / (flat.std() + 1e-12)
            mask = flat.numel() * F.softmax(z, dim=0)
            mask = torch.clamp(mask, max=self.paca_max)
        mask = mask.view_as(param)
        if self.track_stats:
            self.stats["mask_calls"] += 1
            self.stats["mask_min_sum"] += float(mask.min())
            self.stats["mask_max_sum"] += float(mask.max())
        return mask

    # ---------------------------------------------------------------- CaGA
    @torch.no_grad()
    def apply_caga_correction(self, name, param):
        g = param.grad
        if name not in self.old_u_matrices:
            return g
        U = self.old_u_matrices[name].to(g.device)
        if g.shape[-1] != U.shape[0]:
            if name not in self._warned_shape:
                print(f"[CaGA] shape mismatch for {name}: grad {tuple(g.shape)} vs basis {tuple(U.shape)}")
                self._warned_shape.add(name)
            return g

        proj = (g @ U) @ U.T
        rho = proj.norm() / (g.norm() + 1e-8)
        subtract = bool(rho > self.epsilon) and self.epsilon < 1.0   # epsilon 1.0 means CaGA never subtracts

        if subtract and self.caga_mode == "affinity" and name in self.mem_grads:
            mg = self.mem_grads[name].to(g.device)
            if mg.shape == proj.shape:
                aff = F.cosine_similarity(proj.reshape(1, -1), mg.reshape(1, -1)).item()
                subtract = aff < 0
                if self.track_stats:
                    self.stats["affinity_sum"] += aff
                    self.stats["affinity_n"] += 1

        if self.track_stats:
            self.stats["caga_calls"] += 1
            self.stats["caga_subtracted"] += int(subtract)
            self.stats["rho_sum"] += float(rho)
        return g - proj if subtract else g

    def summary(self):
        s = dict(self.stats)
        if s.get("mask_calls"):
            s["mask_min_mean"] = s["mask_min_sum"] / s["mask_calls"]
            s["mask_max_mean"] = s["mask_max_sum"] / s["mask_calls"]
        if s.get("caga_calls"):
            s["caga_subtract_rate"] = s["caga_subtracted"] / s["caga_calls"]
            s["rho_mean"] = s["rho_sum"] / s["caga_calls"]
        dims = self.memory_dims()
        s["memory_dim_max"] = max(dims.values()) if dims else 0
        s["memory_dim_mean"] = sum(dims.values()) / len(dims) if dims else 0
        return s

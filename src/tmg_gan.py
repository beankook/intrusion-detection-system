"""Modified TMG-GAN Training Module (Sec. 3.5.3 / 4.3, Algorithm 1).

Components
    G_1..G_K  one generator per traffic class      x~ = G_k(z),  z ~ N(0, I)
    F         shared feature extractor
    D         discriminator head on F   (real vs synthetic)
    C         classifier head on F      (traffic class)

Losses
    L_D     = -E[log D(x)] - E[log(1 - D(G(z)))]
    L_G     = -E[log D(G(z))]
    L_cos   = 1 - cos(F(x), F(x~))                       (intra-class, pulls x~_k towards real class k)
              + w * mean_{j != k} max(0, cos(F(x~_k), F(x~_j)))   (inter-class, reduces class overlap)
    L_total = L_G + lambda * L_cos  (+ lambda_cls * CE(C(x~_k), k), as in the base TMG-GAN paper)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_

from .utils import get_logger

log = get_logger()


def _mlp(sizes: list[int], act: nn.Module) -> nn.Sequential:
    layers: list[nn.Module] = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), act]
    return nn.Sequential(*layers)


class Generator(nn.Module):
    """G_k : z -> synthetic feature vector in [0, 1]^n (same range as the min-max scaled data)."""

    def __init__(self, z_dim: int, n_features: int, hidden: list[int]):
        super().__init__()
        self.z_dim = z_dim
        self.body = _mlp([z_dim, *hidden], nn.LeakyReLU(0.2))
        self.out = nn.Sequential(nn.Linear(hidden[-1], n_features), nn.Sigmoid())

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.out(self.body(z))

    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        return self(torch.randn(n, self.z_dim, device=device))


class DiscriminatorClassifier(nn.Module):
    """Feature extractor F with a discriminator head D and a classifier head C."""

    def __init__(self, n_features: int, n_classes: int, hidden: list[int]):
        super().__init__()
        self.F = _mlp([n_features, *hidden], nn.LeakyReLU(0.2))
        self.D = nn.Linear(hidden[-1], 1)
        self.C = nn.Linear(hidden[-1], n_classes)

    def forward(self, x: torch.Tensor):
        h = self.F(x)
        return self.D(h).squeeze(1), self.C(h), h   # D logit, C logits, F(x)


def cosine_similarity_loss(h_real, h_fake, inter_w: float) -> torch.Tensor:
    """h_real / h_fake: lists (one entry per class) of F(.) batches of equal size."""
    intra = torch.stack(
        [1 - F_.cosine_similarity(hr.detach(), hf, dim=1).mean() for hr, hf in zip(h_real, h_fake)]
    ).mean()
    if inter_w <= 0 or len(h_fake) < 2:
        return intra
    inter = []
    for k, hk in enumerate(h_fake):
        for j, hj in enumerate(h_fake):
            if j != k:
                inter.append(F_.cosine_similarity(hk, hj.detach(), dim=1).clamp(min=0).mean())
    return intra + inter_w * torch.stack(inter).mean()


class TMGGAN:
    """Tabular Multi-Generator GAN. `fit` is Algorithm 1 of the paper."""

    def __init__(self, n_features: int, n_classes: int, cfg, device: torch.device):
        g = cfg.gan
        self.cfg, self.device, self.K = g, device, n_classes
        self.generators = nn.ModuleList(
            [Generator(g.z_dim, n_features, list(g.g_hidden)) for _ in range(n_classes)]
        ).to(device)                                                    # 1: initialise G_1..G_K
        self.cd = DiscriminatorClassifier(n_features, n_classes, list(g.f_hidden)).to(device)  # 2: D and C
        self.opt_g = torch.optim.Adam(self.generators.parameters(), lr=g.lr_g, betas=(0.5, 0.999))
        self.opt_d = torch.optim.Adam(self.cd.parameters(), lr=g.lr_d, betas=(0.5, 0.999))
        self.history: list[dict] = []

    # ------------------------------------------------------------------ #
    def _real_batches(self, pools: list[torch.Tensor]) -> list[torch.Tensor]:
        bs = self.cfg.batch_size
        return [p[torch.randint(0, len(p), (bs,), device=self.device)] for p in pools]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TMGGAN":
        g, dev, bce = self.cfg, self.device, F_.binary_cross_entropy_with_logits
        Xt = torch.as_tensor(X, dtype=torch.float32, device=dev)
        pools = [Xt[torch.as_tensor(y == k, device=dev)] for k in range(self.K)]
        if any(len(p) == 0 for p in pools):
            raise ValueError("Every class needs at least one training sample.")
        labels = torch.arange(self.K, device=dev).repeat_interleave(g.batch_size)

        for epoch in range(1, g.epochs + 1):                            # 3: for epoch = 1..T
            self.cd.train(); self.generators.train()
            for _ in range(g.d_iters):                                  # 4: for t_d iterations
                real = torch.cat(self._real_batches(pools))             # 5: real batch x
                with torch.no_grad():                                   # 6-7: z, x~ = G_k(z)
                    fake = torch.cat([G.sample(g.batch_size, dev) for G in self.generators])
                d_real, _, _ = self.cd(real)
                d_fake, _, _ = self.cd(fake)
                loss_d = bce(d_real, torch.ones_like(d_real)) + bce(d_fake, torch.zeros_like(d_fake))
                self.opt_d.zero_grad(); loss_d.backward(); self.opt_d.step()   # 8: update theta_D
                _, c_real, _ = self.cd(real)
                loss_c = F_.cross_entropy(c_real, labels)
                self.opt_d.zero_grad(); loss_c.backward(); self.opt_d.step()   # 9: update theta_C

            for _ in range(g.g_iters):                                  # 11: for t_g iterations
                reals = self._real_batches(pools)
                fakes = [G.sample(g.batch_size, dev) for G in self.generators]  # 12-13
                with torch.no_grad():
                    h_real = [self.cd(r)[2] for r in reals]
                d_fake, c_fake, h_all = self.cd(torch.cat(fakes))
                h_fake = list(h_all.split(g.batch_size))
                loss_adv = bce(d_fake, torch.ones_like(d_fake))          # L_G
                loss_cos = cosine_similarity_loss(h_real, h_fake, g.inter_class_weight)  # 14
                loss_cls = F_.cross_entropy(c_fake, labels)
                loss_g = loss_adv + g.lambda_cos * loss_cos + g.lambda_cls * loss_cls
                self.opt_g.zero_grad(); loss_g.backward(); self.opt_g.step()   # 15: update theta_G

            rec = dict(epoch=epoch, loss_d=loss_d.item(), loss_c=loss_c.item(),
                       loss_g=loss_adv.item(), loss_cos=loss_cos.item())
            self.history.append(rec)
            if epoch % g.log_every == 0 or epoch in (1, g.epochs):
                log.info("TMG-GAN %5d/%d | L_D %.4f | L_C %.4f | L_G %.4f | L_cos %.4f",
                         epoch, g.epochs, rec["loss_d"], rec["loss_c"], rec["loss_g"], rec["loss_cos"])
        return self                                                     # 18: theta_G, theta_D, theta_C

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def classify(self, x: torch.Tensor) -> torch.Tensor:
        self.cd.eval()
        return self.cd(x)[1].argmax(1)

    def state_dict(self) -> dict:
        return {"generators": self.generators.state_dict(), "cd": self.cd.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        self.generators.load_state_dict(state["generators"])
        self.cd.load_state_dict(state["cd"])

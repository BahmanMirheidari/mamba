"""
Thin wrappers around ConfigurableMultiModal for the named fusion baselines.
These give clean, comparable numbers for the ablation table.
"""
import torch
import torch.nn as nn

from novelty import ConfigurableMultiModal, NoveltyConfig


def _masked_mean(seq, mask):
    return (seq * mask.unsqueeze(-1)).sum(1) / \
           (mask.sum(1, keepdim=True) + 1e-9)


class _PooledProjector(nn.Module):
    """Simple pooled-feature baselines without any sequence encoder."""

    def __init__(self, dims, n_outputs, hidden=128, dropout=0.2):
        super().__init__()
        total = sum(dims.values())
        self.net = nn.Sequential(
            nn.Linear(total, hidden), nn.LayerNorm(hidden),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_outputs))

    def forward(self, feats):
        order = sorted(feats.keys())
        x = torch.cat([feats[k] for k in order], dim=-1)
        return self.net(x)


class EarlyFusionBaseline(nn.Module):
    """Pool each modality, concat, MLP."""

    def __init__(self, d_a, d_t, d_z, n_outputs, d_model=128, dropout=0.2):
        super().__init__()
        self.a_pool = nn.Linear(d_a, d_model)
        self.t_pool = nn.Linear(d_t, d_model)
        self.z_pool = nn.Linear(d_z, d_model)
        self.net = _PooledProjector(
            {"a": d_model, "t": d_model, "z": d_model},
            n_outputs, hidden=d_model, dropout=dropout)

    def forward(self, a, am, t, tm, z):
        feats = {
            "a": _masked_mean(self.a_pool(a), am),
            "t": _masked_mean(self.t_pool(t), tm),
            "z": self.z_pool(z),
        }
        return self.net(feats), None, None


class LateFusionBaseline(nn.Module):
    """Separate head per modality, learned weighted average of logits."""

    def __init__(self, d_a, d_t, d_z, n_outputs, dropout=0.2):
        super().__init__()
        self.a_pool = nn.Linear(d_a, 128)
        self.t_pool = nn.Linear(d_t, 128)
        self.z_pool = nn.Linear(d_z, 128)
        self.head_a = nn.Sequential(nn.Linear(128, 64), nn.GELU(),
                                    nn.Linear(64, n_outputs))
        self.head_t = nn.Sequential(nn.Linear(128, 64), nn.GELU(),
                                    nn.Linear(64, n_outputs))
        self.head_z = nn.Sequential(nn.Linear(128, 64), nn.GELU(),
                                    nn.Linear(64, n_outputs))
        self.w = nn.Parameter(torch.ones(3))

    def forward(self, a, am, t, tm, z):
        import torch.nn.functional as F
        la = self.head_a(_masked_mean(self.a_pool(a), am))
        lt = self.head_t(_masked_mean(self.t_pool(t), tm))
        lz = self.head_z(self.z_pool(z))
        w = F.softmax(self.w, dim=0)
        return w[0] * la + w[1] * lt + w[2] * lz, None, None


class GatedFusionBaseline(nn.Module):
    """Per-sample softmax gate over projected pooled features."""

    def __init__(self, d_a, d_t, d_z, n_outputs, d_model=128, dropout=0.2):
        super().__init__()
        self.a_pool = nn.Linear(d_a, d_model)
        self.t_pool = nn.Linear(d_t, d_model)
        self.z_pool = nn.Linear(d_z, d_model)
        self.gate = nn.Sequential(
            nn.Linear(3 * d_model, d_model), nn.GELU(),
            nn.Linear(d_model, 3))
        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Dropout(dropout),
            nn.Linear(d_model, n_outputs))

    def forward(self, a, am, t, tm, z):
        import torch.nn.functional as F
        fa = _masked_mean(self.a_pool(a), am)
        ft = _masked_mean(self.t_pool(t), tm)
        fz = self.z_pool(z)
        stack = torch.stack([fa, ft, fz], dim=1)
        w = F.softmax(self.gate(torch.cat([fa, ft, fz], dim=-1)), dim=-1)
        fused = (stack * w.unsqueeze(-1)).sum(1)
        return self.head(fused), None, None


class CrossAttentionBaseline(nn.Module):
    """Minimal cross-attention: audio queries text, pooled, tabular concat."""

    def __init__(self, d_a, d_t, d_z, n_outputs, d_model=128,
                 n_heads=4, dropout=0.2):
        super().__init__()
        self.a_proj = nn.Linear(d_a, d_model)
        self.t_proj = nn.Linear(d_t, d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model + d_z, n_outputs)

    def forward(self, a, am, t, tm, z):
        A = self.a_proj(a)
        T = self.t_proj(t)
        ca, _ = self.attn(A, T, T, key_padding_mask=~tm.bool())
        H = self.norm(A + ca)
        H_pool = _masked_mean(H, am)
        return self.head(torch.cat([H_pool, z], dim=-1)), None, None


def build_novelty(novelty_preset, overrides,
                  d_a, d_t, d_z, n_outputs, d_model):
    from novelty import resolve_novelty
    nc = resolve_novelty(novelty_preset, overrides)
    return ConfigurableMultiModal(
        d_a, d_t, d_z, n_outputs, nc, d_model=d_model)
"""
Configurable multi-modal model. Every knob in NoveltyConfig toggles one
architectural component so ablations reduce to flipping a boolean.
"""
from dataclasses import dataclass, asdict
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import MambaBlock


@dataclass
class NoveltyConfig:
    # encoder
    bidirectional: bool = True
    n_mamba_layers: int = 2
    n_transformer_layers: int = 0
    d_state: int = 16
    d_conv: int = 4

    # fusion strategy
    fusion_mode: str = "mid"           # early | mid | late

    # tabular injection
    tabular_injection: str = "bias"    # bias | gate | concat | none

    # cross-modal attention
    cross_attention: str = "a_to_t"    # a_to_t | t_to_a | bidir | none
    n_heads: int = 4

    # second pass over fused sequence
    second_pass: str = "mamba"         # mamba | transformer | both | none

    # pooling
    pooling: str = "mean"              # mean | max | attention | last

    # gates
    temporal_gate: bool = True
    modality_gate: bool = True

    # regularisation
    modality_dropout: float = 0.0
    contrastive_loss: float = 0.0
    reconstruction_loss: float = 0.0

    # misc
    dropout: float = 0.2


# ---- named presets ----

NOVELTY_PRESETS: Dict[str, NoveltyConfig] = {
    "full":            NoveltyConfig(),

    "no_bidirectional": NoveltyConfig(bidirectional=False),
    "no_tabular":       NoveltyConfig(tabular_injection="none"),
    "no_cross_attn":    NoveltyConfig(cross_attention="none"),
    "no_second_pass":   NoveltyConfig(second_pass="none"),
    "no_temporal_gate": NoveltyConfig(temporal_gate=False),
    "no_modality_gate": NoveltyConfig(modality_gate=False),

    "tab_concat":       NoveltyConfig(tabular_injection="concat"),
    "tab_gate":         NoveltyConfig(tabular_injection="gate"),

    "cross_bidir":      NoveltyConfig(cross_attention="bidir"),
    "cross_t2a":        NoveltyConfig(cross_attention="t_to_a"),

    "second_transformer": NoveltyConfig(second_pass="transformer"),
    "second_both":        NoveltyConfig(second_pass="both"),

    "fusion_early":     NoveltyConfig(fusion_mode="early"),
    "fusion_late":      NoveltyConfig(fusion_mode="late"),

    "hybrid_2t":        NoveltyConfig(n_transformer_layers=2),
    "hybrid_4t":        NoveltyConfig(n_transformer_layers=4),

    "pool_attention":   NoveltyConfig(pooling="attention"),
    "pool_max":         NoveltyConfig(pooling="max"),
    "pool_last":        NoveltyConfig(pooling="last"),

    "regularised":      NoveltyConfig(modality_dropout=0.3,
                                      contrastive_loss=0.1,
                                      reconstruction_loss=0.1),
    "contrastive":      NoveltyConfig(contrastive_loss=0.2),
}


def resolve_novelty(name: str = None, overrides: str = None) -> NoveltyConfig:
    if name and name in NOVELTY_PRESETS:
        cfg = NoveltyConfig(**asdict(NOVELTY_PRESETS[name]))
    elif name:
        raise KeyError(f"Unknown preset '{name}'. "
                       f"Available: {sorted(NOVELTY_PRESETS)}")
    else:
        cfg = NoveltyConfig()

    if overrides:
        for pair in overrides.split(","):
            pair = pair.strip()
            if not pair:
                continue
            k, v = pair.split("=", 1)
            k, v = k.strip(), v.strip()
            if not hasattr(cfg, k):
                raise KeyError(f"Unknown field '{k}'")
            cur = getattr(cfg, k)
            if isinstance(cur, bool):
                v = v.lower() in ("1", "true", "yes", "y", "t")
            elif isinstance(cur, int) and not isinstance(cur, bool):
                v = int(v)
            elif isinstance(cur, float):
                v = float(v)
            setattr(cfg, k, v)
    return cfg


# ---- building blocks ----

class _TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=4 * d_model, dropout=dropout,
            batch_first=True, activation="gelu", norm_first=True)

    def forward(self, x, mask=None):
        km = (~mask.bool()) if mask is not None else None
        return self.layer(x, src_key_padding_mask=km)


class _AttentionPool(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.q = nn.Linear(d_model, 1)

    def forward(self, x, mask):
        s = self.q(x).squeeze(-1)
        if mask is not None:
            s = s.masked_fill(~mask.bool(), float("-inf"))
        w = torch.softmax(s, dim=-1)
        return (x * w.unsqueeze(-1)).sum(1)


# ---- the configurable model ----

class ConfigurableMultiModal(nn.Module):
    def __init__(self, d_audio, d_text, d_tabular, n_outputs,
                 nc: NoveltyConfig, d_model: int = 256):
        super().__init__()
        self.nc = nc
        self.d_model = d_model
        self.encodes_jointly = (nc.fusion_mode == "early")

        self.a_proj = nn.Linear(d_audio, d_model)
        self.t_proj = nn.Linear(d_text, d_model)
        if nc.tabular_injection != "none":
            self.z_proj = nn.Sequential(
                nn.Linear(d_tabular, d_model),
                nn.LayerNorm(d_model), nn.GELU())

        if not self.encodes_jointly:
            self.a_fwd, self.a_bwd = self._mamba_stacks()
            self.t_fwd, self.t_bwd = self._mamba_stacks()
            dim_out = 2 * d_model if nc.bidirectional else d_model
            self.a_fuse = nn.Linear(dim_out, d_model)
            self.t_fuse = nn.Linear(dim_out, d_model)
            self.a_norm = nn.LayerNorm(d_model)
            self.t_norm = nn.LayerNorm(d_model)

            if nc.n_transformer_layers > 0:
                self.a_tf = nn.ModuleList([
                    _TransformerBlock(d_model, nc.n_heads, nc.dropout)
                    for _ in range(nc.n_transformer_layers)])
                self.t_tf = nn.ModuleList([
                    _TransformerBlock(d_model, nc.n_heads, nc.dropout)
                    for _ in range(nc.n_transformer_layers)])

        if nc.cross_attention != "none":
            self.cross_a2t = nn.MultiheadAttention(
                d_model, nc.n_heads, dropout=nc.dropout, batch_first=True)
            self.cross_t2a = nn.MultiheadAttention(
                d_model, nc.n_heads, dropout=nc.dropout, batch_first=True)
            self.cross_norm = nn.LayerNorm(d_model)
            if nc.temporal_gate:
                self.temporal_gate = nn.Sequential(
                    nn.Linear(2 * d_model, d_model), nn.Sigmoid())

        if nc.second_pass in ("mamba", "both"):
            self.j_fwd, self.j_bwd = self._mamba_stacks()
            dim_out = 2 * d_model if nc.bidirectional else d_model
            self.j_fuse = nn.Linear(dim_out, d_model)
            self.j_norm = nn.LayerNorm(d_model)
        if nc.second_pass in ("transformer", "both"):
            self.j_tf = nn.ModuleList([
                _TransformerBlock(d_model, nc.n_heads, nc.dropout)
                for _ in range(2)])

        if nc.modality_gate:
            self.modality_gate = nn.Sequential(
                nn.Linear(3 * d_model, d_model), nn.GELU(),
                nn.Linear(d_model, 2), nn.Softmax(dim=-1))

        if nc.pooling == "attention":
            self.pool_a = _AttentionPool(d_model)
            self.pool_t = _AttentionPool(d_model)
            self.pool_h = _AttentionPool(d_model)

        if nc.contrastive_loss > 0:
            self.contrast_a = nn.Linear(d_model, d_model)
            self.contrast_t = nn.Linear(d_model, d_model)
        if nc.reconstruction_loss > 0:
            self.recon_a = nn.Linear(d_model, d_model)
            self.recon_t = nn.Linear(d_model, d_model)

        head_in = d_model + (d_tabular if nc.tabular_injection == "concat" else 0)
        self.drop = nn.Dropout(nc.dropout)
        self.head = nn.Linear(head_in, n_outputs)

    def _mamba_stacks(self):
        fwd = nn.ModuleList([
            MambaBlock(self.d_model, self.nc.d_state,
                       d_conv=self.nc.d_conv, dropout=self.nc.dropout)
            for _ in range(self.nc.n_mamba_layers)])
        bwd = (nn.ModuleList([
            MambaBlock(self.d_model, self.nc.d_state,
                       d_conv=self.nc.d_conv, dropout=self.nc.dropout)
            for _ in range(self.nc.n_mamba_layers)])
            if self.nc.bidirectional else None)
        return fwd, bwd

    def _encode(self, x, fwd, bwd, tf=None, mask=None):
        h_f = x
        for blk in fwd:
            h_f = blk(h_f)
        if bwd is not None:
            h_b = torch.flip(x, dims=[1])
            for blk in bwd:
                h_b = blk(h_b)
            h_b = torch.flip(h_b, dims=[1])
            h = torch.cat([h_f, h_b], dim=-1)
        else:
            h = h_f
        if tf is not None:
            for blk in tf:
                h = blk(h, mask)
        return h

    def _pool(self, x, mask, pool_module=None):
        if self.nc.pooling == "attention" and pool_module is not None:
            return pool_module(x, mask)
        if self.nc.pooling == "max":
            if mask is not None:
                x = x.masked_fill(~mask.bool().unsqueeze(-1), float("-inf"))
            return x.max(1).values
        if self.nc.pooling == "last":
            if mask is not None:
                lengths = mask.sum(1).long().clamp(min=1) - 1
                return x[torch.arange(x.size(0)), lengths]
            return x[:, -1]
        if mask is not None:
            return (x * mask.unsqueeze(-1)).sum(1) / \
                   (mask.sum(1, keepdim=True) + 1e-9)
        return x.mean(1)

    def forward(self, audio_seq, audio_mask, text_seq, text_mask, tabular):
        train = self.training
        A = self.a_proj(audio_seq)
        T = self.t_proj(text_seq)
        Z = (self.z_proj(tabular)
             if self.nc.tabular_injection != "none" else None)

        # ---- early fusion ----
        if self.encodes_jointly:
            if train and self.nc.modality_dropout > 0:
                B = A.size(0)
                dm = torch.rand(B, device=A.device) < self.nc.modality_dropout
                keep = torch.rand(B, device=A.device) < 0.5
                A = A * (~(dm & keep)).float().view(B, 1, 1)
                T = T * (~(dm & ~keep)).float().view(B, 1, 1)

            X = torch.cat([A, T], dim=1)
            if Z is not None:
                X = X + Z.unsqueeze(1)
            joint_mask = torch.cat([audio_mask, text_mask], dim=1)
            H = self._encode(X, self.a_fwd, self.a_bwd, mask=joint_mask)
            H = self.a_norm(H)
            H_pool = self._pool(H, joint_mask,
                                getattr(self, "pool_h", None))
            if self.nc.tabular_injection == "concat":
                H_pool = torch.cat([H_pool, tabular], dim=-1)
            logits = self.head(self.drop(H_pool))
            return logits, {"mod_w": None,
                            "aux_loss": torch.tensor(0.0, device=logits.device)}

        # ---- mid / late fusion ----
        if train and self.nc.modality_dropout > 0:
            B = A.size(0)
            dm = torch.rand(B, device=A.device) < self.nc.modality_dropout
            keep = torch.rand(B, device=A.device) < 0.5
            A = A * (~(dm & keep)).float().view(B, 1, 1)
            T = T * (~(dm & ~keep)).float().view(B, 1, 1)

        A_enc = self._encode(A, self.a_fwd, self.a_bwd,
                             tf=getattr(self, "a_tf", None),
                             mask=audio_mask)
        T_enc = self._encode(T, self.t_fwd, self.t_bwd,
                             tf=getattr(self, "t_tf", None),
                             mask=text_mask)
        A_enc = self.a_norm(self.a_fuse(A_enc))
        T_enc = self.t_norm(self.t_fuse(T_enc))

        if Z is not None:
            if self.nc.tabular_injection == "bias":
                A_enc = A_enc + Z.unsqueeze(1)
                T_enc = T_enc + Z.unsqueeze(1)
            elif self.nc.tabular_injection == "gate":
                g = torch.sigmoid(Z)
                A_enc = A_enc * g.unsqueeze(1)
                T_enc = T_enc * g.unsqueeze(1)

        if self.nc.cross_attention == "a_to_t":
            ca, _ = self.cross_a2t(A_enc, T_enc, T_enc,
                                   key_padding_mask=~text_mask.bool())
            F_ = ca
        elif self.nc.cross_attention == "t_to_a":
            ca, _ = self.cross_t2a(T_enc, A_enc, A_enc,
                                   key_padding_mask=~audio_mask.bool())
            F_ = ca
        elif self.nc.cross_attention == "bidir":
            ca_at, _ = self.cross_a2t(A_enc, T_enc, T_enc,
                                      key_padding_mask=~text_mask.bool())
            ca_ta, _ = self.cross_t2a(T_enc, A_enc, A_enc,
                                      key_padding_mask=~audio_mask.bool())
            F_ = ca_at + ca_ta
        else:
            F_ = A_enc

        if self.nc.cross_attention != "none" and self.nc.temporal_gate:
            g = self.temporal_gate(torch.cat([A_enc, F_], dim=-1))
            H = self.cross_norm(g * F_ + (1 - g) * A_enc)
        else:
            H = self.cross_norm(F_)

        if self.nc.second_pass in ("mamba", "both"):
            H = self._encode(H, self.j_fwd, self.j_bwd, mask=audio_mask)
            H = self.j_norm(self.j_fuse(H))
        if self.nc.second_pass in ("transformer", "both"):
            for blk in self.j_tf:
                H = blk(H, audio_mask)

        H_pool = self._pool(H, audio_mask, getattr(self, "pool_h", None))
        A_pool = self._pool(A_enc, audio_mask, getattr(self, "pool_a", None))
        T_pool = self._pool(T_enc, text_mask, getattr(self, "pool_t", None))

        mod_w = None
        if self.nc.modality_gate:
            mod_w = self.modality_gate(
                torch.cat([A_pool, T_pool, H_pool], dim=-1))

        if self.nc.tabular_injection == "concat" and Z is not None:
            H_pool = torch.cat([H_pool, tabular], dim=-1)

        logits = self.head(self.drop(H_pool))

        aux = torch.tensor(0.0, device=logits.device)
        if self.nc.contrastive_loss > 0:
            a_p = F.normalize(self.contrast_a(A_pool), dim=-1)
            t_p = F.normalize(self.contrast_t(T_pool), dim=-1)
            lt = a_p @ t_p.t() / 0.07
            labels = torch.arange(a_p.size(0), device=a_p.device)
            aux = aux + self.nc.contrastive_loss * 0.5 * (
                F.cross_entropy(lt, labels) + F.cross_entropy(lt.t(), labels))
        if self.nc.reconstruction_loss > 0:
            aux = aux + self.nc.reconstruction_loss * (
                F.mse_loss(self.recon_a(H_pool), A_pool.detach())
                + F.mse_loss(self.recon_t(H_pool), T_pool.detach()))

        return logits, {"mod_w": mod_w, "aux_loss": aux}
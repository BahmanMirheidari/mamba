"""
Baseline models: classical (XGB/LGBM/LogReg) and unimodal Mamba sequence
classifiers.
"""
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("[models] mamba-ssm not installed — Mamba will not be available")


def train_classical(X_tr, y_tr, X_va, model_type: str = "xgboost", **kwargs):
    if model_type == "xgboost":
        if y_tr.dtype.kind in 'iu':
            from xgboost import XGBClassifier
            clf = XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                use_label_encoder=False, eval_metric="logloss",
                random_state=42, **kwargs)
        else:
            from xgboost import XGBRegressor
            clf = XGBRegressor(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, **kwargs)
    elif model_type == "lightgbm":
        if y_tr.dtype.kind in 'iu':
            from lightgbm import LGBMClassifier
            clf = LGBMClassifier(n_estimators=300, max_depth=4,
                                 learning_rate=0.05, subsample=0.8,
                                 colsample_bytree=0.8, random_state=42,
                                 verbosity=-1, **kwargs)
        else:
            from lightgbm import LGBMRegressor
            clf = LGBMRegressor(n_estimators=300, max_depth=4,
                                learning_rate=0.05, subsample=0.8,
                                colsample_bytree=0.8, random_state=42,
                                verbosity=-1, **kwargs)
    elif model_type == "logreg":
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=2000, C=1.0, **kwargs)
    elif model_type == "ridge":
        from sklearn.linear_model import Ridge
        clf = Ridge(alpha=1.0, **kwargs)
    else:
        raise ValueError(model_type)

    clf.fit(X_tr, y_tr)
    if hasattr(clf, "predict_proba"):
        pred = clf.predict_proba(X_va)
    else:
        pred = clf.predict(X_va)
    return clf, pred


class MambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16,
                 d_conv: int = 4, expand: int = 2, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(d_model=d_model, d_state=d_state,
                           d_conv=d_conv, expand=expand)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.drop(self.mamba(self.norm(x)))


class MambaSeqClassifier(nn.Module):
    def __init__(self, d_in: int, d_model: int = 256, n_layers: int = 4,
                 d_state: int = 16, d_conv: int = 4, expand: int = 2,
                 n_outputs: int = 2, dropout: float = 0.1,
                 pooling: str = "mean"):
        super().__init__()
        assert MAMBA_AVAILABLE, "pip install mamba-ssm causal-conv1d"
        self.pooling = pooling
        self.proj = nn.Linear(d_in, d_model)
        self.blocks = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_outputs)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        h = self.proj(x)
        for blk in self.blocks:
            h = blk(h)
        h = self.norm(h)
        if self.pooling == "mean":
            if mask is not None:
                h = (h * mask.unsqueeze(-1)).sum(1) / \
                    (mask.sum(1, keepdim=True) + 1e-9)
            else:
                h = h.mean(1)
        elif self.pooling == "last":
            h = h[:, -1]
        elif self.pooling == "max":
            h = h.max(1).values
        return self.head(h)
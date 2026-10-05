"""DeadReckoningTransformer, a Transformer comparison arm for the dead-
reckoning task (Track A, Phase A3 of the coursework-grounded extension
roadmap).

Hyperparameters (d_model=64, nhead=4, dropout=0.1, norm_first=True) are
taken directly from the confirmed working TinyViT reference at
D:\\ΠΜΣ\\Υπολογιστική όραση\\Εργασία\\project-2\\ex3_cifar.py. d_model=64
deliberately matches DeadReckoningLSTM's hidden_size=64, so any accuracy
difference in the ablation matrix is attributable to the architecture, not
a bigger/smaller model. A 10-timestep x 14-feature IMU window has no 2-D
patch structure, so PatchEmbed's Conv2d is replaced by a per-timestep
Linear projection, the temporal analogue.

Built with a custom encoder layer (not nn.TransformerEncoderLayer) so
attention weights can be extracted for the analysis deliverable (which
timesteps the model attends to when predicting the position delta) without
depending on private internals of the stock module. The scaled-dot-product-
attention math itself follows the same formulation independently verified
against torch's own implementation in
D:\\ΠΜΣ\\Βαθιά Μάθηση\\Εργασίες εξαμήνου\\3η Άσκηση\\...\\25118.py.

Given the data-sufficiency finding (imu_data.csv is one continuous 37.8-
minute flight, not diverse sessions, see the roadmap's Data sufficiency
check), this model leans on dropout (0.1, matching TinyViT) plus optional
time-series data augmentation (see augment_batch()) as mitigation against a
Transformer's weaker inductive bias overfitting the limited diversity here,
rather than assuming that risk away.
"""
import os
import sys

import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_processing.dataset_parser import INPUT_SIZE
WINDOW_SIZE = 10  # must match evaluate_trajectory.WINDOW_SIZE / create_dead_reckoning_dataset


class _EncoderLayerWithAttn(nn.Module):
    """Pre-norm Transformer encoder layer (norm_first=True, matching
    TinyViT) built from nn.MultiheadAttention directly so attention weights
    are retrievable via need_weights=True without touching private state of
    nn.TransformerEncoderLayer."""

    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout_ff = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x, need_weights=False):
        normed = self.norm1(x)
        attn_out, attn_weights = self.self_attn(
            normed, normed, normed,
            need_weights=need_weights, average_attn_weights=False)
        x = x + self.dropout1(attn_out)

        normed2 = self.norm2(x)
        ff = self.linear2(self.dropout_ff(self.activation(self.linear1(normed2))))
        x = x + self.dropout2(ff)
        return x, attn_weights


class TransformerTrunk(nn.Module):
    """The embed + cls-token + positional-embedding + encoder-layer stack,
    factored out as its own submodule so it can be shared, architecturally
    identical, between DeadReckoningTransformer and
    ssl_pretext_transformer.py's MaskedTransformerAutoencoder, the same
    role nn.LSTM(14, 64, 2, ...) plays for DeadReckoningLSTM /
    MaskedLSTMAutoencoder in the LSTM arm. Weight transfer after SSL
    pretraining is then a plain trunk.load_state_dict(), no key remapping.

    forward(x) -> (b, window+1, d_model) token sequence (cls token at index
    0, one token per input timestep after it), plus attention weights if
    need_weights=True.
    """

    def __init__(self, input_size=INPUT_SIZE, d_model=64, nhead=4,
                 num_layers=4, dim_feedforward=256, dropout=0.1,
                 window_size=WINDOW_SIZE):
        super().__init__()
        self.d_model = d_model
        self.window_size = window_size

        self.embed = nn.Linear(input_size, d_model)          # per-timestep "patch" embed
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, window_size + 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            _EncoderLayerWithAttn(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, need_weights=False):
        b = x.size(0)
        tokens = self.embed(x)                                   # (b, window, d_model)
        cls = self.cls_token.expand(b, -1, -1)                    # (b, 1, d_model)
        tokens = torch.cat([cls, tokens], dim=1)                  # (b, window+1, d_model)
        tokens = tokens + self.pos_embed
        tokens = self.dropout(tokens)

        attn_weights_all = [] if need_weights else None
        for layer in self.layers:
            tokens, w = layer(tokens, need_weights=need_weights)
            if need_weights:
                attn_weights_all.append(w)

        tokens = self.norm(tokens)
        if need_weights:
            return tokens, attn_weights_all
        return tokens


class DeadReckoningTransformer(nn.Module):
    """forward(x) -> (batch, output_size), same call signature as
    DeadReckoningLSTM for drop-in use in evaluate_trajectory.py. Attention
    weights are available via forward(x, need_weights=True), returning
    (output, attn_weights) where attn_weights is a list of length
    num_layers, each (batch, nhead, seq_len_with_cls, seq_len_with_cls)."""

    def __init__(self, input_size=INPUT_SIZE, d_model=64, nhead=4,
                 num_layers=4, dim_feedforward=256, dropout=0.1,
                 output_size=3, window_size=WINDOW_SIZE):
        super().__init__()
        self.trunk = TransformerTrunk(input_size, d_model, nhead, num_layers,
                                       dim_feedforward, dropout, window_size)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, output_size),
        )

    def forward(self, x, need_weights=False):
        if need_weights:
            tokens, attn_weights_all = self.trunk(x, need_weights=True)
            cls_out = tokens[:, 0, :]     # cls token's final representation
            return self.fc(cls_out), attn_weights_all
        tokens = self.trunk(x, need_weights=False)
        cls_out = tokens[:, 0, :]
        return self.fc(cls_out)


def augment_batch(x: torch.Tensor, jitter_std: float = 0.03, scale_range=(0.9, 1.1),
                   time_warp_prob: float = 0.3) -> torch.Tensor:
    """Time-series analogues of the CV coursework's RandomHorizontalFlip/
    RandomRotation/ColorJitter augmentation, adapted rather than copied since
    those specific image transforms don't apply to sensor sequences:

    - Gaussian jitter: small additive noise per-channel, the sensor-domain
      equivalent of ColorJitter, simulates ordinary measurement noise the
      real IMU already has, at a level the model shouldn't overfit to.
    - Magnitude scaling: a single random per-sample multiplier, simulating
      sensor-gain variation between units/flights.
    - Time warp (mild, probabilistic): re-samples the window along a
      slightly nonlinear time axis, simulating minor speed variation in the
      underlying maneuver, applied only with probability time_warp_prob per
      sample since it's the most aggressive of the three.

    x: (batch, window, features), already scaler-normalized. Returns an
    augmented copy; call only on training batches, never on val/eval data.
    """
    b, w, f = x.shape
    out = x.clone()

    jitter = torch.randn_like(out) * jitter_std
    out = out + jitter

    scale = torch.empty(b, 1, 1, device=x.device).uniform_(*scale_range)
    out = out * scale

    # Vectorized time warp: computed for the whole batch at once (cheap,
    # batched tensor ops) rather than a Python-level per-sample loop, then
    # blended against the un-warped batch via warp_mask, avoiding per-sample
    # Python overhead, which matters here since this runs every training
    # batch across every Track A/A5 training run.
    warp_mask = torch.rand(b, device=x.device) < time_warp_prob
    if warp_mask.any():
        idx = torch.arange(w, dtype=torch.float32, device=x.device).view(1, w)      # (1, w)
        bend = (torch.rand(b, 1, device=x.device) - 0.5) * 0.6                       # (b, 1), +/-0.3
        denom = max(w - 1, 1)
        # Mild random monotonic warp: quadratic perturbation of the sample
        # index (zero at both endpoints, peaking mid-window), same shape as
        # the original per-sample formula, applied batched.
        warped_idx = idx + bend * (idx / denom) * (denom - idx) / denom              # (b, w)
        warped_idx = warped_idx.round().clamp(0, w - 1).long()

        gather_idx = warped_idx.unsqueeze(-1).expand(-1, -1, f)                      # (b, w, f)
        warped_out = torch.gather(out, dim=1, index=gather_idx)
        out = torch.where(warp_mask.view(b, 1, 1), warped_out, out)

    return out


if __name__ == '__main__':
    model = DeadReckoningTransformer()
    dummy = torch.randn(16, WINDOW_SIZE, INPUT_SIZE)

    out = model(dummy)
    print(f"Output shape: {out.shape}  expected (16, 3)")
    assert out.shape == (16, 3)

    model.eval()   # dropout is applied to attention weights in train mode,
                    # which would break the sums-to-1 check below
    with torch.no_grad():
        out2, attn = model(dummy, need_weights=True)
    print(f"Output (with attn) shape: {out2.shape}  expected (16, 3)")
    print(f"Attention weights: {len(attn)} layers, each shape {attn[0].shape}  "
          f"expected (16, 4, {WINDOW_SIZE + 1}, {WINDOW_SIZE + 1})")
    assert len(attn) == 4
    assert attn[0].shape == (16, 4, WINDOW_SIZE + 1, WINDOW_SIZE + 1)
    # Attention rows should sum to ~1 (softmax over keys).
    row_sums = attn[0][0, 0].sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)

    aug = augment_batch(dummy)
    print(f"Augmented batch shape: {aug.shape}  expected same as input")
    assert aug.shape == dummy.shape
    assert not torch.allclose(aug, dummy)   # augmentation should actually change values

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")
    print("All smoke tests passed.")

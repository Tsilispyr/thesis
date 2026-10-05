"""DeadReckoningLSTMUncertainty -- a second output head added to the
DeadReckoningLSTM architecture so it reports a per-axis confidence alongside
its position-delta estimate, not just a point value.

New model class rather than a modification of DeadReckoningLSTM (Track A,
Phase A2 of the coursework-grounded extension roadmap): the live
recovery_orchestrator.py loads dr_lstm_imu_norm.pth by DeadReckoningLSTM's
exact class shape, so that model and its class stay untouched. This is an
additive, offline-only arm.

No uncertainty-quantification code exists anywhere in the user's MSc
coursework (checked directly) -- unlike the Transformer arm, this is
designed fresh, not ported. The technique itself is standard heteroscedastic
regression: a second linear head predicts per-axis log-variance alongside
the mean, trained with Gaussian negative log-likelihood instead of plain
MSE. The classical EKF (models/ekf_baseline.py) already exposes exactly
this kind of uncertainty for free via its covariance
(`position_uncertainty` property) -- this model is the learned counterpart.
"""
import os
import sys

import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_processing.dataset_parser import INPUT_SIZE

# log-variance is clamped during training/inference to keep exp(log_var)
# numerically sane before the model has learned a meaningful scale --
# unclamped, an early, poorly-initialized head can produce log_var values
# large/small enough that exp() over/underflows.
LOG_VAR_MIN = -10.0
LOG_VAR_MAX = 10.0


class DeadReckoningLSTMUncertainty(nn.Module):
    """Same LSTM backbone as DeadReckoningLSTM (hidden_size=64, num_layers=2,
    dropout=0.2) so any accuracy difference in the ablation matrix is
    attributable to the uncertainty head/loss, not a smaller/larger backbone.
    forward() returns (mu, log_var), each shape (batch, output_size).
    """

    def __init__(self, input_size=INPUT_SIZE, hidden_size=64,
                 num_layers=2, output_size=3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                             batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, output_size * 2),   # first half = mu, second half = log_var
        )

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        raw = self.fc(out[:, -1, :])
        mu = raw[:, :self.output_size]
        log_var = raw[:, self.output_size:].clamp(LOG_VAR_MIN, LOG_VAR_MAX)
        return mu, log_var


def gaussian_nll_loss(mu: torch.Tensor, log_var: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Heteroscedastic Gaussian negative log-likelihood, per Kendall & Gal
    (2017)'s standard formulation for learned-variance regression:

        NLL = 0.5 * (log_var + (target - mu)^2 / exp(log_var))

    averaged over the batch and output dimensions (constant 0.5*log(2*pi)
    term dropped, it doesn't affect gradients or model comparison since it's
    the same constant for every prediction).
    """
    inv_var = torch.exp(-log_var)
    sq_err = (target - mu) ** 2
    return (0.5 * (log_var + sq_err * inv_var)).mean()


if __name__ == '__main__':
    model = DeadReckoningLSTMUncertainty()
    dummy = torch.randn(16, 10, INPUT_SIZE)
    mu, log_var = model(dummy)
    print(f"mu shape: {mu.shape}  expected (16, 3)")
    print(f"log_var shape: {log_var.shape}  expected (16, 3)")
    assert mu.shape == (16, 3) and log_var.shape == (16, 3)

    target = torch.randn(16, 3)
    loss = gaussian_nll_loss(mu, log_var, target)
    print(f"gaussian_nll_loss: {loss.item():.4f} (scalar, finite: {torch.isfinite(loss).item()})")
    assert torch.isfinite(loss)

    # Sanity check: given a fixed, real prediction error (mu is off by 1.0),
    # NLL should penalize an overconfident (too-small) log_var more than a
    # well-calibrated one -- claiming near-zero uncertainty while actually
    # wrong by 1.0 should cost more than admitting var~1 matches that error.
    off_by_one_mu = target + 1.0
    calibrated_log_var = torch.zeros_like(target)           # var=1, matches the actual error scale
    overconfident_log_var = torch.full_like(target, -5.0)   # var~0.007, claims near-certainty
    calibrated_loss = gaussian_nll_loss(off_by_one_mu, calibrated_log_var, target)
    overconfident_loss = gaussian_nll_loss(off_by_one_mu, overconfident_log_var, target)
    print(f"well-calibrated log_var loss: {calibrated_loss.item():.4f}")
    print(f"overconfident log_var loss:   {overconfident_loss.item():.4f}  (should be higher)")
    assert overconfident_loss.item() > calibrated_loss.item()
    print("All smoke tests passed.")

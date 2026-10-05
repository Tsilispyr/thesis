# Ablation Matrix Run: cpu_run1

Generated: 2026-08-13T22:07:17
Config: seeds=3 epochs=30 sources=['imu'] quick=False
Models: ['LSTM', 'SSL-LSTM', 'Transformer', 'SSL-Transformer', 'RandomForest', 'GBT']

Full per-epoch console output: `full_console_log.txt`
Raw per-seed numbers: `results.json`
Chart: `ablation_matrix.png`

## Final chained-trajectory error

| Model | Final err [m] | 95% CI | Mean err [m] | 95% CI |
|---|---|---|---|---|
| EKF | 132.47 | n/a (deterministic) | 66.12 | n/a (deterministic) |
| Pure Inertial | 143.00 | n/a (deterministic) | 71.39 | n/a (deterministic) |
| LSTM | 31.71 | [23.23, 40.19] | 20.01 | [16.28, 23.75] |
| SSL-LSTM | 35.43 | [27.84, 43.03] | 22.03 | [17.48, 26.58] |
| Transformer | 8.18 | [0.00, 21.96] | 7.28 | [4.00, 10.56] |
| SSL-Transformer | 14.72 | [3.51, 25.94] | 9.14 | [3.02, 15.26] |
| RandomForest | 26.68 | [25.32, 28.04] | 17.33 | [16.54, 18.13] |
| GBT | 18.73 | [14.04, 23.43] | 12.06 | [9.92, 14.20] |

See `ANALYSIS.md` in this folder for interpretation, if present.

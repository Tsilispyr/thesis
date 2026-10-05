# Ablation Matrix Run: cpu_5seeds

Generated: 2026-08-14T08:14:50
Config: seeds=5 epochs=30 sources=['imu'] quick=False
Models: ['LSTM', 'SSL-LSTM', 'Transformer', 'SSL-Transformer', 'RandomForest', 'GBT']

Full per-epoch console output: `full_console_log.txt`
Raw per-seed numbers: `results.json`
Chart: `ablation_matrix.png`

## Final chained-trajectory error

| Model | Final err [m] | 95% CI | Mean err [m] | 95% CI |
|---|---|---|---|---|
| EKF | 132.47 | n/a (deterministic) | 66.12 | n/a (deterministic) |
| Pure Inertial | 143.00 | n/a (deterministic) | 71.39 | n/a (deterministic) |
| LSTM | 29.68 | [24.82, 34.53] | 18.55 | [15.71, 21.40] |
| SSL-LSTM | 34.57 | [31.50, 37.64] | 21.45 | [19.55, 23.34] |
| Transformer | 9.10 | [2.61, 15.60] | 7.44 | [5.74, 9.14] |
| SSL-Transformer | 13.99 | [8.83, 19.16] | 8.76 | [6.29, 11.24] |
| RandomForest | 26.42 | [25.76, 27.08] | 17.24 | [16.91, 17.57] |
| GBT | 18.56 | [16.87, 20.24] | 11.96 | [11.18, 12.74] |

See `ANALYSIS.md` in this folder for interpretation, if present.

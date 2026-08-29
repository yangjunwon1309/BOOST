# Bottle-open ablation analysis by subject

## Scope and aggregation rule

All logs under `METHOD/SUBJECT/` were re-analysed. Every matched CSV/summary
pair is one independent trial; the numeric suffix in a trial filename is used
only to locate that file's own summary and is never matched across methods or
subjects. The two subjects (`YJW`, `SGH`) are reported separately, with an
additional pooled (`combined`) estimate that weights every complete trial
equally.

Primary comparisons include complete CSV/summary pairs only. The three
incomplete pairs remain visible in `data_quality_audit_by_person.csv` but are
excluded from performance and belief comparison summaries.

## Performance metrics by subject and pooled

| method | group | n | time mean (s) | EE distance mean (m) | EE angle mean (deg) |
| --- | --- | --- | --- | --- | --- |
| PATH | YJW | 8 | 27.08 | 3.214 | 1451.9 |
| PATH | SGH | 2 | 30.63 | 4.194 | 1847.2 |
| PATH | combined | 10 | 27.79 | 3.410 | 1530.9 |
| No jump diffusion | YJW | 3 | 32.70 | 5.015 | 2198.2 |
| No jump diffusion | SGH | 2 | 33.99 | 3.608 | 1716.5 |
| No jump diffusion | combined | 5 | 33.22 | 4.452 | 2005.5 |
| No orientation | YJW | 4 | 43.19 | 6.413 | 2548.4 |
| No orientation | SGH | 6 | 46.54 | 4.681 | 2430.6 |
| No orientation | combined | 10 | 45.20 | 5.374 | 2477.7 |
| Position Only | YJW | 2 | 46.05 | 6.105 | 2490.8 |
| Position Only | SGH | 5 | 40.24 | 4.221 | 1808.2 |
| Position Only | combined | 7 | 41.90 | 4.759 | 2003.2 |
| Alpha blend | YJW | 3 | 28.84 | 4.483 | 1903.5 |
| Alpha blend | SGH | 5 | 37.75 | 3.391 | 1583.9 |
| Alpha blend | combined | 8 | 34.41 | 3.800 | 1703.7 |

![Performance by subject](ablation_performance_by_person.png)

## Ablation change versus PATH

Positive values mean more time, translation, or angular displacement than
PATH. The pooled result uses all complete trials, not an average of the two
subject means.

| method | group | time delta (s) | time delta (%) | distance delta (m) | distance delta (%) | angle delta (deg) | angle delta (%) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| No jump diffusion | combined | 5.42 | 19.5 | 1.042 | 30.6 | 474.5 | 31.0 |
| No orientation | combined | 17.41 | 62.6 | 1.964 | 57.6 | 946.8 | 61.8 |
| Position Only | combined | 14.11 | 50.8 | 1.349 | 39.6 | 472.3 | 30.9 |
| Alpha blend | combined | 6.62 | 23.8 | 0.390 | 11.4 | 172.8 | 11.3 |

The full YJW, SGH, and pooled comparison is available in
`ablation_delta_vs_path.csv`.

## Belief metrics

The following values pool left and right hands within each group. Lower jitter
and command-tracking RMSE are preferable; tracking is unavailable (`N/A`) when
the method does not apply belief position guidance.

| method | group | availability (%) | ID retention (%) | jitter RMSE (mm) | tracking RMSE (mm) |
| --- | --- | --- | --- | --- | --- |
| PATH | YJW | 60.8 | 80.2 | 26.82 | 30.10 |
| PATH | SGH | 75.4 | 73.6 | 28.63 | 40.98 |
| PATH | combined | 63.9 | 78.5 | 27.27 | 32.52 |
| No jump diffusion | YJW | 7.2 | 83.7 | 31.80 | 20.81 |
| No jump diffusion | SGH | 79.1 | 76.1 | 24.16 | 51.72 |
| No jump diffusion | combined | 24.0 | 77.8 | 26.17 | 37.23 |
| No orientation | YJW | 6.9 | 87.0 | 24.36 | 81.92 |
| No orientation | SGH | 54.1 | 80.7 | 23.15 | 51.81 |
| No orientation | combined | 32.5 | 81.4 | 23.31 | 63.10 |
| Position Only | YJW | 13.0 | 83.1 | 21.03 | N/A |
| Position Only | SGH | 76.7 | 84.3 | 27.22 | N/A |
| Position Only | combined | 56.7 | 83.7 | 24.36 | N/A |
| Alpha blend | YJW | 59.1 | 78.3 | 28.04 | N/A |
| Alpha blend | SGH | 73.1 | 85.4 | 25.00 | N/A |
| Alpha blend | combined | 68.1 | 83.2 | 25.92 | N/A |

![Belief metrics by subject](ablation_belief_by_person.png)

## Generated analysis files

- `trial_metrics_by_person.csv`: independent-trial metrics
- `ablation_person_metric_table.csv`: performance mean and sample SD by subject and pooled group
- `ablation_combined_metric_table.csv`: pooled performance table
- `ablation_delta_vs_path.csv`: within-group ablation differences from PATH
- `belief_metrics_by_method_person_hand.csv`: left, right, and combined belief metrics
- `data_quality_audit_by_person.csv`: complete/incomplete pair audit
- `ablation_performance_by_person.png` and `ablation_belief_by_person.png`: visual summaries

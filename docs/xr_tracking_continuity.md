# Galaxy XR tracking loss and recovery

The PATH arm bridge and Revo2 MPPI runner hold their last command during brief
tracking loss, rather than extrapolating unseen hand motion. The hold lasts at
most 1.0 s from the last valid source sample. Older input releases command
ownership; this does not extend or falsify the XR source timestamps. A source
gap above 150 ms or an explicit invalid tracking frame requires a recovery
handoff, even when the apparent average velocity across that gap is small.

For the arms, recovery reuses the existing bounded SE(3) damping. Its current
limits are 0.06 m/s and 0.6 rad/s. Short losses restart from the last commanded
wrist pose. After the hold expires, recovery restarts from fresh robot FK and
resets the IK warm start. Normal tracking retains the existing 5 Hz SE(3)
low-pass. Settings are in `configs/teleop/path_reference.yaml`.

For the fingers, recovery starts at the held command, or measured Revo2 state
after command ownership expires. A 0.10 s exponential filter and a 2.0/s
normalized joint speed limit apply during recovery. Normal tracking has no
low-pass; a generous 5.0/s speed limit only clips large frame-to-frame jumps.
No stale landmark frame is reoptimized. Settings are under
`control.tracking_continuity` in `configs/hands/brainco_revo2_mppi.yaml`.

The current Televuer input commits the hand pair together. A missing hand can
therefore hold both arms and hands. Filtering cannot reconstruct motion that
the headset did not observe, and a long loss can still invoke downstream
command-loss behavior. Test acquisition and occlusion on the robot before
collecting trials.

The recorder stores `xr_tracking_state` in each `left_ipopt_target_json` and
`right_ipopt_target_json`: `tracking`, `recovering`, `hold`, `stale_release`,
or `waiting`. These labels allow datasets to exclude held or recovery
segments. Residual-offset decay is included in `recovering`. Old recordings
have no such label. Hand-MPPI console logs report its own per-hand mode.

## Recorded finger contact criterion

`configs/hands/brainco_revo2.yaml:contact_detection` controls the existing
command-following-error estimator. It observes the actual BrainCo DDS command
and measured state, both normalized to the 0=open / 1=closed range.

| Criterion | Current value |
| --- | --- |
| Error | signed `q_cmd - q_meas` |
| Raw contact | error > 0.08 |
| Confirm contact | error > 0.08 continuously for at least 0.10 s |
| Release contact | error < 0.04 continuously for at least 0.10 s |
| Unavailable | command or state age > 0.25 s; contact bits become null |
| Thumb channel | `thumb_aux`, the proximal flexion channel |
| Other fingers | their corresponding Index/Middle/Ring/Pinky channels |

The 0.04–0.08 band preserves the previous confirmed contact bit. The recorder
stores `q_cmd`, `q_meas`, `dq_meas`, `tau_est`, `err`, `contact_raw`, and
`contact` in each hand JSON. Neither tactile values nor torque are used to
decide these contact bits. Motor lag, joint limits, and fast closure can also
create positive tracking error, so this is a contact estimate rather than a
tactile-confirmed label. The estimator thresholds are unchanged by the XR
continuity change.

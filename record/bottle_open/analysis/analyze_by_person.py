#!/usr/bin/env python3
"""Subject-stratified bottle-open analysis.

Each CSV/summary pair under ``METHOD/SUBJECT`` is one independent trial.  The
trial filename is only used to pair its own CSV and summary; numeric suffixes
are never aligned across methods or subjects.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import analyze_bottle_open as base

METHODS = base.METHOD_ORDER
SUBJECTS = ["YJW", "SGH"]
HANDS = base.HANDS


def values(frame, column):
    return base.numeric_list(frame, column)


def performance_row(method, group, trials):
    row = {"method": method, "group": group, "n_complete_pairs": len(trials), "success_rate_pct": 100.0}
    for source, label, unit in [
        ("elapsed_s", "completion_time", "s"),
        ("ee_distance_left_m", "ee_distance_left", "m"),
        ("ee_distance_right_m", "ee_distance_right", "m"),
        ("ee_distance_total_m", "ee_distance_total", "m"),
        ("ee_angular_left_deg", "ee_angular_left", "deg"),
        ("ee_angular_right_deg", "ee_angular_right", "deg"),
        ("ee_angular_total_deg", "ee_angular_total", "deg"),
    ]:
        x = values(trials, source)
        row[f"{label}_mean_{unit}"] = base.mean(x)
        row[f"{label}_sd_{unit}"] = base.std(x)
    return row


def aggregate_belief_group(method, group, hand, updates, transitions, applied, denominators):
    selected_hands = HANDS if hand == "combined" else [hand]
    filt = lambda x: x[(x["method"] == method) & (x["group"] == group) & (x["hand"].isin(selected_hands))]
    d = [r for r in denominators if r["method"] == method and r["group"] == group and r["hand"] in selected_hands]
    return {"method": method, "group": group, "hand": hand, **base.aggregate_belief(filt(updates), filt(transitions), filt(applied), d)}


def plot_performance(perf, out):
    specs = [
        ("completion_time_mean_s", "Completion time", "s"),
        ("ee_distance_total_mean_m", "Cumulative EE translation", "m"),
        ("ee_angular_total_mean_deg", "Cumulative EE angular displacement", "degrees"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    x = np.arange(len(METHODS)); width = 0.24
    groups = SUBJECTS + ["combined"]
    colors = ["#56B4E9", "#D55E00", "#555555"]
    for ax, (metric, title, unit) in zip(axes, specs):
        for i, (group, color) in enumerate(zip(groups, colors)):
            selected = perf[perf.group == group].set_index("method").reindex(METHODS)
            err = metric.replace("_mean_", "_sd_")
            ax.bar(x + (i-1)*width, selected[metric], width, yerr=selected[err], capsize=3, color=color, label=group, alpha=.82)
        ax.set_title(title); ax.set_ylabel(unit); ax.set_xticks(x, METHODS, rotation=22, ha="right")
        ax.grid(axis="y", alpha=.25); ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(title="group", frameon=False)
    fig.suptitle("Bottle-opening performance: subject-separated and pooled", fontsize=14)
    fig.savefig(out, dpi=220, bbox_inches="tight"); plt.close(fig)


def plot_belief(belief, out):
    metrics = [
        ("belief_availability_pct", "Belief availability", "%"),
        ("target_id_retention_pct", "Target-ID retention", "%"),
        ("same_id_position_step_rmse_mm", "Same-ID position jitter", "mm"),
        ("ee_to_belief_command_position_rmse_mm", "EE-to-belief command tracking", "mm"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    x = np.arange(len(METHODS)); width=.24
    for ax, (metric, title, unit) in zip(axes.flat, metrics):
        for i, (group, color) in enumerate(zip(SUBJECTS + ["combined"], ["#56B4E9", "#D55E00", "#555555"])):
            s = belief[(belief.group == group) & (belief.hand == "combined")].set_index("method").reindex(METHODS)
            ax.bar(x+(i-1)*width, s[metric], width, color=color, alpha=.82, label=group)
        ax.set_title(title); ax.set_ylabel(unit); ax.set_xticks(x, METHODS, rotation=22, ha="right")
        ax.grid(axis="y", alpha=.25); ax.spines[["top", "right"]].set_visible(False)
    axes[0,0].legend(title="group", frameon=False)
    fig.suptitle("Belief metrics (left + right pooled within each group)", fontsize=14)
    fig.savefig(out, dpi=220, bbox_inches="tight"); plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(); root = args.root.resolve(); out = (args.output or root / "analysis").resolve(); out.mkdir(parents=True, exist_ok=True)
    csvs = {p.relative_to(root).with_suffix("").as_posix(): p for p in root.glob("*/*/trial_*.csv")}
    summaries = {p.relative_to(root).as_posix().removesuffix(".summary.json"): p for p in root.glob("*/*/trial_*.summary.json")}
    keys = sorted(set(csvs) | set(summaries), key=lambda k: (METHODS.index(k.split('/')[0]), k.split('/')[1], k.split('/')[2]))
    trials=[]; quality=[]; updates_all=[]; transitions_all=[]; applied_all=[]; denominators=[]
    for key in keys:
        method, subject, trial = key.split("/")
        csv_path, summary_path = csvs.get(key), summaries.get(key)
        summary = base.safe_json(summary_path.read_text(encoding="utf-8")) if summary_path else None
        data = base.read_csv_log(csv_path, method, trial) if csv_path else None
        status = "complete_pair" if csv_path and summary_path else ("csv_only_incomplete" if csv_path else "summary_only")
        included = status == "complete_pair"; se = base.finite(summary.get("elapsed_s")) if isinstance(summary, dict) else None
        oe = base.finite(data.get("observed_elapsed_s")) if data else None
        sd = base.finite(summary.get("ee_distance_total_m")) if isinstance(summary, dict) else None
        cd = sum(data["distance_increment"].values()) if data else None
        trial_id = f"{method}/{subject}/{trial}"
        trials.append({"method":method,"subject":subject,"trial":trial,"trial_id":trial_id,"status":status,"primary_included":included,"success_pct":100.0,"samples_csv":data["rows"] if data else math.nan,"samples_summary":summary.get("samples") if isinstance(summary,dict) else math.nan,"elapsed_s":se if se is not None else oe,"observed_csv_elapsed_s":oe,"ee_distance_left_m":base.finite(summary.get("ee_distance_left_m")) if isinstance(summary,dict) else (data["distance_increment"]["left"] if data else math.nan),"ee_distance_right_m":base.finite(summary.get("ee_distance_right_m")) if isinstance(summary,dict) else (data["distance_increment"]["right"] if data else math.nan),"ee_distance_total_m":sd if sd is not None else cd,"ee_angular_left_deg":data["angular_path_deg"]["left"] if data else math.nan,"ee_angular_right_deg":data["angular_path_deg"]["right"] if data else math.nan,"ee_angular_total_deg":sum(data["angular_path_deg"].values()) if data else math.nan})
        quality.append({"method":method,"subject":subject,"trial":trial,"status":status,"primary_included":included,"csv_present":bool(csv_path),"summary_present":bool(summary_path),"summary_basename_matches":Path(summary.get("csv", "")).name == csv_path.name if csv_path and isinstance(summary,dict) else math.nan,"notes":"Excluded from primary comparison: incomplete pair" if not included else ""})
        if not data: continue
        duration = se if se is not None else oe
        for hand in HANDS:
            denominators.append({"method":method,"group":subject,"trial_id":trial_id,"hand":hand,"unique_belief_sequences":data["unique_belief_sequences"],"valid_ee_samples":data["valid_ee_samples"][hand]})
            for r in data["belief_updates"][hand]:
                r.update(subject=subject, group=subject, trial_id=trial_id, primary_included=included); updates_all.append(r)
            for r in base.build_transitions(data["belief_updates"][hand], se, oe):
                r.update(subject=subject, group=subject, trial_id=trial_id, primary_included=included); transitions_all.append(r)
        for r in data["applied_records"]:
            r.update(subject=subject, group=subject, trial_id=trial_id, primary_included=included); applied_all.append(r)
    trial_df=pd.DataFrame(trials); quality_df=pd.DataFrame(quality); updates=pd.DataFrame(updates_all); transitions=pd.DataFrame(transitions_all); applied=pd.DataFrame(applied_all)
    # Analyses use only complete pairs.  Add a pooled group that treats every trial equally.
    updates=updates[updates.primary_included].copy(); transitions=transitions[transitions.primary_included].copy(); applied=applied[applied.primary_included].copy()
    pooled=[]
    for df in [updates, transitions, applied]:
        if not df.empty:
            z=df.copy(); z["group"]="combined"; pooled.append(z)
        else: pooled.append(df)
    updates_all_groups=pd.concat([updates,pooled[0]], ignore_index=True); transitions_all_groups=pd.concat([transitions,pooled[1]], ignore_index=True); applied_all_groups=pd.concat([applied,pooled[2]], ignore_index=True)
    denom_all=denominators + [{**r, "group":"combined"} for r in denominators if trial_df.loc[trial_df.trial_id == r["trial_id"], "primary_included"].iloc[0]]
    included_trials=trial_df[trial_df.primary_included].copy(); perf=[]
    for method in METHODS:
        for group in SUBJECTS + ["combined"]:
            t=included_trials[(included_trials.method==method) & ((included_trials.subject==group) if group != "combined" else pd.Series(True,index=included_trials.index))]
            perf.append(performance_row(method, group, t))
    perf=pd.DataFrame(perf)
    belief=[]
    for method in METHODS:
        for group in SUBJECTS + ["combined"]:
            for hand in HANDS+["combined"]: belief.append(aggregate_belief_group(method, group, hand, updates_all_groups, transitions_all_groups, applied_all_groups, denom_all))
    belief=pd.DataFrame(belief)
    # Explicit ablation effect versus PATH within the same subject/pooled group.
    delta_rows=[]
    compare_metrics=["completion_time_mean_s", "ee_distance_total_mean_m", "ee_angular_total_mean_deg"]
    for group in SUBJECTS + ["combined"]:
        reference=perf[(perf.method=="PATH") & (perf.group==group)].iloc[0]
        for _, row in perf[perf.group==group].iterrows():
            result={"method":row.method,"group":group,"n_complete_pairs":int(row.n_complete_pairs),"reference_method":"PATH"}
            for metric in compare_metrics:
                delta=row[metric]-reference[metric]
                result[f"{metric}_delta_vs_path"]=delta
                result[f"{metric}_pct_vs_path"]=100.0*delta/reference[metric] if reference[metric] else math.nan
            delta_rows.append(result)
    delta_df=pd.DataFrame(delta_rows)
    trial_df.to_csv(out/"trial_metrics_by_person.csv",index=False,float_format="%.6f")
    perf.to_csv(out/"ablation_person_metric_table.csv",index=False,float_format="%.6f")
    perf[perf.group=="combined"].to_csv(out/"ablation_combined_metric_table.csv",index=False,float_format="%.6f")
    delta_df.to_csv(out/"ablation_delta_vs_path.csv",index=False,float_format="%.6f")
    belief.to_csv(out/"belief_metrics_by_method_person_hand.csv",index=False,float_format="%.6f")
    quality_df.to_csv(out/"data_quality_audit_by_person.csv",index=False,float_format="%.6f")
    plot_performance(perf,out/"ablation_performance_by_person.png"); plot_belief(belief,out/"ablation_belief_by_person.png")
    report = "# Bottle-open ablation analysis by subject\n\n"
    report += "Each matched CSV/summary pair is an independent trial. Trial-number suffixes are not matched or compared across methods/subjects. Primary tables exclude incomplete pairs.\n\n"
    report += "## Performance metrics by subject and pooled\n\n" + base.markdown_table(perf, [("method","method",None),("group","group",None),("n_complete_pairs","n",0),("completion_time_mean_s","time mean (s)",2),("ee_distance_total_mean_m","EE distance mean (m)",3),("ee_angular_total_mean_deg","EE angle mean (deg)",1)])
    report += "\n\n![Performance](ablation_performance_by_person.png)\n\n## Belief metrics (complete pairs)\n\n"
    report += "## Ablation change versus PATH\n\n" + base.markdown_table(delta_df, [("method","method",None),("group","group",None),("completion_time_mean_s_delta_vs_path","time Δ (s)",2),("completion_time_mean_s_pct_vs_path","time Δ (%)",1),("ee_distance_total_mean_m_delta_vs_path","distance Δ (m)",3),("ee_distance_total_mean_m_pct_vs_path","distance Δ (%)",1),("ee_angular_total_mean_deg_delta_vs_path","angle Δ (deg)",1),("ee_angular_total_mean_deg_pct_vs_path","angle Δ (%)",1)])
    report += "\n\n## Belief metrics (complete pairs)\n\n"
    report += base.markdown_table(belief[belief.hand=="combined"], [("method","method",None),("group","group",None),("belief_availability_pct","availability (%)",1),("target_id_retention_pct","ID retention (%)",1),("same_id_position_step_rmse_mm","jitter RMSE (mm)",2),("ee_to_belief_command_position_rmse_mm","tracking RMSE (mm)",2)])
    report += "\n\n![Belief](ablation_belief_by_person.png)\n\n"
    report += "Detailed left/right/combined belief metrics are in `belief_metrics_by_method_person_hand.csv`.\n"
    (out/"analysis_report_by_person.md").write_text(report,encoding="utf-8")
    print(f"Analyzed {len(trial_df)} independent trials; {len(included_trials)} complete pairs.")

if __name__ == "__main__": main()

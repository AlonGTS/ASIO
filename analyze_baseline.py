#!/usr/bin/env python3
"""
Summarize a baseline CSV log written by tracker-so.py (see
TRACKER_ACCURACY_ROADMAP.md, "Step A" / "Experiment 1").

Usage:
    python3 analyze_baseline.py                      # latest logs/baseline_*.csv
    python3 analyze_baseline.py logs/baseline_20260806_104131.csv
    python3 analyze_baseline.py --jump-threshold 20
"""
import argparse
import csv
import statistics as st
from pathlib import Path

_HERE = Path(__file__).parent


def _latest_baseline_csv():
    candidates = sorted((_HERE / "logs").glob("baseline_*.csv"))
    if not candidates:
        raise SystemExit("No logs/baseline_*.csv found — run tracker-so.py with "
                          "[logging] baseline_enabled = true first.")
    return candidates[-1]


def _float(row, key):
    v = row[key]
    return float(v) if v not in ("", None) else None


def analyze(path: Path, jump_threshold: float):
    rows = list(csv.DictReader(open(path)))
    if not rows:
        print(f"{path}: empty log")
        return

    n = len(rows)
    sessions = sorted(set(r["session_id"] for r in rows), key=int)
    success_rows = [r for r in rows if r["success"] == "1"]
    lost_rows    = [r for r in rows if r["success"] == "0"]
    drift_events = [r for r in rows if r["drift_event"] == "1"]

    update_ms = [_float(r, "update_ms") for r in rows if _float(r, "update_ms") is not None]
    tq        = [_float(r, "tq_score") for r in success_rows if _float(r, "tq_score") is not None]
    dist      = [_float(r, "center_dist") for r in success_rows if _float(r, "center_dist") is not None]
    fps       = [_float(r, "inst_fps") for r in rows if _float(r, "inst_fps") is not None]

    print(f"file: {path}")
    print(f"rows: {n}   sessions (re-acquisitions): {sessions}")
    print(f"success frames: {len(success_rows)} ({100*len(success_rows)/n:.1f}%)   "
          f"lost frames: {len(lost_rows)} ({100*len(lost_rows)/n:.1f}%)")
    print(f"drift events: {len(drift_events)}")
    print()

    if update_ms:
        print(f"update_ms   mean={st.mean(update_ms):.2f}  median={st.median(update_ms):.2f}  "
              f"max={max(update_ms):.2f}  stdev={st.pstdev(update_ms):.2f}")
    if fps:
        print(f"inst_fps    mean={st.mean(fps):.2f}  min={min(fps):.2f}")
    if tq:
        print(f"tq_score    mean={st.mean(tq):.3f}  min={min(tq):.3f}")
    if dist:
        dist_sorted = sorted(dist)
        p95 = dist_sorted[int(0.95 * len(dist_sorted))]
        print(f"center_dist mean={st.mean(dist):.2f}px  stdev={st.pstdev(dist):.2f}px  "
              f"max={max(dist):.2f}px  p95={p95:.2f}px")
        jumps = sum(1 for d in dist if d > jump_threshold)
        print(f"frames w/ jump > {jump_threshold}px: {jumps} ({100*jumps/len(dist):.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", help="Path to a baseline_*.csv (default: latest in logs/)")
    parser.add_argument("--jump-threshold", type=float, default=15.0,
                         help="Center-jump distance (MAIN px) above which a frame counts as a 'jump' (default: 15.0)")
    args = parser.parse_args()

    csv_path = Path(args.path) if args.path else _latest_baseline_csv()
    analyze(csv_path, args.jump_threshold)

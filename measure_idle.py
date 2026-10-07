#!/usr/bin/env python3
"""Measure the idle power baseline and record a warm-up temperature curve.

Everything downstream subtracts this, so it runs before any measured sweep.

    cd ~/projects/energy-cost
    source ~/dp-energy/bin/activate
    python3 measure_idle.py

Leave the machine alone while this runs. An SSH session sitting idle is fine , 
that is the same condition sweeps run under, but do not browse, compile, or
run anything else.

Writes: idle_baseline.json, idle_provenance.json
Optional: --duration SECONDS (default 300), --skip-warmup, --no-drift-check
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import paper_utils_energy as pe

import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")


def banner(t):
    print(f"\n{'=' * 62}\n{t}\n{'=' * 62}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=300.0,
                    help="idle measurement window in seconds (default 300)")
    ap.add_argument("--warmup", type=float, default=60.0,
                    help="GPU warm-up seconds before measuring (default 60)")
    ap.add_argument("--skip-warmup", action="store_true")
    ap.add_argument("--no-drift-check", action="store_true",
                    help="skip the split-half stability check")
    ap.add_argument("--out", default="idle_baseline.json")
    args = ap.parse_args()

    # ---------------------------------------------------------------- checks
    banner("0. Instrumentation check")
    pkgs = pe.list_rapl_packages()
    if not pkgs:
        sys.exit("FAIL: no readable RAPL package. Check the sudoers rule.")
    print(f"  RAPL packages: {pkgs}")
    print(f"  GPU energy counter: {pe.gpu_energy_counter_available()}")
    print(f"  GPU temp now: {pe.read_gpu_temp_c():.0f} C")

    # ---------------------------------------------------------------- warmup
    # Idle measured from a cold GPU understates the idle floor that real runs
    # actually sit at, because runs are warm. Warm first, then measure.
    if not args.skip_warmup:
        banner(f"1. Warm-up ({args.warmup:.0f}s)")
        t_before = pe.read_gpu_temp_c()
        pe.warmup(args.warmup)
        t_after = pe.read_gpu_temp_c()
        print(f"  GPU {t_before:.0f} C -> {t_after:.0f} C")
        print("  Settling 30s before measuring idle...")
        time.sleep(30)
    else:
        banner("1. Warm-up skipped")

    # ------------------------------------------------------------------ idle
    banner(f"2. Idle measurement ({args.duration:.0f}s), do not touch the machine")
    print(f"  Started {time.strftime('%H:%M:%S')}, "
          f"finishes ~{time.strftime('%H:%M:%S', time.localtime(time.time() + args.duration))}")

    with pe.EnergyMeter(poll_interval_s=1.0) as em:
        time.sleep(args.duration)
    r = em.reading

    idle = pe.IdleBaseline(
        gpu_power_w=r.gpu_joules / r.elapsed_s,
        cpu_power_w=r.cpu_joules / r.elapsed_s,
        duration_s=r.elapsed_s,
        n_samples=r.n_samples,
        gpu_temp_c=r.gpu_temp_end_c,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )

    print(f"\n  GPU idle : {idle.gpu_power_w:7.2f} W   ({r.gpu_joules:.1f} J over {r.elapsed_s:.0f}s)")
    print(f"  CPU idle : {idle.cpu_power_w:7.2f} W   ({r.cpu_joules:.1f} J)")
    print(f"  Total    : {idle.gpu_power_w + idle.cpu_power_w:7.2f} W")
    print(f"  Samples  : {r.n_samples}   RAPL wraps: {r.n_rapl_wraps}")
    print(f"  GPU temp : {r.gpu_temp_start_c:.0f} -> {r.gpu_temp_end_c:.0f} C")
    print(f"  GPU method: {r.gpu_method}")

    # --------------------------------------------------------------- sanity
    banner("3. Sanity checks")

    # NVML instantaneous reading vs the counter-derived average: a large gap
    # means one of the two is not measuring what we think it is.
    inst = pe.read_gpu_power_w()
    gap = abs(inst - idle.gpu_power_w)
    print(f"  [{'PASS' if gap < 5 else 'WARN'}] instantaneous {inst:.2f} W vs "
          f"measured mean {idle.gpu_power_w:.2f} W (gap {gap:.2f} W)")

    if r.gpu_method != "counter":
        print("  [WARN] GPU fell back to power integration, flag in methods")
    else:
        print("  [PASS] GPU used the NVML energy counter")

    expected_wraps = int(r.elapsed_s / 2621) if r.elapsed_s > 2621 else 0
    print(f"  [INFO] RAPL wraps: {r.n_rapl_wraps} "
          f"(≈{expected_wraps} expected at 100 W; idle draws less, so fewer is normal)")

    if idle.cpu_power_w <= 0 or idle.gpu_power_w <= 0:
        print("  [FAIL] non-positive idle power, instrumentation problem, do not proceed")

    # Drift check: split the power trace in half. A rising second half means the
    # machine had not settled and the baseline is biased low.
    if not args.no_drift_check and len(em.power_trace) > 20:
        w = [p for _, p in em.power_trace]
        h1 = sum(w[:len(w)//2]) / (len(w)//2)
        h2 = sum(w[len(w)//2:]) / (len(w) - len(w)//2)
        drift = abs(h2 - h1)
        pct = 100 * drift / h1 if h1 else 0
        flag = "PASS" if pct < 5 else "WARN"
        print(f"  [{flag}] GPU power drift across window: "
              f"{h1:.2f} -> {h2:.2f} W ({pct:.1f}%)")
        if pct >= 5:
            print("         Machine had not settled. Re-run, or lengthen --warmup.")

    # ------------------------------------------------------------------ save
    idle.save(args.out)
    prov = pe.measurement_provenance()
    prov["idle_baseline"] = {
        "gpu_power_w": idle.gpu_power_w,
        "cpu_power_w": idle.cpu_power_w,
        "duration_s": idle.duration_s,
        "n_samples": idle.n_samples,
        "warmup_s": 0.0 if args.skip_warmup else args.warmup,
        "gpu_method": r.gpu_method,
    }
    Path("idle_provenance.json").write_text(json.dumps(prov, indent=2))

    banner("DONE")
    print(f"  Wrote {args.out} and idle_provenance.json")
    print(f"\n  Subtraction rule: E_attributable = E_measured − "
          f"{idle.gpu_power_w + idle.cpu_power_w:.2f} W × T_run")
    print("\n  Next: re-validated smoke test on PTB-XL.")


if __name__ == "__main__":
    main()

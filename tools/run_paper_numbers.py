"""Recompute every number the paper reports from the sweep CSVs.

Reads results/sweep/*.csv, writes results/paper_numbers/{table2,table3}.csv
and numbers.json, and checks each value against the figure printed in the
paper. Exit status is non-zero if any check fails, so this doubles as a test.

    python3 tools/run_paper_numbers.py

Paper: Liquidating the Fund: Differential Privacy and the Accounting of an
Exhaustible Stock (Section 6, Tables 2 and 3).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
SWEEP = ROOT / "results" / "sweep"
OUT = ROOT / "results" / "paper_numbers"
OUT.mkdir(parents=True, exist_ok=True)

# Values stated in the paper, checked below. Tolerances are the paper's own
# rounding, so a check fails only if the recomputed value would print differently.
PAPER = {
    "baseline_kj": 22.0,
    "dp_mean_kj": 43.4,
    "overhead_mean_pct": 97.6,
    "overhead_min_pct": 95.4,
    "overhead_max_pct": 99.2,
    "auc_baseline": 0.976,
    "auc_eps025": 0.551,
    "auc_eps5": 0.787,
    "step_025_to_1": 0.128,
    "step_4_to_5": 0.010,
    "paired_drop_5_4_min": 0.006,
    "paired_drop_5_4_max": 0.013,
    "paired_drop_5_3_min": 0.017,
    "paired_drop_5_3_max": 0.032,
    "slope_all_runs": 0.13,
    "ci_all_runs": (-0.21, 0.48),
    "p_all_runs": 0.44,
    "slope_grid": 0.13,
    "ci_grid": (-0.37, 0.64),
    "p_grid": 0.56,
    "tost_grid_p": 0.002,
    "cv_max_pct": 1.5,
    "cv_other_max_pct": 0.7,
    "eps_realized_total": 57.6,
    "eps_declared_total": 57.75,
    "eps_audit_total": 21.0,
    "canary_unprotected_min": 0.43,
    "canary_unprotected_max": 0.61,
    "canary_protected_zero_of_nine": 7,
    "canary_protected_max": 0.09,
    "audit_cost_over_train": 2.0,
    "lira_unprotected_eps": 0.12,
    "lira_models": 64,
    "reserve20_now": 0.777, "reserve20_later": 0.679, "reserve20_drop": 0.010,
    "reserve40_now": 0.763, "reserve40_later": 0.736, "reserve40_drop": 0.025,
    "delta": 4.6e-5,
    "n_train": 8709,
}

fails = []


def check(name, value, expect, tol):
    ok = abs(value - expect) <= tol
    print(f"  [{'ok' if ok else 'FAIL'}] {name}: {value:.4g} (paper {expect})")
    if not ok:
        fails.append(name)


def ols(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float); n = len(x)
    X = np.column_stack([np.ones(n), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    res = y - X @ beta
    s2 = res @ res / (n - 2)
    se = np.sqrt((s2 * np.linalg.inv(X.T @ X))[1, 1])
    b = beta[1]
    t = stats.t.ppf(0.975, n - 2)
    p = 2 * (1 - stats.t.cdf(abs(b / se), n - 2))
    tost = max(stats.t.cdf((b - 1) / se, n - 2), 1 - stats.t.cdf((b + 1) / se, n - 2))
    return dict(slope=b, se=se, lo=b - t * se, hi=b + t * se, p=p, tost_p=tost)


# ---------------------------------------------------------------- Table 2
t = pd.read_csv(SWEEP / "targets_summary.csv")
base = t[t.epsilon.isna()]
dp = t[t.epsilon.notna()].copy()
base_j = base.total_j_net.mean()
dp["overhead_pct"] = (dp.total_j_net / base_j - 1) * 100

g = dp.groupby("epsilon")
table2 = pd.DataFrame({
    "energy_kj": g.total_j_net.mean() / 1e3,
    "energy_sd_kj": g.total_j_net.std() / 1e3,
    "overhead_pct": g.overhead_pct.mean(),
    "test_auc": g.test_auc.mean(),
    "auc_sd": g.test_auc.std(),
    "cv_pct": g.total_j_net.std() / g.total_j_net.mean() * 100,
    "eps_realized": g.epsilon_spent.mean(),
})
base_row = pd.DataFrame({"energy_kj": [base_j / 1e3], "energy_sd_kj": [base.total_j_net.std() / 1e3],
                         "overhead_pct": [np.nan], "test_auc": [base.test_auc.mean()],
                         "auc_sd": [base.test_auc.std()], "cv_pct": [np.nan], "eps_realized": [np.nan]},
                        index=pd.Index(["non-private"], name="epsilon"))
pd.concat([base_row, table2]).round(4).to_csv(OUT / "table2.csv")

print("Table 2")
check("baseline kJ", base_j / 1e3, PAPER["baseline_kj"], 0.05)
check("DP mean kJ", dp.total_j_net.mean() / 1e3, PAPER["dp_mean_kj"], 0.05)
check("overhead mean %", table2.overhead_pct.mean(), PAPER["overhead_mean_pct"], 0.05)
check("overhead min %", table2.overhead_pct.min(), PAPER["overhead_min_pct"], 0.05)
check("overhead max %", table2.overhead_pct.max(), PAPER["overhead_max_pct"], 0.05)
check("AUC baseline", base.test_auc.mean(), PAPER["auc_baseline"], 0.0005)
check("AUC eps 0.25", table2.test_auc[0.25], PAPER["auc_eps025"], 0.0005)
check("AUC eps 5.0", table2.test_auc[5.0], PAPER["auc_eps5"], 0.0005)
check("step 0.25->1.0", table2.test_auc[1.0] - table2.test_auc[0.25], PAPER["step_025_to_1"], 0.0005)
check("step 4.0->5.0", table2.test_auc[5.0] - table2.test_auc[4.0], PAPER["step_4_to_5"], 0.0005)
check("CV at eps 0.25 %", table2.cv_pct[0.25], PAPER["cv_max_pct"], 0.05)
cv_other = table2.cv_pct.drop(0.25).max()
print(f"  [{'ok' if cv_other < PAPER['cv_other_max_pct'] else 'FAIL'}] CV elsewhere below {PAPER['cv_other_max_pct']}%: max {cv_other:.2f}")
if cv_other >= PAPER["cv_other_max_pct"]: fails.append("cv_other")
assert (dp.epsilon_spent <= dp.epsilon + 1e-9).all(), "realized eps exceeded target"
check("eps realized total", dp.epsilon_spent.sum(), PAPER["eps_realized_total"], 0.05)
check("eps declared total", dp.epsilon.sum(), PAPER["eps_declared_total"], 0.005)

# paired seed differences
pv = dp.pivot_table(index="seed", columns="epsilon", values="test_auc")
d54 = pv[5.0] - pv[4.0]; d53 = pv[5.0] - pv[3.0]
check("paired drop 5->4 min", d54.min(), PAPER["paired_drop_5_4_min"], 0.0005)
check("paired drop 5->4 max", d54.max(), PAPER["paired_drop_5_4_max"], 0.0005)
check("paired drop 5->3 min", d53.min(), PAPER["paired_drop_5_3_min"], 0.0005)
check("paired drop 5->3 max", d53.max(), PAPER["paired_drop_5_3_max"], 0.0005)

# regressions
print("Regression of overhead on epsilon")
r_all = ols(dp.epsilon, dp.overhead_pct)
check("slope, all runs", r_all["slope"], PAPER["slope_all_runs"], 0.005)
check("CI lo, all runs", r_all["lo"], PAPER["ci_all_runs"][0], 0.005)
check("CI hi, all runs", r_all["hi"], PAPER["ci_all_runs"][1], 0.005)
check("p, all runs", r_all["p"], PAPER["p_all_runs"], 0.005)
print(f"  [{'ok' if r_all['tost_p'] < 0.001 else 'FAIL'}] equivalence p < 0.001, all runs: {r_all['tost_p']:.2e}")
if r_all["tost_p"] >= 0.001: fails.append("tost_all")
r_grid = ols(table2.index, table2.overhead_pct)
check("slope, grid means", r_grid["slope"], PAPER["slope_grid"], 0.005)
check("CI lo, grid means", r_grid["lo"], PAPER["ci_grid"][0], 0.005)
check("CI hi, grid means", r_grid["hi"], PAPER["ci_grid"][1], 0.005)
check("p, grid means", r_grid["p"], PAPER["p_grid"], 0.005)
check("equivalence p, grid means", r_grid["tost_p"], PAPER["tost_grid_p"], 0.0005)
per_seed = {int(s): ols(gs.epsilon, gs.overhead_pct)["slope"] for s, gs in dp.groupby("seed")}
print(f"  per-seed slopes: {', '.join(f'{v:.2f}' for v in per_seed.values())} (paper 0.31, 0.11, -0.02)")

# ---------------------------------------------------------------- Table 3
print("Table 3")
a = table2.test_auc
table3 = pd.DataFrame([
    dict(rule="Spend all now", eps_now=5.0, eps_reserve=0.0, auc_now=a[5.0], auc_later=np.nan),
    dict(rule="20% reserve", eps_now=4.0, eps_reserve=1.0, auc_now=a[4.0], auc_later=a[1.0]),
    dict(rule="40% reserve", eps_now=3.0, eps_reserve=2.0, auc_now=a[3.0], auc_later=a[2.0]),
])
table3.round(3).to_csv(OUT / "table3.csv", index=False)
check("20% reserve AUC now", a[4.0], PAPER["reserve20_now"], 0.0005)
check("20% reserve AUC later", a[1.0], PAPER["reserve20_later"], 0.0005)
check("20% reserve drop", a[5.0] - a[4.0], PAPER["reserve20_drop"], 0.0005)
check("40% reserve AUC now", a[3.0], PAPER["reserve40_now"], 0.0005)
check("40% reserve AUC later", a[2.0], PAPER["reserve40_later"], 0.0005)
check("40% reserve drop", a[5.0] - a[3.0], PAPER["reserve40_drop"], 0.0005)
print(f"  delta per model {PAPER['delta']:.1e}, two-model rule {2*PAPER['delta']:.1e}; 1/n_train = {1/PAPER['n_train']:.2e}")
assert PAPER["delta"] < 1 / PAPER["n_train"]

# ---------------------------------------------------------------- Audit
print("Audit (Section 6)")
c = pd.read_csv(SWEEP / "canary_summary_ep30.csv")
unp = c[c.epsilon.isna()]; prot = c[c.epsilon.notna()]
check("canary unprotected min", unp.epsilon_emp.min(), PAPER["canary_unprotected_min"], 0.005)
check("canary unprotected max", unp.epsilon_emp.max(), PAPER["canary_unprotected_max"], 0.005)
check("protected runs with zero bound", int((prot.epsilon_emp == 0).sum()), PAPER["canary_protected_zero_of_nine"], 0)
check("protected max bound", prot.epsilon_emp.max(), PAPER["canary_protected_max"], 0.005)
check("audit eps total", prot.epsilon_spent.sum(), PAPER["eps_audit_total"], 0.05)
audit_over_train = unp.total_j_net.mean() / base_j
check("audit cost / training run", audit_over_train, PAPER["audit_cost_over_train"], 0.05)
at = pd.read_csv(SWEEP / "attacks_summary.csv")
lira = at[(at.attack == "lira") & at.epsilon.isna()]
check("LiRA reference models", lira.n_models.iloc[0], PAPER["lira_models"], 0)
check("LiRA eps bound, unprotected", lira.epsilon_emp.mean(), PAPER["lira_unprotected_eps"], 0.005)

json.dump({**{k: (list(v) if isinstance(v, tuple) else v) for k, v in PAPER.items()},
           "recomputed": {"regression_all_runs": r_all, "regression_grid_means": r_grid,
                          "per_seed_slopes": per_seed, "audit_over_train": audit_over_train}},
          open(OUT / "numbers.json", "w"), indent=2, default=float)

print(f"\n{len(fails)} check(s) failed" if fails else "\nAll checks passed")
sys.exit(1 if fails else 0)

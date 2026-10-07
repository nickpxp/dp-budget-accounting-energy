"""Figures for Liquidating the Fund. Figure 1: fund conversion schematic. Figure 2: energy and AUC by size of draw (Table 2 values)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from scipy import stats

TEAL = "#1F6F78"; TEAL_L = "#BFE0E3"; AMBER = "#C8792B"; AMBER_L = "#F3D9BF"
INK = "#222222"; GREY = "#9A9A9A"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK,
                     "axes.linewidth": 0.8, "pdf.fonttype": 42})

# ---------------------------------------------------------------- Figure 1
draws = [0.17, 0.10, 0.23, 0.08, 0.15, 0.27]
n = len(draws); xmax = n + 2.6
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)
fig.subplots_adjust(wspace=0.12, left=0.09, right=0.98, top=0.86, bottom=0.2)

def frame(ax, title):
    ax.set_xlim(0, xmax); ax.set_ylim(0, 1.32)
    ax.set_title(title, loc="left", fontsize=9.5, fontweight="bold", color=INK, pad=8)
    ax.set_xticks([]); ax.set_yticks([0, 1]); ax.set_yticklabels(["0", "full"])
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.set_xlabel("successive queries", color=INK)

# left: fund. capacity stays full; each query draws service and leaves capacity intact
frame(ax1, "Fund as recorded: no formal guarantee")
ax1.fill_between([0, xmax], 0, 1, color=TEAL_L, lw=0)
ax1.plot([0, xmax], [1, 1], color=TEAL, lw=2.2)
for i in range(n + 2):
    x = i + 0.6
    alpha = 1.0 if i < n else 0.45
    ax1.annotate("", xy=(x, 1.02), xytext=(x, 1.24),
                 arrowprops=dict(arrowstyle="-|>", color=AMBER, lw=1.6, alpha=alpha, mutation_scale=11))
ax1.text(0.6, 1.27, "service drawn", color=AMBER, fontsize=8, va="bottom")
ax1.text(0.6, 0.5, "recorded capacity\nunchanged after every query", color=TEAL, fontsize=8.5, va="center")
ax1.annotate("", xy=(xmax, 1.0), xytext=(xmax - 0.9, 1.0),
             arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=2.2, mutation_scale=13))
ax1.set_ylabel("remaining capacity", color=INK)

# right: converted fund. each query removes what it draws; capacity reaches zero inside the frame
frame(ax2, "Converted: each draw counted")
cap = 1.0; x = 0.0; steps = []
for i, d in enumerate(draws):
    x0, x1 = x, x + 1.0
    ax2.fill_between([x0, x1], 0, cap, color=TEAL_L, lw=0)
    ax2.add_patch(Rectangle((x0 + 0.15, cap - d), 0.7, d, facecolor=AMBER_L, edgecolor=AMBER, lw=1.0, hatch="////"))
    ax2.annotate("", xy=(x0 + 0.5, cap - d + 0.005), xytext=(x0 + 0.5, cap + 0.16),
                 arrowprops=dict(arrowstyle="-|>", color=AMBER, lw=1.6, mutation_scale=11))
    steps.append((x0, x1, cap))
    cap -= d; x = x1
cap = max(cap, 0)
# staircase line
px, py = [], []
for x0, x1, c in steps:
    px += [x0, x1]; py += [c, c]
ax2.plot(px, py, color=TEAL, lw=2.2)
ax2.plot([x, x], [steps[-1][2] - draws[-1], 0], color=TEAL, lw=2.2)
ax2.plot([x, xmax], [0, 0], color=TEAL, lw=2.2, ls=(0, (3, 2)))
# after exhaustion: queries arrive, nothing served
for i in range(2):
    xq = x + 0.6 + i
    ax2.annotate("", xy=(xq, 0.03), xytext=(xq, 0.24),
                 arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.4, ls="--", mutation_scale=11))
ax2.plot([x, x], [0, 1.18], color=INK, lw=0.8, ls=":")  # stop below the label
ax2.text(x + 0.12, 0.62, "budget\nexhausted", fontsize=8.5, color=INK, va="center")
ax2.text(x + 0.12, 0.34, "queries arrive,\nno service", fontsize=8, color=GREY, va="center")
ax2.text(0.25, 1.27, "amount drawn, chosen by the operator", color=AMBER, fontsize=8, va="bottom")
ax2.text(0.35, 0.28, "remaining\ncapacity", color=TEAL, fontsize=8.5, va="center")
fig.savefig("figure1.pdf"); fig.savefig("figure1.png", dpi=600)
plt.close(fig)

# ---------------------------------------------------------------- Figure 2
eps = np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 5.0])
energy = np.array([42.9, 43.3, 43.2, 43.5, 43.7, 43.5, 43.4, 43.4, 43.4, 43.4])
esd = np.array([0.65, 0.17, 0.14, 0.21, 0.22, 0.12, 0.28, 0.07, 0.15, 0.03])
over = np.array([95.4, 97.3, 96.7, 98.2, 99.2, 98.3, 97.8, 97.6, 97.6, 97.5])
osd = esd / 22.0 * 100
auc = np.array([0.551, 0.611, 0.651, 0.679, 0.699, 0.714, 0.736, 0.763, 0.778, 0.787])
asd = np.array([0.034, 0.032, 0.029, 0.027, 0.026, 0.025, 0.023, 0.020, 0.017, 0.014])

# OLS with t-based band on 8 df
X = np.column_stack([np.ones_like(eps), eps])
beta, *_ = np.linalg.lstsq(X, over, rcond=None)
resid = over - X @ beta; s2 = resid @ resid / 8
cov = s2 * np.linalg.inv(X.T @ X)
xx = np.linspace(0.2, 5.3, 200); Xx = np.column_stack([np.ones_like(xx), xx])
fit = Xx @ beta; se = np.sqrt(np.einsum("ij,jk,ik->i", Xx, cov, Xx))
t = stats.t.ppf(0.975, 8)
print("slope %.3f  CI [%.3f, %.3f]" % (beta[1], beta[1] - t*np.sqrt(cov[1,1]), beta[1] + t*np.sqrt(cov[1,1])))
xbar, ybar = eps.mean(), over.mean()

fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.3))
fig.subplots_adjust(left=0.09, right=0.98, top=0.95, bottom=0.17, wspace=0.28)
for ax in (a, b):
    ax.set_xscale("log"); ax.set_xticks(eps); ax.set_xticklabels(["0.25","0.5","","1","","1.5","2","3","4","5"])
    ax.tick_params(which="minor", bottom=False); ax.grid(axis="y", color="#E6E6E6", lw=0.6)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.set_xlabel("size of draw, ε")
a.fill_between(xx, ybar - 1*(xx - xbar), ybar + 1*(xx - xbar), color="#DDDDDD", lw=0, alpha=0.7)
a.fill_between(xx, fit - t*se, fit + t*se, color=AMBER_L, lw=0, alpha=0.8)
a.plot(xx, fit, color=AMBER, lw=1.4, ls="--")
a.errorbar(eps, over, yerr=osd, fmt="o", color=TEAL, ecolor=TEAL, capsize=3, ms=5, lw=1.4)
a.plot(eps, over, color=TEAL, lw=1.0)
a.set_ylim(90, 105); a.set_ylabel("energy overhead of the mechanism (%)")
a.text(0.97, 0.04, "dashed: least-squares fit; band: 95% CI\ngray: ±1 point per unit ε equivalence margin",
       transform=a.transAxes, ha="right", va="bottom", fontsize=7.2, color=INK)
b.axhline(0.976, color=INK, lw=1.0, ls=(0, (2, 2))); b.text(5.2, 0.984, "no mechanism (0.976)", ha="right", fontsize=7.8)
b.axhline(0.5, color=GREY, lw=0.8, ls=":"); b.text(5.2, 0.51, "chance", ha="right", fontsize=7.8, color=GREY)
b.errorbar(eps, auc, yerr=asd, fmt="s", color=TEAL, ecolor=TEAL, capsize=3, ms=4.5, lw=1.4)
b.plot(eps, auc, color=TEAL, lw=1.0)
b.set_ylim(0.45, 1.02); b.set_ylabel("test AUC")
fig.savefig("figure2.pdf"); fig.savefig("figure2.png", dpi=600)

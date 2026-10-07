

"""run_full_sweep.py

The measured sweep: 33 target models, the shadow pool, the population attacks,
and the assembled tables.

    python3 run_full_sweep.py --out-dir results/sweep_new

Stages (all run by default; --stages to select, one comma-separated token):
  targets   33 target models (10 eps x 3 seeds + 3 baseline), energy measured
  shadows   284 shadow models in two tiers, energy measured per model
  attacks   loss-threshold, RMIA, LiRA over every target
  results   assembled tables

"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent))
import paper_utils_dpcnn as pu
import paper_utils_energy as pe

pe.silence_opacus_hook_warnings()
log = pu.setup_logging("sweep")

# --- Design constants ------------------------------------------------------
EPSILON_GRID = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 5.0]
ANCHOR_EPSILONS = [None, 0.5, 1.5, 5.0]      # None = non-private ceiling
N_SHADOWS_ANCHOR = 64
N_SHADOWS_COVERAGE = 4
N_SEEDS = 3
EPOCHS = 30
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
MAX_GRAD_NORM = 1.0
SHADOW_FRACTION = 0.5
RMIA_N_REFERENCES = 4
RMIA_GAMMA = 1.0


def banner(t):
    log.info("")
    log.info("=" * 70)
    log.info(t)
    log.info("=" * 70)


def shadow_mask(idx, n_pool, fraction, base_seed=0):
    """Deterministic ~fraction subset of the pool for shadow model `idx`."""
    rng = np.random.default_rng(base_seed + 10_000 + idx)
    n_in = int(round(n_pool * fraction))
    chosen = rng.choice(n_pool, size=n_in, replace=False)
    m = np.zeros(n_pool, dtype=bool)
    m[chosen] = True
    return m


def eps_tag(eps):
    return "baseline" if eps is None else f"eps{eps}"


@torch.no_grad()
def logits_for(model, X, device, batch_size=512):
    model.eval()
    ds = TensorDataset(torch.from_numpy(X).float())
    return np.concatenate([model(xb.to(device)).cpu().numpy()
                           for (xb,) in DataLoader(ds, batch_size=batch_size)])


def prob_true_class(logits, labels):
    """Pr(x | theta) on the true class for a single-logit binary head."""
    p = 1.0 / (1.0 + np.exp(-logits))
    return np.where(labels == 1, p, 1.0 - p)


# ===========================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=None,
                    help="default: sweep_YYYYMMDD (dated, per manual 16.1)")
    ap.add_argument("--stages", default="targets,shadows,attacks,results")
    ap.add_argument("--idle", default="idle_baseline.json")
    ap.add_argument("--skip-warmup", action="store_true")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()

    stages = [s.strip() for s in args.stages.split(",")]
    out = Path(args.out_dir or f"sweep_{time.strftime('%Y%m%d')}")
    (out / "targets").mkdir(parents=True, exist_ok=True)
    (out / "shadows").mkdir(parents=True, exist_ok=True)

    device = pu.get_device()
    idle = pe.IdleBaseline.load(args.idle)

    banner("SETUP")
    log.info(f"  out dir : {out}")
    log.info(f"  stages  : {stages}")
    log.info(f"  device  : {device}")
    log.info(f"  idle    : {idle.gpu_power_w:.2f} W GPU + {idle.cpu_power_w:.2f} W CPU")

    prov = pe.measurement_provenance()
    prov.update(dict(epsilon_grid=EPSILON_GRID, anchor_epsilons=[str(e) for e in ANCHOR_EPSILONS],
                     n_shadows_anchor=N_SHADOWS_ANCHOR, n_shadows_coverage=N_SHADOWS_COVERAGE,
                     n_seeds=N_SEEDS, epochs=args.epochs, batch_size=BATCH_SIZE,
                     shadow_fraction=SHADOW_FRACTION,
                     idle_gpu_w=idle.gpu_power_w, idle_cpu_w=idle.cpu_power_w))
    (out / "provenance.json").write_text(json.dumps(prov, indent=2, default=str))

    # --- data ---------------------------------------------------------------
    data = pu.load_ptbxl_cache("data/ptbxl_raw_100hz.npz")
    splits = pu.canonical_split(data)
    pool = splits["train"]
    n_pool = len(pool)

    mean, std = pu.fit_normalizer(data.X[pool])
    X_pool = pu.apply_normalizer(data.X[pool], mean, std)
    y_pool = data.y[pool]
    X_val = pu.apply_normalizer(data.X[splits["val"]], mean, std)
    y_val = data.y[splits["val"]]
    X_test = pu.apply_normalizer(data.X[splits["test"]], mean, std)
    y_test = data.y[splits["test"]]

    prev = float(y_pool.mean())
    pos_weight_full = (1.0 - prev) / prev
    log.info(f"  pool {n_pool}, prevalence {prev:.4f}")

    def mk_loader(X, y, shuffle=False):
        return DataLoader(TensorDataset(torch.from_numpy(X).float(),
                                        torch.from_numpy(y).float()),
                          batch_size=BATCH_SIZE, shuffle=shuffle)

    val_loader = mk_loader(X_val, y_val)
    test_loader = mk_loader(X_test, y_test)

    if not args.skip_warmup:
        pe.warmup(60.0)
        time.sleep(10)

    # --- shared trainer -----------------------------------------------------
    def train_one(mask, eps, seed, ckpt):
        """Train one model on `mask` of the pool. Returns (result, energy|None)."""
        if ckpt.exists():
            meta = json.loads(ckpt.with_suffix(".pt.json").read_text())
            return meta, None                      # cached: no energy available

        Xin, yin = X_pool[mask], y_pool[mask]
        p = float(yin.mean())
        cfg = pu.TrainConfig(
            epochs=args.epochs, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE,
            max_grad_norm=MAX_GRAD_NORM, target_epsilon=eps,
            pos_weight=(1 - p) / p if 0 < p < 1 else pos_weight_full, seed=seed)

        torch.manual_seed(seed)
        np.random.seed(seed)
        model = pu.InceptionTime1D()

        with pe.EnergyMeter() as em:
            res = pu.train(model, mk_loader(Xin, yin, shuffle=True),
                           val_loader, test_loader, cfg, device=device)
        pu.save_checkpoint(res, model, ckpt)
        return res, em.reading

    def energy_row(reading):
        if reading is None:
            return dict(gpu_j=np.nan, cpu_j=np.nan, total_j=np.nan,
                        gpu_j_net=np.nan, cpu_j_net=np.nan, total_j_net=np.nan,
                        wall_s=np.nan, mean_gpu_w=np.nan, gpu_temp_end=np.nan,
                        rapl_wraps=np.nan, energy_measured=False)
        net = reading.net_of_idle(idle)
        return dict(gpu_j=reading.gpu_joules, cpu_j=reading.cpu_joules,
                    total_j=reading.total_joules,
                    gpu_j_net=net["gpu_joules_net"], cpu_j_net=net["cpu_joules_net"],
                    total_j_net=net["total_joules_net"],
                    wall_s=reading.elapsed_s, mean_gpu_w=reading.mean_gpu_power_w,
                    gpu_temp_end=reading.gpu_temp_end_c,
                    rapl_wraps=reading.n_rapl_wraps, energy_measured=True)

    t_start = time.time()

    # =======================================================================
    # STAGE 1: TARGETS
    # =======================================================================
    if "targets" in stages:
        banner("STAGE 1: TARGET MODELS")
        settings = [None] + EPSILON_GRID
        rows, n = [], len(settings) * N_SEEDS
        i = 0
        # Targets train on the same fixed half-pool as shadows, so membership
        # ground truth is exact and comparable across every attack.
        target_mask = shadow_mask(999_999, n_pool, SHADOW_FRACTION)
        np.save(out / "target_mask.npy", target_mask)

        for eps in settings:
            for seed in range(N_SEEDS):
                i += 1
                tag = f"target_{eps_tag(eps)}_seed{seed:02d}"
                ckpt = out / "targets" / f"{tag}.pt"
                t0 = time.time()
                res, reading = train_one(target_mask, eps, seed, ckpt)

                cached = reading is None
                auc = res["test_auc"] if cached else res.test_auc
                spent = (res.get("epsilon_spent") if cached else res.epsilon_spent)
                log.info(f"  [{i}/{n}] {tag}: auc={auc:.4f}"
                         + (f" eps_spent={spent:.3f}" if spent else "")
                         + ("  (cached)" if cached else
                            f"  {(time.time()-t0)/60:.1f} min  "
                            f"{reading.total_joules/1000:.1f} kJ"))

                rows.append(dict(kind="target", epsilon=eps, seed=seed, tag=tag,
                                 test_auc=auc, epsilon_spent=spent,
                                 **energy_row(reading)))

        pd.DataFrame(rows).to_csv(out / "targets_summary.csv", index=False)
        log.info(f"  wrote targets_summary.csv  ({(time.time()-t_start)/3600:.2f} h elapsed)")

    # =======================================================================
    # STAGE 2: SHADOWS (two tiers)
    # =======================================================================
    if "shadows" in stages:
        banner("STAGE 2: SHADOW MODELS")
        plan = ([(e, N_SHADOWS_ANCHOR, "anchor") for e in ANCHOR_EPSILONS] +
                [(e, N_SHADOWS_COVERAGE, "coverage")
                 for e in EPSILON_GRID if e not in ANCHOR_EPSILONS])
        total = sum(k for _, k, _ in plan)
        log.info(f"  {total} shadows: "
                 f"{len(ANCHOR_EPSILONS)}x{N_SHADOWS_ANCHOR} anchor + "
                 f"{len(plan)-len(ANCHOR_EPSILONS)}x{N_SHADOWS_COVERAGE} coverage")

        rows, i = [], 0
        for eps, k, tier in plan:
            for s in range(k):
                i += 1
                tag = f"shadow_{eps_tag(eps)}_{s:03d}"
                ckpt = out / "shadows" / f"{tag}.pt"
                mask = shadow_mask(s, n_pool, SHADOW_FRACTION)
                t0 = time.time()
                res, reading = train_one(mask, eps, s, ckpt)
                cached = reading is None
                auc = res["test_auc"] if cached else res.test_auc

                if i % 10 == 0 or not cached:
                    log.info(f"  [{i}/{total}] {tag} ({tier}): auc={auc:.4f}"
                             + ("  (cached)" if cached else
                                f"  {(time.time()-t0)/60:.1f} min"))

                np.save(out / "shadows" / f"{tag}.mask.npy", mask)
                rows.append(dict(kind="shadow", tier=tier, epsilon=eps,
                                 shadow_idx=s, tag=tag, test_auc=auc,
                                 **energy_row(reading)))

        pd.DataFrame(rows).to_csv(out / "shadows_summary.csv", index=False)
        log.info(f"  wrote shadows_summary.csv  ({(time.time()-t_start)/3600:.2f} h elapsed)")

    # =======================================================================
    # STAGE 3: ATTACKS
    # =======================================================================
    if "attacks" in stages:
        banner("STAGE 3: ATTACKS")
        target_mask = np.load(out / "target_mask.npy")
        is_member = target_mask
        shadows_df = pd.read_csv(out / "shadows_summary.csv")

        def load_model(path):
            m, _ = pu.load_checkpoint(path, device=device)
            return m

        # Shadow probabilities are computed once per epsilon and reused by both
        # RMIA and LiRA; energy is attributed per attack in the results stage.
        rows = []
        for eps in [None] + EPSILON_GRID:
            et = eps_tag(eps)
            sh = shadows_df[shadows_df.epsilon.isna() if eps is None
                            else shadows_df.epsilon == eps]
            if len(sh) == 0:
                log.info(f"  {et}: no shadows, skipping")
                continue

            log.info(f"  {et}: loading {len(sh)} shadows")
            with pe.EnergyMeter() as em_shadow_inf:
                sp, masks = [], []
                for tag in sh.tag:
                    sm = load_model(out / "shadows" / f"{tag}.pt")
                    sp.append(prob_true_class(logits_for(sm, X_pool, device), y_pool))
                    masks.append(np.load(out / "shadows" / f"{tag}.mask.npy"))
                    del sm
                    torch.cuda.empty_cache()
            sp = np.stack(sp)
            masks = np.stack(masks)

            for seed in range(N_SEEDS):
                tag = f"target_{et}_seed{seed:02d}"
                ck = out / "targets" / f"{tag}.pt"
                if not ck.exists():
                    continue
                tm = load_model(ck)
                p_target = prob_true_class(logits_for(tm, X_pool, device), y_pool)
                del tm
                torch.cuda.empty_cache()

                # --- 1. Yeom loss-threshold: zero additional models ----------
                with pe.EnergyMeter() as em:
                    s_loss = p_target
                rows.append(dict(attack="loss_threshold", epsilon=eps, seed=seed,
                                 n_models=0, **score_metrics(s_loss, is_member),
                                 **energy_row(em.reading)))

                # --- 2. RMIA: N reference models -----------------------------
                n_ref = min(RMIA_N_REFERENCES, len(sp))
                with pe.EnergyMeter() as em:
                    pr = sp[:n_ref].mean(axis=0)
                    s_rmia = (p_target / np.maximum(pr, 1e-7)) / RMIA_GAMMA
                rows.append(dict(attack="rmia", epsilon=eps, seed=seed,
                                 n_models=n_ref, **score_metrics(s_rmia, is_member),
                                 **energy_row(em.reading)))

                # --- 3. LiRA: full shadow set, per-record IN/OUT gaussians ---
                if len(sp) >= 16:
                    with pe.EnergyMeter() as em:
                        s_lira = lira_scores(p_target, sp, masks)
                    rows.append(dict(attack="lira", epsilon=eps, seed=seed,
                                     n_models=len(sp),
                                     **score_metrics(s_lira, is_member),
                                     **energy_row(em.reading)))

                log.info(f"    {tag}: "
                         + "  ".join(f"{r['attack']}={r['auc']:.4f}"
                                     for r in rows[-3:]))

        pd.DataFrame(rows).to_csv(out / "attacks_summary.csv", index=False)
        log.info(f"  wrote attacks_summary.csv  ({(time.time()-t_start)/3600:.2f} h elapsed)")

    # =======================================================================
    # STAGE 4: RESULTS
    # =======================================================================
    if "results" in stages:
        banner("STAGE 4: RESULTS")
        assemble_results(out, idle)

    banner("DONE")
    log.info(f"  total wall {(time.time()-t_start)/3600:.2f} h")
    log.info(f"  outputs in {out}/")


# ---------------------------------------------------------------------------

def score_metrics(scores, is_member):
    """Attack-power and empirical-epsilon metrics for one score vector."""
    eps_e = pe.empirical_epsilon(scores, is_member, delta=pu.DEFAULT_DELTA)
    return dict(auc=pu.mia_auc(scores, is_member),
                tpr_at_fpr_001=pu.mia_tpr_at_fpr(scores, is_member, 0.001),
                tpr_at_fpr_01=pu.mia_tpr_at_fpr(scores, is_member, 0.01),
                epsilon_emp=eps_e.epsilon_emp,
                eps_tpr=eps_e.tpr_raw, eps_fpr=eps_e.fpr_raw)


def lira_scores(p_target, shadow_probs, masks, eps=1e-7):
    """Online LiRA: per-record IN/OUT gaussians over shadow confidences.

    Uses logit-scaled confidences ; score is the
    likelihood ratio of the target's confidence under the two fitted normals.
    """
    from scipy.stats import norm

    def logit(p):
        p = np.clip(p, eps, 1 - eps)
        return np.log(p / (1 - p))

    sl = logit(shadow_probs)          # (n_shadows, n_records)
    tl = logit(p_target)              # (n_records,)

    scores = np.zeros_like(tl)
    for j in range(sl.shape[1]):
        in_v = sl[masks[:, j], j]
        out_v = sl[~masks[:, j], j]
        if len(in_v) < 2 or len(out_v) < 2:
            scores[j] = 0.0
            continue
        mu_i, sd_i = in_v.mean(), in_v.std() + eps
        mu_o, sd_o = out_v.mean(), out_v.std() + eps
        scores[j] = (norm.logpdf(tl[j], mu_i, sd_i)
                     - norm.logpdf(tl[j], mu_o, sd_o))
    return scores


def assemble_results(out: Path, idle):
    """Join energy and attack results into the paper's headline tables."""
    tg = pd.read_csv(out / "targets_summary.csv")
    sh = pd.read_csv(out / "shadows_summary.csv")
    at = pd.read_csv(out / "attacks_summary.csv")

    citable = bool(tg.energy_measured.all() and sh.energy_measured.all())
    if not citable:
        log.info("  WARNING: some runs were cached, energy incomplete.")
        log.info("           Re-run in a fresh directory before reporting these numbers.")

    # --- Defensive expenditure: DP overhead vs non-private baseline ---------
    base = tg[tg.epsilon.isna()].total_j_net.mean()
    de = (tg[tg.epsilon.notna()].groupby("epsilon")
          .agg(energy_j=("total_j_net", "mean"),
               energy_sd=("total_j_net", "std"),
               test_auc=("test_auc", "mean"),
               wall_s=("wall_s", "mean")).reset_index())
    de["baseline_j"] = base
    de["overhead_pct"] = 100 * (de.energy_j - base) / base
    de["kwh"] = de.energy_j / 3.6e6
    de.to_csv(out / "table_defensive_expenditure.csv", index=False)

    log.info("\n  DEFENSIVE EXPENDITURE (gate 1: kill if < 10%)")
    log.info(f"    baseline {base/1000:.1f} kJ")
    for _, r in de.iterrows():
        log.info(f"    eps={r.epsilon:<5} {r.energy_j/1000:7.1f} kJ  "
                 f"{r.overhead_pct:+7.1f}%   auc={r.test_auc:.4f}")
    gate1 = de.overhead_pct.max() >= 10
    log.info(f"    -> gate 1 {'PASS' if gate1 else 'FAIL: no measurable overhead'}")

    # --- Audit cost: shadow energy charged per configuration ---------------
    # Functional unit is one verified privacy claim per configuration, so the
    # shadow pool is charged once per epsilon, not once per seed (decision 8).
    shadow_cost = (sh.groupby("epsilon", dropna=False)
                   .agg(shadow_j=("total_j_net", "sum"),
                        n_shadows=("tag", "count")).reset_index())
    shadow_cost["per_shadow_j"] = shadow_cost.shadow_j / shadow_cost.n_shadows

    at2 = at.merge(shadow_cost, on="epsilon", how="left")
    # Each attack pays only for the models it consumes.
    at2["attack_energy_j"] = at2.total_j_net + at2.n_models * at2.per_shadow_j
    train_cost = tg.groupby("epsilon", dropna=False).total_j_net.mean().rename("train_j")
    at2 = at2.merge(train_cost, on="epsilon", how="left")
    at2["audit_over_train"] = at2.attack_energy_j / at2.train_j
    at2["eps_per_joule"] = at2.epsilon_emp / at2.attack_energy_j
    at2.to_csv(out / "table_audit_cost.csv", index=False)

    log.info("\n  AUDIT COST AND YIELD (by attack, averaged over seeds)")
    summ = (at2.groupby("attack")
            .agg(models=("n_models", "mean"),
                 energy_kj=("attack_energy_j", lambda s: s.mean() / 1000),
                 x_training=("audit_over_train", "mean"),
                 auc=("auc", "mean"),
                 tpr001=("tpr_at_fpr_001", "mean"),
                 eps_emp=("epsilon_emp", "mean")).reset_index())
    for _, r in summ.iterrows():
        log.info(f"    {r.attack:15s} {r.models:6.0f} models  "
                 f"{r.energy_kj:9.1f} kJ  {r.x_training:7.1f}x training  "
                 f"auc={r.auc:.4f}  eps_emp={r.eps_emp:.4f}")
    summ.to_csv(out / "table_attack_comparison.csv", index=False)

    # --- Gate 2 and the fork condition -------------------------------------
    log.info("\n  GATES")
    max_eps = at2.epsilon_emp.max()
    log.info(f"    gate 2 (eps_emp degeneracy): max eps_emp = {max_eps:.4f}")
    log.info("      " + ("PASS, measurable" if max_eps >= 0.01 else
                         "FAIL: empirical epsilon degenerate; the finding is "
                         "'expenditure with unverifiable return'"))

    lt = at2[at2.attack == "loss_threshold"].epsilon_emp.mean()
    lr = at2[at2.attack == "lira"].epsilon_emp.mean()
    if lr and not np.isnan(lr) and lr > 0:
        gap = 100 * abs(lt - lr) / lr
        log.info(f"    fork (cheap vs expensive): loss_threshold within {gap:.1f}% of LiRA")
        log.info("      " + ("-> framing: THE CHEAP AUDIT SUFFICES" if gap < 10 else
                             "-> framing: VERIFICATION IS EXPENSIVE"))

    (out / "citable.json").write_text(json.dumps(
        {"citable": citable, "gate1_pass": bool(gate1),
         "gate2_max_eps_emp": float(max_eps)}, indent=2))
    log.info(f"\n  Tables written to {out}/")


if __name__ == "__main__":
    main()

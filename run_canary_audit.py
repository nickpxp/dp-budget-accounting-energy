"""run_canary_audit.py

One-run canary audit of the sweep's trained models, with energy measured.

The population attacks (loss-threshold, RMIA, LiRA) ask how much a real record
leaks on average. The canary audit asks whether the implementation leaks at
all in the worst case. Generate m canaries, include each independently with
probability 1/2, train once, score every canary, guess the k most and least
suspicious as in and out, and convert the number of correct guesses into an
epsilon lower bound with a binomial tail test. Independent inclusion is what
makes one training run sufficient.

    python3 run_canary_audit.py --sweep results/sweep --canary-kind flipped --epochs 30

One training run per configuration, about 2 training runs including the
scoring pass, against 64 shadow models for LiRA.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import binom
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent))
import paper_utils_dpcnn as pu
import paper_utils_energy as pe

pe.silence_opacus_hook_warnings()
log = pu.setup_logging("canary")

ANCHOR_EPSILONS = [None, 0.5, 1.5, 5.0]
N_CANARIES = 1000
INCLUSION_P = 0.5
N_SEEDS = 3
EPOCHS = 30
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
MAX_GRAD_NORM = 1.0


# Canary construction


def make_canaries(n, signal_len, n_leads, kind="noise", rng=None, X_real=None,
                  y_real=None):
    """Build n canary records designed to be maximally detectable.

    Shape follows the cache layout, (n, n_leads, signal_len), so canaries can be
    concatenated onto the training pool directly.

    Canaries are deliberately atypical so that a model which memorizes anything
    memorizes these first. Three constructions, in rough order of how far they
    sit from the data manifold:

    noise      Gaussian noise waveforms with random labels. The ECG analogue of
               the random/dirac canaries used in the one-run literature. Furthest
               off-manifold, so most detectable, but least like a real record.
    flipped    Real waveforms with flipped labels. On-manifold but contradictory,
               so the model must memorize to fit them. Closer to a realistic
               worst-case record.
    extreme    Real waveforms scaled to implausible amplitude, label kept. Tests
               whether magnitude alone drives detectability.

    Returns (X, y). The `kind` choice is a stated design decision, not a default:
    it sets how worst-case the "worst case" actually is, and the reported epsilon
    lower bound is only as tight as the canaries are detectable.
    """
    rng = rng or np.random.default_rng(0)

    if kind == "noise":
        X = rng.standard_normal((n, n_leads, signal_len)).astype(np.float32)
        y = rng.integers(0, 2, size=n).astype(np.int64)

    elif kind == "flipped":
        if X_real is None:
            raise ValueError("flipped canaries need X_real/y_real")
        idx = rng.choice(len(X_real), size=n, replace=False)
        X = X_real[idx].copy()
        y = (1 - y_real[idx]).astype(np.int64)

    elif kind == "extreme":
        if X_real is None:
            raise ValueError("extreme canaries need X_real/y_real")
        idx = rng.choice(len(X_real), size=n, replace=False)
        X = (X_real[idx] * rng.uniform(5, 10, size=(n, 1, 1))).astype(np.float32)
        y = y_real[idx].astype(np.int64)

    else:
        raise ValueError(f"unknown canary kind {kind!r}")

    return X, y


# One-run audit scoring

def one_run_epsilon_lower_bound(n_correct, n_guesses, delta, n_canaries,
                            alpha=0.05, eps_max=20.0, grid=2000):
    """Epsilon lower bound from correct guesses.

    Under (eps, delta)-DP the number of correct guesses W out of r guesses
    satisfies

        Pr[W >= v] <= beta + 2*m*delta*alpha,
        beta = Pr[Bin(r, e^eps / (e^eps + 1)) >= v]

    so an eps is ruled out at level alpha when beta <= alpha*(1 - 2*m*delta).
    The bound is the largest eps that is still ruled out; beta increases
    monotonically in eps, so the sweep can stop at the first eps that survives.

    Returns 0.0 when nothing can be ruled out. That is the honest answer for an
    audit that found no signal, not a failure to compute.
    """
    if n_guesses <= 0:
        return 0.0

    # Budget left for the binomial tail after the delta correction. With many
    # canaries and a loose delta this can go non-positive, in which case no
    # bound is certifiable at any eps and the audit is uninformative by
    # construction, worth checking before spending the compute.
    beta_budget = alpha * (1.0 - 2.0 * n_canaries * delta)
    if beta_budget <= 0:
        return 0.0

    lower = 0.0
    for eps in np.linspace(0.0, eps_max, grid):
        p = np.exp(eps) / (np.exp(eps) + 1.0)
        beta = float(binom.sf(n_correct - 1, n_guesses, p))
        if beta <= beta_budget:
            lower = float(eps)              # this eps is ruled out; bound rises
        else:
            break                           # beta grows with eps; stop here
    return lower


def max_informative_canaries(delta, alpha=0.05):
    """Largest canary count for which the delta correction leaves headroom.

    Beyond this the correction alone exceeds the confidence level and no bound
    is certifiable regardless of how well the attack performs.
    """
    return int(np.floor(1.0 / (2.0 * delta)))


def audit_from_scores(scores, included, delta, n_canaries, alpha=0.05,
                      guess_fractions=(0.01, 0.02, 0.05, 0.1, 0.2, 0.5),
                      correct_multiplicity=True):
    """Guess the most and least suspicious canaries, then bound epsilon.

    Abstention is the point: guessing only the extremes trades coverage for
    accuracy. The audit sweeps several guess budgets and keeps the best bound,
    which is standard practice, but keeping the best of several is itself a
    multiple comparison, so alpha is divided across the budgets. Without this
    the reported bound holds well below its nominal confidence and pure noise
    yields spurious non-zero epsilon.
    """
    alpha_eff = (alpha / len(guess_fractions)) if correct_multiplicity else alpha
    order = np.argsort(scores)              # ascending: low score = likely OUT
    best = dict(epsilon_emp=0.0, n_guesses=0, n_correct=0,
                guess_fraction=float("nan"), accuracy=float("nan"),
                alpha_eff=alpha_eff)

    for frac in guess_fractions:
        k = max(1, int(len(scores) * frac / 2))     # k each end
        low, high = order[:k], order[-k:]
        # high score -> guess IN, low score -> guess OUT
        correct = int((~included[low]).sum() + included[high].sum())
        n_g = 2 * k
        eps = one_run_epsilon_lower_bound(correct, n_g, delta, n_canaries, alpha_eff)
        if eps > best["epsilon_emp"]:
            best = dict(epsilon_emp=eps, n_guesses=n_g, n_correct=correct,
                        guess_fraction=float(frac), accuracy=correct / n_g,
                        alpha_eff=alpha_eff)
    return best




@torch.no_grad()
def canary_scores(model, Xc, yc, device, batch=256):
    """Per-canary membership score: confidence on the assigned label.

    Higher means the model fits that canary better, so it is more likely to have
    been trained on it.
    """
    model.eval()
    ds = TensorDataset(torch.from_numpy(Xc).float())
    logits = np.concatenate([model(xb.to(device)).cpu().numpy()
                             for (xb,) in DataLoader(ds, batch_size=batch)])
    p = 1.0 / (1.0 + np.exp(-logits))
    return np.where(yc == 1, p, 1.0 - p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True, help="existing sweep directory")
    # No default: running without this flag once produced a whole noise-canary
    # audit that looked like the intended flipped one in every artifact except
    # the log banner. Make the choice explicit.
    ap.add_argument("--canary-kind", required=True,
                    choices=["noise", "flipped", "extreme"])
    ap.add_argument("--n-canaries", type=int, default=N_CANARIES)
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--idle", default="idle_baseline.json")
    ap.add_argument("--skip-warmup", action="store_true")
    args = ap.parse_args()

    out = Path(args.sweep)
    # Epoch-tagged so re-running at a different training length cannot
    # overwrite a prior arm. The 100-epoch and 30-epoch canary audits are
    # different experiments and both are reportable.
    tag_suffix = f"_ep{args.epochs}"
    cdir = out / f"canary{tag_suffix}"
    cdir.mkdir(parents=True, exist_ok=True)
    device = pu.get_device()
    idle = pe.IdleBaseline.load(args.idle)

    log.info("=" * 70)
    log.info("CANARY AUDIT (one training run)")
    log.info("=" * 70)
    log.info(f"  canaries : {args.n_canaries} ({args.canary_kind}), "
             f"inclusion p={INCLUSION_P}")
    log.info(f"  epochs   : {args.epochs}"
             + ("" if args.epochs == EPOCHS else
                f"  (NOTE: sweep targets trained at {EPOCHS}; this arm is not "
                f"length-matched to them)"))
    log.info(f"  epsilons : {ANCHOR_EPSILONS}")
    log.info(f"  seeds    : {args.seeds}")
    log.info(f"  cost     : 1 training run per (eps, seed)")

    # --- data ---------------------------------------------------------------
    data = pu.load_ptbxl_cache("data/ptbxl_raw_100hz.npz")
    splits = pu.canonical_split(data)
    pool = splits["train"]
    mean, std = pu.fit_normalizer(data.X[pool])
    X_pool = pu.apply_normalizer(data.X[pool], mean, std)
    y_pool = data.y[pool]
    X_val = pu.apply_normalizer(data.X[splits["val"]], mean, std)
    X_test = pu.apply_normalizer(data.X[splits["test"]], mean, std)

    val_loader = DataLoader(TensorDataset(
        torch.from_numpy(X_val).float(),
        torch.from_numpy(data.y[splits["val"]]).float()), batch_size=BATCH_SIZE)
    test_loader = DataLoader(TensorDataset(
        torch.from_numpy(X_test).float(),
        torch.from_numpy(data.y[splits["test"]]).float()), batch_size=BATCH_SIZE)

    prev = float(y_pool.mean())
    pos_weight = (1.0 - prev) / prev

    # Match the sweep's target training-set size. Sweep targets trained on half
    # the pool; training canary models on the full pool would make their energy
    # reflect data volume rather than audit cost, and the cost ladder would be
    # comparing different things.
    tm_path = out / "target_mask.npy"
    if tm_path.exists():
        tm = np.load(tm_path)
        X_pool, y_pool = X_pool[tm], y_pool[tm]
        prev = float(y_pool.mean())
        pos_weight = (1.0 - prev) / prev
        log.info(f"  size-matched to sweep targets: {tm.sum()} records")
    else:
        log.info(f"  WARNING: {tm_path} not found, training on the full pool. "
                 f"Energy will NOT be comparable to the sweep's attacks.")

    if not args.skip_warmup:
        pe.warmup(60.0)
        time.sleep(10)

    rows = []
    total = len(ANCHOR_EPSILONS) * args.seeds
    i = 0

    for eps in ANCHOR_EPSILONS:
        for seed in range(args.seeds):
            i += 1
            tag = f"canary_{'baseline' if eps is None else f'eps{eps}'}_seed{seed:02d}"
            rng = np.random.default_rng(1000 + seed)

            # Shape is taken from the pool rather than from constants, so a
            # cache stored as (N, leads, time) or (N, time, leads) both work.
            n_leads, signal_len = X_pool.shape[1], X_pool.shape[2]
            Xc, yc = make_canaries(args.n_canaries, signal_len, n_leads,
                                   args.canary_kind, rng, X_pool, y_pool)
            assert Xc.shape[1:] == X_pool.shape[1:], (
                f"canary shape {Xc.shape[1:]} != pool shape {X_pool.shape[1:]}")
            included = rng.random(args.n_canaries) < INCLUSION_P

            X_train = np.concatenate([X_pool, Xc[included]])
            y_train = np.concatenate([y_pool, yc[included]])
            log.info(f"\n  [{i}/{total}] {tag}: {included.sum()} canaries in, "
                     f"{(~included).sum()} out")

            cfg = pu.TrainConfig(
                epochs=args.epochs, batch_size=BATCH_SIZE,
                learning_rate=LEARNING_RATE, max_grad_norm=MAX_GRAD_NORM,
                target_epsilon=eps, pos_weight=pos_weight, seed=seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = pu.InceptionTime1D()

            loader = DataLoader(TensorDataset(
                torch.from_numpy(X_train).float(),
                torch.from_numpy(y_train).float()),
                batch_size=BATCH_SIZE, shuffle=True)

            with pe.EnergyMeter() as em:
                res = pu.train(model, loader, val_loader, test_loader,
                               cfg, device=device)
                scores = canary_scores(model, Xc, yc, device)
                audit = audit_from_scores(scores, included, pu.DEFAULT_DELTA,
                                          args.n_canaries)
            r = em.reading
            net = r.net_of_idle(idle)

            log.info(f"      auc={res.test_auc:.4f}  "
                     f"{r.elapsed_s:.0f}s  {net['total_joules_net']/1000:.1f} kJ")
            log.info(f"      guessed {audit['n_guesses']}, correct "
                     f"{audit['n_correct']} ({audit['accuracy']:.3f}) "
                     f"-> eps_emp={audit['epsilon_emp']:.4f}")

            np.savez_compressed(cdir / f"{tag}.scores.npz",
                                scores=scores, included=included)
            pu.save_checkpoint(res, model, cdir / f"{tag}.pt")

            rows.append(dict(
                attack="canary_oneruntrain", epsilon=eps, seed=seed,
                epochs=args.epochs,
                canary_kind=args.canary_kind, n_canaries=args.n_canaries,
                n_included=int(included.sum()),
                test_auc=res.test_auc, epsilon_spent=res.epsilon_spent,
                n_models=1,                       # one training run
                total_j_net=net["total_joules_net"],
                gpu_j_net=net["gpu_joules_net"], cpu_j_net=net["cpu_joules_net"],
                wall_s=r.elapsed_s, energy_measured=True,
                **audit))
            del model
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(out / f"canary_summary{tag_suffix}.csv", index=False)

    # --- summary ------------------------------------------------------------
    log.info("\n" + "=" * 70)
    log.info("CANARY AUDIT RESULTS")
    log.info("=" * 70)
    g = df.groupby("epsilon", dropna=False).agg(
        eps_emp=("epsilon_emp", "mean"), eps_emp_max=("epsilon_emp", "max"),
        acc=("accuracy", "mean"), kJ=("total_j_net", lambda s: s.mean() / 1000),
        auc=("test_auc", "mean"))
    log.info("\n" + g.round(4).to_string())

    log.info(f"\n  Max epsilon certified: {df.epsilon_emp.max():.4f}")
    if df.epsilon_emp.max() < 0.01:
        log.info("  Canary audit also certified nothing. The null is not an")
        log.info("  artefact of using weak population attacks.")
    else:
        log.info("  Canary audit certifies a real bound at one training run , ")
        log.info("  cheaper than LiRA (64 models), which certified nothing.")

    # Comparison against the population attacks already measured
    ap_path = out / "attacks_summary.csv"
    if ap_path.exists():
        pop = pd.read_csv(ap_path)
        tg = pd.read_csv(out / "targets_summary.csv")
        train_j = tg.total_j_net.mean()
        log.info("\n  COST LADDER (energy per audit / energy per training run)")
        log.info(f"  {'audit':22} {'models':>7} {'kJ':>9} {'xtrain':>8} {'eps_emp':>9}")
        sh = pd.read_csv(out / "shadows_summary.csv")
        per_shadow = sh.total_j_net.mean()
        n_seeds = max(tg.seed.nunique(), 1)
        for atk, gg in pop.groupby("attack"):
            j = gg.total_j_net.mean() + gg.n_models.mean() * per_shadow / n_seeds
            log.info(f"  {atk:22} {gg.n_models.mean():>7.0f} {j/1000:>9.1f} "
                     f"{j/train_j:>8.2f} {gg.epsilon_emp.mean():>9.4f}")
        cj = df.total_j_net.mean()
        log.info(f"  {'canary_oneruntrain':22} {1:>7} {cj/1000:>9.1f} "
                 f"{cj/train_j:>8.2f} {df.epsilon_emp.mean():>9.4f}")

    prov = pe.measurement_provenance()
    prov.update(dict(canary_kind=args.canary_kind, n_canaries=args.n_canaries,
                     inclusion_p=INCLUSION_P, method="one-run canary audit",
                     alpha=0.05, delta=pu.DEFAULT_DELTA,
                     epochs=args.epochs, seeds=args.seeds,
                     batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE,
                     max_grad_norm=MAX_GRAD_NORM))
    (out / f"canary_provenance{tag_suffix}.json").write_text(json.dumps(prov, indent=2, default=str))
    log.info(f"\n  Wrote {out}/canary_summary{tag_suffix}.csv")


if __name__ == "__main__":
    main()

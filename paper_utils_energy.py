"""Energy instrumentation and audit metrics for "Liquidating the Fund".

Companion module to ``paper_utils_dpcnn.py``. Kept separate rather than merged so
the existing, already-validated training code is untouched.

Contents
--------
1. RAPL (CPU package) reader with wraparound handling
2. NVML (GPU) energy counter with power-polling fallback
3. EnergyMeter context manager
4. Idle baseline measurement and subtraction
5. Empirical epsilon (one-sided binomial bounds, privacy-region conversion)
6. Provenance recording

Design decisions:
- Measurement boundary: GPU + CPU, idle-subtracted (headline); GPU-only reported
  as a secondary column.
- RAPL access via passwordless sudoers rule (decision on file).
- The audit statistic is epsilon_emp; TPR@low-FPR and AUC are descriptive columns.
"""
from __future__ import annotations

import json
import logging
import platform
import subprocess
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.stats import beta

log = logging.getLogger("paper_utils_energy")

# RAPL package-0 is the CPU die. intel-rapl:1 on this machine is psys (platform
# domain), which is a superset of package-0, summing them double-counts, so
# scope is package-0 only.
RAPL_ROOT = Path("/sys/class/powercap")
RAPL_PACKAGES = ("intel-rapl:0",)

DEFAULT_POLL_INTERVAL_S = 1.0
DEFAULT_IDLE_DURATION_S = 300.0     # 5 min, open decision
DEFAULT_WARMUP_S = 30.0


def silence_opacus_hook_warnings() -> None:
    """Suppress the benign Opacus full-backward-hook UserWarning.

    Opacus registers full backward hooks on every module to capture per-sample
    gradients. The first layer's inputs do not require gradients, so the hook
    fires without one and PyTorch emits a warning per batch, thousands of lines
    over a sweep. It does not affect the privacy guarantee or the gradients
    Opacus clips.

    Deliberately narrow: matches only this message, so real Opacus warnings
    about privacy accounting or hook failures still surface.
    """
    import warnings
    warnings.filterwarnings(
        "ignore",
        message=r".*Full backward hook is firing when gradients are computed.*",
        category=UserWarning,
    )


# ===========================================================================
# 1. RAPL (CPU package energy)
# ===========================================================================

def _read_rapl_raw(package: str = "intel-rapl:0", use_sudo: bool = True) -> int:
    """Current RAPL energy counter in microjoules.

    Uses ``sudo -n`` so a missing sudoers rule fails immediately rather than
    blocking on a password prompt inside a notebook or a detached run.
    """
    path = RAPL_ROOT / package / "energy_uj"
    cmd = (["sudo", "-n", "cat", str(path)] if use_sudo else ["cat", str(path)])
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return int(out.stdout.strip())


def read_rapl_max_range(package: str = "intel-rapl:0", use_sudo: bool = False) -> int:
    """Counter wraparound point in microjoules. Read once, before measuring.

    Unlike energy_uj this file is world-readable, so sudo is off by default.
    """
    path = RAPL_ROOT / package / "max_energy_range_uj"
    try:
        return int(path.read_text().strip())
    except PermissionError:
        cmd = ["sudo", "-n", "cat", str(path)]
        out = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return int(out.stdout.strip())


def rapl_available(package: str = "intel-rapl:0", use_sudo: bool = True) -> bool:
    """True if this RAPL package can actually be read."""
    try:
        _read_rapl_raw(package, use_sudo)
        return True
    except Exception:
        return False


def list_rapl_packages(use_sudo: bool = True) -> list[str]:
    """Which RAPL packages are present and readable."""
    return [p for p in RAPL_PACKAGES if (RAPL_ROOT / p).exists()
            and rapl_available(p, use_sudo)]


class _RaplAccumulator:
    """Accumulates RAPL energy across counter wraparounds.

    The counter wraps roughly every 60s under load, so a naive end-minus-start
    delta is wrong for any run longer than that. Must be polled.
    """

    def __init__(self, package: str = "intel-rapl:0", use_sudo: bool = True):
        self.package = package
        self.use_sudo = use_sudo
        self.max_range = read_rapl_max_range(package, use_sudo)
        self.total_uj = 0
        self._prev = _read_rapl_raw(package, use_sudo)
        self.n_wraps = 0

    def poll(self) -> None:
        cur = _read_rapl_raw(self.package, self.use_sudo)
        delta = cur - self._prev
        if delta < 0:                      # counter wrapped since last poll
            delta += self.max_range
            self.n_wraps += 1
        self.total_uj += delta
        self._prev = cur

    @property
    def joules(self) -> float:
        return self.total_uj / 1e6


# ===========================================================================
# 2. NVML (GPU energy)
# ===========================================================================

def _nvml_handle(device_index: int = 0):
    import pynvml
    pynvml.nvmlInit()
    return pynvml, pynvml.nvmlDeviceGetHandleByIndex(device_index)


def gpu_energy_counter_available(device_index: int = 0) -> bool:
    """True if the GPU exposes the cumulative energy counter (Volta or newer)."""
    try:
        pynvml, h = _nvml_handle(device_index)
        pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
        return True
    except Exception:
        return False


def read_gpu_energy_mj(device_index: int = 0) -> int:
    """Cumulative GPU energy in millijoules since the driver was last loaded."""
    pynvml, h = _nvml_handle(device_index)
    return int(pynvml.nvmlDeviceGetTotalEnergyConsumption(h))


def read_gpu_power_w(device_index: int = 0) -> float:
    """Instantaneous GPU power draw in watts. Secondary trace only."""
    pynvml, h = _nvml_handle(device_index)
    return pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0


def read_gpu_temp_c(device_index: int = 0) -> float:
    """GPU temperature in Celsius. Recorded for thermal-stability provenance."""
    pynvml, h = _nvml_handle(device_index)
    return float(pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU))


# ===========================================================================
# 3. EnergyMeter
# ===========================================================================

@dataclass
class EnergyReading:
    """Result of one measured window."""
    elapsed_s: float
    gpu_joules: float
    cpu_joules: float
    gpu_method: str                  # "counter" or "power_integration"
    cpu_packages: list[str] = field(default_factory=list)
    n_samples: int = 0
    n_rapl_wraps: int = 0
    gpu_temp_start_c: float = float("nan")
    gpu_temp_end_c: float = float("nan")
    mean_gpu_power_w: float = float("nan")

    @property
    def total_joules(self) -> float:
        return self.gpu_joules + self.cpu_joules

    def net_of_idle(self, idle: "IdleBaseline") -> dict:
        """Energy attributable to the workload, idle power removed."""
        gpu_net = self.gpu_joules - idle.gpu_power_w * self.elapsed_s
        cpu_net = self.cpu_joules - idle.cpu_power_w * self.elapsed_s
        return {
            "gpu_joules_net": gpu_net,
            "cpu_joules_net": cpu_net,
            "total_joules_net": gpu_net + cpu_net,
            "gpu_joules_gross": self.gpu_joules,
            "cpu_joules_gross": self.cpu_joules,
            "total_joules_gross": self.total_joules,
        }


class EnergyMeter:
    """Measures CPU (RAPL) and GPU (NVML) energy over a code block.

    Fixes two bugs found in the smoke-test harness:
    - RAPL is polled and accumulated across counter wraparounds, not read
      start-to-end (which silently corrupts any run over ~60s).
    - GPU uses NVML's cumulative energy counter where available instead of
      approximating with mean power times elapsed time.

    Usage
    -----
    >>> with EnergyMeter() as em:
    ...     train(...)
    >>> em.reading.total_joules
    """

    def __init__(
        self,
        device_index: int = 0,
        packages: Optional[list[str]] = None,
        poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
        use_sudo: bool = True,
        keep_power_trace: bool = True,
    ):
        self.device_index = device_index
        self.poll_interval_s = poll_interval_s
        self.use_sudo = use_sudo
        self.keep_power_trace = keep_power_trace
        self.packages = packages if packages is not None else list_rapl_packages(use_sudo)
        self.power_trace: list[tuple[float, float]] = []
        self.reading: Optional[EnergyReading] = None
        self._stop = threading.Event()
        self._accumulators: list[_RaplAccumulator] = []

    def _poll_loop(self) -> None:
        while not self._stop.is_set():
            for acc in self._accumulators:
                try:
                    acc.poll()
                except Exception as e:            # keep the run alive; flag in log
                    log.warning(f"RAPL poll failed on {acc.package}: {e}")
            if self.keep_power_trace:
                try:
                    self.power_trace.append(
                        (time.time() - self._t0, read_gpu_power_w(self.device_index)))
                except Exception:
                    pass
            self._stop.wait(self.poll_interval_s)

    def __enter__(self) -> "EnergyMeter":
        self._accumulators = [_RaplAccumulator(p, self.use_sudo) for p in self.packages]

        self._gpu_counter = gpu_energy_counter_available(self.device_index)
        if self._gpu_counter:
            self._gpu_start_mj = read_gpu_energy_mj(self.device_index)
        else:
            log.warning("NVML energy counter unavailable; falling back to power integration")

        try:
            self._temp_start = read_gpu_temp_c(self.device_index)
        except Exception:
            self._temp_start = float("nan")

        self._t0 = time.time()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()
        elapsed = time.time() - self._t0

        for acc in self._accumulators:           # final poll closes the last interval
            try:
                acc.poll()
            except Exception:
                pass

        if self._gpu_counter:
            gpu_j = (read_gpu_energy_mj(self.device_index) - self._gpu_start_mj) / 1e3
            method = "counter"
        else:
            watts = [w for _, w in self.power_trace]
            gpu_j = (sum(watts) / len(watts)) * elapsed if watts else float("nan")
            method = "power_integration"

        try:
            temp_end = read_gpu_temp_c(self.device_index)
        except Exception:
            temp_end = float("nan")

        watts = [w for _, w in self.power_trace]
        self.reading = EnergyReading(
            elapsed_s=elapsed,
            gpu_joules=gpu_j,
            cpu_joules=sum(a.joules for a in self._accumulators),
            gpu_method=method,
            cpu_packages=list(self.packages),
            n_samples=len(self.power_trace),
            n_rapl_wraps=sum(a.n_wraps for a in self._accumulators),
            gpu_temp_start_c=self._temp_start,
            gpu_temp_end_c=temp_end,
            mean_gpu_power_w=(sum(watts) / len(watts)) if watts else float("nan"),
        )


# ===========================================================================
# 4. Idle baseline
# ===========================================================================

@dataclass
class IdleBaseline:
    """Idle power rates, measured not estimated. Subtracted as power x duration."""
    gpu_power_w: float
    cpu_power_w: float
    duration_s: float
    n_samples: int
    gpu_temp_c: float
    timestamp: str

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "IdleBaseline":
        return cls(**json.loads(Path(path).read_text()))


def measure_idle_baseline(
    duration_s: float = DEFAULT_IDLE_DURATION_S,
    device_index: int = 0,
    packages: Optional[list[str]] = None,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    use_sudo: bool = True,
) -> IdleBaseline:
    """Measure idle power under the same frozen conditions used for runs.

    Run this with no workload active, after the same warm-up the real runs get.
    """
    log.info(f"Measuring idle baseline for {duration_s:.0f}s, do not use the machine")
    with EnergyMeter(device_index, packages, poll_interval_s, use_sudo) as em:
        time.sleep(duration_s)
    r = em.reading
    idle = IdleBaseline(
        gpu_power_w=r.gpu_joules / r.elapsed_s,
        cpu_power_w=r.cpu_joules / r.elapsed_s,
        duration_s=r.elapsed_s,
        n_samples=r.n_samples,
        gpu_temp_c=r.gpu_temp_end_c,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    log.info(f"Idle: GPU {idle.gpu_power_w:.2f} W, CPU {idle.cpu_power_w:.2f} W")
    return idle


def warmup(seconds: float = DEFAULT_WARMUP_S, device_index: int = 0) -> float:
    """Spin the GPU briefly so measurements start from a stable temperature."""
    import torch
    if not torch.cuda.is_available():
        log.warning("No CUDA device; skipping warm-up")
        return float("nan")
    dev = torch.device(f"cuda:{device_index}")
    a = torch.randn(4096, 4096, device=dev)
    t0 = time.time()
    while time.time() - t0 < seconds:
        a = (a @ a).clamp(-1, 1)
    torch.cuda.synchronize()
    del a
    torch.cuda.empty_cache()
    temp = read_gpu_temp_c(device_index)
    log.info(f"Warm-up complete ({seconds:.0f}s), GPU at {temp:.0f} C")
    return temp


# ===========================================================================
# 5. Empirical epsilon
# ===========================================================================
#
# An MIA achieving (TPR, FPR) implies a lower bound on the true epsilon of the
# mechanism. Without confidence bounds this is not statistically valid: a single
# true positive with zero false positives would give epsilon = infinity. So TPR is
# lower-bounded and FPR upper-bounded by one-sided binomial intervals
# first. One-sided is used deliberately, the bound depends on one limit only, so
# one-sided intervals roughly double the effective significance.

def clopper_pearson_lower(k: int, n: int, alpha: float = 0.05) -> float:
    """One-sided lower confidence bound on a binomial rate."""
    if k <= 0:
        return 0.0
    return float(beta.ppf(alpha, k, n - k + 1))


def clopper_pearson_upper(k: int, n: int, alpha: float = 0.05) -> float:
    """One-sided upper confidence bound on a binomial rate."""
    if k >= n:
        return 1.0
    return float(beta.ppf(1 - alpha, k + 1, n - k))


def _eps_from_rates(tpr_lo: float, fpr_hi: float, delta: float,
                    eps_floor: float = 1e-12) -> float:
    """Privacy-region conversion, both directions, at one threshold.

    Both directions are legitimate: the second is the complementary test, and a
    tail asymmetry there is real signal even when bulk AUC is near chance. Only
    guards against actual division by zero.
    """
    tnr_lo = 1.0 - fpr_hi          # bounds flip for the complementary test
    fnr_hi = 1.0 - tpr_lo

    candidates = []
    if fpr_hi > eps_floor and (tpr_lo - delta) > 0:
        candidates.append(np.log((tpr_lo - delta) / fpr_hi))
    if fnr_hi > eps_floor and (tnr_lo - delta) > 0:
        candidates.append(np.log((tnr_lo - delta) / fnr_hi))

    return float(max(candidates)) if candidates else 0.0


@dataclass
class EpsilonEmpirical:
    """Empirical epsilon lower bound and the operating point that produced it."""
    epsilon_emp: float
    threshold: float
    tpr_raw: float
    fpr_raw: float
    tpr_lower: float
    fpr_upper: float
    n_members: int
    n_nonmembers: int
    delta: float
    alpha: float


def empirical_epsilon(
    scores: np.ndarray,
    is_member: np.ndarray,
    delta: float = 4.6e-5,
    alpha: float = 0.05,
    n_thresholds: int = 500,
    correct_multiplicity: bool = True,
) -> EpsilonEmpirical:
    """Empirical epsilon lower bound from MIA scores.

    Sweeps decision thresholds, applies one-sided binomial bounds at each,
    converts via the privacy region, and returns the maximum.

    Parameters
    ----------
    scores
        Attack scores; higher means "more likely a member".
    is_member
        Ground-truth membership, 1 = member.
    delta
        Must match the delta the mechanism was trained with.
    alpha
        One-sided confidence level (0.05 -> 95%) for the final bound.
    correct_multiplicity
        Bonferroni-correct alpha across the threshold sweep. Taking the max over
        many thresholds is itself a multiple-comparison problem: without this the
        reported bound holds at far below its nominal confidence. Leave on for
        anything reported; turn off only to compare against uncorrected
        literature values.

    Notes
    -----
    This is a population-level audit: trials are test records, not retrained
    models. Bounds are therefore looser than canary-based audits, which is the
    intended scope, the paper measures what a standard audit certifies.
    """
    scores = np.asarray(scores, dtype=float)
    is_member = np.asarray(is_member).astype(bool)

    members, nonmembers = scores[is_member], scores[~is_member]
    n_m, n_n = len(members), len(nonmembers)
    if n_m == 0 or n_n == 0:
        raise ValueError("Need both members and non-members to compute epsilon")

    lo, hi = float(np.min(scores)), float(np.max(scores))
    thresholds = np.linspace(lo, hi, n_thresholds)

    # Maximizing over many thresholds inflates the error rate; correct for it.
    # Two CP bounds are computed per threshold (tpr_lo and fpr_hi) and both
    # privacy-region directions are maximized over, so the union bound is over
    # n_thresholds * 2 events, not n_thresholds.
    alpha_eff = alpha / (n_thresholds * 2) if correct_multiplicity else alpha

    best = EpsilonEmpirical(0.0, float("nan"), float("nan"), float("nan"),
                            float("nan"), float("nan"), n_m, n_n, delta, alpha_eff)

    for t in thresholds:
        tp = int((members >= t).sum())
        fp = int((nonmembers >= t).sum())

        tpr_lo = clopper_pearson_lower(tp, n_m, alpha_eff)
        fpr_hi = clopper_pearson_upper(fp, n_n, alpha_eff)
        eps = _eps_from_rates(tpr_lo, fpr_hi, delta)

        if eps > best.epsilon_emp:
            best = EpsilonEmpirical(
                epsilon_emp=eps, threshold=float(t),
                tpr_raw=tp / n_m, fpr_raw=fp / n_n,
                tpr_lower=tpr_lo, fpr_upper=fpr_hi,
                n_members=n_m, n_nonmembers=n_n, delta=delta, alpha=alpha_eff)

    return best


def audit_tightness(epsilon_emp: float, epsilon_theoretical: Optional[float]) -> float:
    """Fraction of the nominal guarantee the audit actually certified.

    At low epsilon the ratio itself degenerates
    because epsilon_emp approaches zero, so report this alongside.
    """
    if epsilon_theoretical is None or epsilon_theoretical <= 0:
        return float("nan")
    return epsilon_emp / epsilon_theoretical


# ===========================================================================
# 6. Provenance
# ===========================================================================

def measurement_provenance(use_sudo: bool = True, device_index: int = 0) -> dict:
    """Environment facts that must accompany any reported energy number."""
    prov = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "hostname": platform.node(),
        "kernel": platform.release(),
        "python": platform.python_version(),
        "rapl_packages": list_rapl_packages(use_sudo),
        "rapl_access": "sudo" if use_sudo else "direct",
        "gpu_energy_counter": gpu_energy_counter_available(device_index),
    }

    for p in prov["rapl_packages"]:
        try:
            prov[f"rapl_max_range_uj_{p}"] = read_rapl_max_range(p, use_sudo)
        except Exception as e:
            prov[f"rapl_max_range_uj_{p}"] = f"unreadable: {e}"

    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        prov["gpu_name"] = pynvml.nvmlDeviceGetName(h)
        prov["nvml_driver"] = pynvml.nvmlSystemGetDriverVersion()
        if isinstance(prov["gpu_name"], bytes):
            prov["gpu_name"] = prov["gpu_name"].decode()
        if isinstance(prov["nvml_driver"], bytes):
            prov["nvml_driver"] = prov["nvml_driver"].decode()
    except Exception as e:
        prov["gpu_name"] = f"unreadable: {e}"

    try:
        import torch
        prov["torch"] = torch.__version__
        prov["cuda_capability"] = list(torch.cuda.get_device_capability())
    except Exception:
        pass

    return prov


def preflight(use_sudo: bool = True, device_index: int = 0) -> dict:
    """Run the pre-sweep checks and report what passed. Does not raise."""
    checks = {}

    pkgs = list_rapl_packages(use_sudo)
    checks["rapl_readable"] = (len(pkgs) > 0, f"packages: {pkgs or 'none'}")

    try:
        rng = read_rapl_max_range(pkgs[0], use_sudo) if pkgs else 0
        checks["rapl_max_range"] = (rng > 0, f"{rng} uJ")
    except Exception as e:
        checks["rapl_max_range"] = (False, str(e))

    have = gpu_energy_counter_available(device_index)
    checks["gpu_energy_counter"] = (have, "counter" if have else "falls back to power")

    try:
        checks["gpu_temp"] = (True, f"{read_gpu_temp_c(device_index):.0f} C")
    except Exception as e:
        checks["gpu_temp"] = (False, str(e))

    cache = Path("data/ptbxl_raw_100hz.npz")
    checks["ptbxl_cache"] = (cache.exists(), str(cache))

    for name, (ok, detail) in checks.items():
        log.info(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    return checks

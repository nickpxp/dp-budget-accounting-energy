"""preprocess_ptbxl.py

Build the raw-waveform cache the sweep reads.

Reads the PTB-XL download (ptbxl_database.csv plus the records100/ files),
extracts every 100 Hz 12-lead signal, derives the binary AF label, and writes
one compressed .npz with the arrays the training code expects.

    python3 preprocess_ptbxl.py --ptbxl data/ptbxl --out data/ptbxl_raw_100hz.npz

Output schema, data/ptbxl_raw_100hz.npz:
    X           (N, 1000, 12) float32   raw signals, leads I, II, III, aVR, aVL, aVF, V1-V6
    y           (N,)          int64     1 if AFIB or AFLT appears in scp_codes
    fold        (N,)          int64     strat_fold 1-10; 1-8 train, 9 val, 10 test
    patient_id  (N,)          int64
    ecg_id      (N,)          int64

Normalization is not applied here. The sweep fits per-lead statistics on its
own training records so nothing leaks across splits.
"""
from __future__ import annotations

import argparse
import ast
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm

log = logging.getLogger("preprocess_ptbxl")

# ---------------------------------------------------------------------------
# Module-level defaults
# ---------------------------------------------------------------------------
SAMPLING_RATE: int = 100
SIGNAL_LENGTH: int = 1000          # 10 s at 100 Hz
NUM_LEADS: int = 12
FILENAME_COL: str = "filename_lr"
LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]

AF_POSITIVE_CODES: frozenset = frozenset({"AFIB", "AFLT"})

EXPECTED_RECORD_COUNT: int = 21799
EXPECTED_AF_PREVALENCE: float = 0.0727
PREVALENCE_TOLERANCE: float = 0.005


# ---------------------------------------------------------------------------
# 1. Labels
# ---------------------------------------------------------------------------
def extract_af_label(scp_codes_str: Optional[str]) -> int:
    """Binary AF label from the scp_codes string.

    Rhythm statements carry likelihood 0 throughout PTB-XL, so the test is
    key presence, not a value threshold.
    """
    if not isinstance(scp_codes_str, str):
        return 0
    try:
        codes = ast.literal_eval(scp_codes_str)
    except (ValueError, SyntaxError):
        return 0
    if not isinstance(codes, dict):
        return 0
    return int(bool(AF_POSITIVE_CODES & set(codes.keys())))


def _label_self_test() -> None:
    cases = [
        ("{'AFIB': 0.0, 'NORM': 0.0}", 1),
        ("{'AFLT': 0.0}", 1),
        ("{'NORM': 0.0, 'SR': 0.0}", 0),
        ("{'SVARR': 0.0}", 0),
        ("{}", 0),
        (None, 0),
    ]
    for inp, expected in cases:
        assert extract_af_label(inp) == expected, f"label test failed on {inp!r}"


# ---------------------------------------------------------------------------
# 2. Lead-order check
# ---------------------------------------------------------------------------
def check_first_record(ptbxl_dir: Path, metadata: pd.DataFrame) -> None:
    """Read one record and confirm shape, rate and lead order before the bulk pass.

    A silent lead reorder would invalidate every downstream model, so this
    fails loudly. Casing of aVR/aVL/aVF differs between wfdb versions; order
    is what matters.
    """
    path = str(ptbxl_dir / metadata.iloc[0][FILENAME_COL])
    sig, meta = wfdb.rdsamp(path)
    assert sig.shape == (SIGNAL_LENGTH, NUM_LEADS), \
        f"unexpected shape {sig.shape}, expected ({SIGNAL_LENGTH}, {NUM_LEADS})"
    assert meta["fs"] == SAMPLING_RATE, \
        f"unexpected sampling rate {meta['fs']}, expected {SAMPLING_RATE}"
    got = [s.lower() for s in meta["sig_name"]]
    want = [s.lower() for s in LEAD_ORDER]
    assert got == want, f"lead order mismatch: {meta['sig_name']} != {LEAD_ORDER}"
    log.info(f"first record ok: {sig.shape} at {meta['fs']} Hz, leads {meta['sig_name']}")


# ---------------------------------------------------------------------------
# 3. Extraction
# ---------------------------------------------------------------------------
def extract_all_signals(
    ptbxl_dir: Path,
    metadata: pd.DataFrame,
    sample_limit: Optional[int] = None,
) -> dict:
    """Read every record's 12-lead waveform into one preallocated array.

    Args:
        ptbxl_dir: PTB-XL root, containing records100/.
        metadata: ptbxl_database.csv indexed by ecg_id.
        sample_limit: process only the first N records (smoke test).

    Returns:
        Dict with X, y, fold, patient_id, ecg_id and a failures list. A
        record that fails to read is left zero-filled and logged.
    """
    if sample_limit:
        metadata = metadata.head(sample_limit)
    n = len(metadata)

    X = np.zeros((n, SIGNAL_LENGTH, NUM_LEADS), dtype=np.float32)
    y = np.zeros(n, dtype=np.int64)
    fold = np.zeros(n, dtype=np.int64)
    patient_id = np.zeros(n, dtype=np.int64)
    ecg_id = np.zeros(n, dtype=np.int64)
    failures = []

    for i, (eid, row) in enumerate(tqdm(metadata.iterrows(), total=n, desc="extracting")):
        ecg_id[i] = eid
        patient_id[i] = int(row["patient_id"])
        fold[i] = int(row["strat_fold"])
        y[i] = extract_af_label(row["scp_codes"])
        try:
            sig, _ = wfdb.rdsamp(str(ptbxl_dir / row[FILENAME_COL]))
            if sig.shape != (SIGNAL_LENGTH, NUM_LEADS):
                failures.append(dict(ecg_id=eid, af=int(y[i]), reason=f"shape {sig.shape}"))
                continue
            if not np.isfinite(sig).all():
                # zero the bad samples but keep the record, and say so
                n_bad = int((~np.isfinite(sig)).sum())
                sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)
                failures.append(dict(ecg_id=eid, af=int(y[i]), reason=f"{n_bad} non-finite values zeroed"))
            X[i] = sig.astype(np.float32)
        except Exception as e:
            failures.append(dict(ecg_id=eid, af=int(y[i]), reason=f"{type(e).__name__}: {str(e)[:100]}"))

    return dict(X=X, y=y, fold=fold, patient_id=patient_id, ecg_id=ecg_id, failures=failures)


# ---------------------------------------------------------------------------
# 4. Checks
# ---------------------------------------------------------------------------
def check_cache(X: np.ndarray, y: np.ndarray, fold: np.ndarray, patient_id: np.ndarray) -> None:
    """Record count, AF prevalence, and patient disjointness across the folds."""
    assert len(X) == EXPECTED_RECORD_COUNT, \
        f"record count {len(X)} != expected {EXPECTED_RECORD_COUNT}"
    prev = float(y.mean())
    assert abs(prev - EXPECTED_AF_PREVALENCE) < PREVALENCE_TOLERANCE, \
        f"AF prevalence {prev:.4f} outside tolerance of {EXPECTED_AF_PREVALENCE:.4f}"

    train = set(patient_id[fold <= 8])
    val = set(patient_id[fold == 9])
    test = set(patient_id[fold == 10])
    assert not (train & test), f"patient leakage: train and test share {len(train & test)} patients"
    assert not (train & val), f"patient leakage: train and val share {len(train & val)} patients"

    log.info(f"records {len(X)}, AF prevalence {prev:.4f} ({int(y.sum())} AF)")
    log.info(f"patients: train {len(train)}, val {len(val)}, test {len(test)}, disjoint")
    for name, m in [("train", fold <= 8), ("val", fold == 9), ("test", fold == 10)]:
        log.info(f"  {name:5s} {int(m.sum()):5d} records, AF prevalence {y[m].mean():.4f}")


def report_failures(failures: list, y: np.ndarray, path: Path) -> None:
    """Write the failure log and check whether failures lean toward AF records."""
    if not failures:
        log.info("no extraction failures")
        return
    df = pd.DataFrame(failures)
    df.to_csv(path, index=False)
    n_af, n_non = int((df.af == 1).sum()), int((df.af == 0).sum())
    log.info(f"wrote {len(df)} failures to {path}")
    log.info(f"  AF fail rate {n_af}/{int(y.sum())}, non-AF fail rate {n_non}/{int(len(y) - y.sum())}")
    for reason, count in df.reason.value_counts().head(10).items():
        log.info(f"  {count:4d}  {reason}")


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ptbxl", default="data/ptbxl", help="PTB-XL root directory")
    ap.add_argument("--out", default="data/ptbxl_raw_100hz.npz")
    ap.add_argument("--failures", default="data/ptbxl_extraction_failures.csv")
    ap.add_argument("--smoke", type=int, default=50, help="records in the smoke test; 0 to skip")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    _label_self_test()

    ptbxl_dir = Path(args.ptbxl)
    meta_path = ptbxl_dir / "ptbxl_database.csv"
    if not meta_path.exists():
        meta_path = ptbxl_dir / "ptbxl.csv"
    metadata = pd.read_csv(meta_path, index_col="ecg_id")
    log.info(f"loaded metadata: {len(metadata)} records")

    check_first_record(ptbxl_dir, metadata)

    if args.smoke:
        smoke = extract_all_signals(ptbxl_dir, metadata, sample_limit=args.smoke)
        assert smoke["X"].shape == (args.smoke, SIGNAL_LENGTH, NUM_LEADS)
        log.info(f"smoke test: {args.smoke} records, {len(smoke['failures'])} failures")

    r = extract_all_signals(ptbxl_dir, metadata)
    X, y, fold, patient_id, ecg_id = r["X"], r["y"], r["fold"], r["patient_id"], r["ecg_id"]
    log.info(f"X {X.shape} {X.dtype}, {X.nbytes / 1e9:.2f} GB in memory")

    check_cache(X, y, fold, patient_id)
    report_failures(r["failures"], y, Path(args.failures))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=X, y=y, fold=fold, patient_id=patient_id, ecg_id=ecg_id)
    log.info(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")

    # round trip
    z = np.load(out)
    assert np.array_equal(z["y"], y) and np.allclose(z["X"], X)
    log.info(f"reload ok: keys {', '.join(z.files)}")


if __name__ == "__main__":
    main()

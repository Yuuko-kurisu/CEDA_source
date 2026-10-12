"""Pure historical helper excerpt; no experiment code imported."""
from typing import Any, Sequence
import numpy as np

def cluster_mean_ci(rows: Sequence[dict[str, Any]], metric: str, n_boot: int = 2000, seed: int = 17) -> tuple[float, float, float, int]:
    if not rows:
        return 0.0, 0.0, 0.0, 0
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(str(row["case_id"]), []).append(float(row.get(metric, 0.0)))
    clusters = sorted(grouped)
    values = np.asarray([np.mean(grouped[c]) for c in clusters], dtype=float)
    mean = float(np.mean(values)) if values.size else 0.0
    if values.size <= 1:
        return mean, mean, mean, int(values.size)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    for idx in range(n_boot):
        sample = rng.integers(0, values.size, size=values.size)
        boots[idx] = float(np.mean(values[sample]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return mean, float(lo), float(hi), int(values.size)


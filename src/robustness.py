"""Reproducible trial seeds, disturbances and condition summaries for Track A."""
from dataclasses import dataclass
import numpy as np

@dataclass
class Disturbance:
    """Optional reproducible disturbances for an evaluation trial."""
    seed: int = 0
    encoder_bias_rad_per_sample: float = 0.0
    lidar_bias_m: float = 0.0
    camera_dropout_probability: float = 0.0
    lidar_dropout_probability: float = 0.0
    wheel_scale_left: float = 1.0
    wheel_scale_right: float = 1.0
    start_x_offset_m: float = 0.0
    start_y_offset_m: float = 0.0
    start_heading_offset_rad: float = 0.0

    def validate(self):
        for p in (self.camera_dropout_probability, self.lidar_dropout_probability):
            if not 0.0 <= p <= 1.0:
                raise ValueError("Dropout probabilities must be in [0, 1]")
        if self.wheel_scale_left <= 0 or self.wheel_scale_right <= 0:
            raise ValueError("Wheel scale factors must be positive")


def make_trial_seeds(n_trials=20, base_seed=41014):
    """Fixed seed list so results are reproducible and comparable."""
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    return [int(base_seed + i) for i in range(n_trials)]


def compare_conditions(condition_results):
    """Return a simple table-like list for baseline vs disturbed condition results.

    condition_results: dict[str, list[dict]] where each trial dict contains
    the evaluation metrics returned by sensing pose_error_metrics(), plus
    optional 'success' and 'collisions' fields.
    """
    rows = []
    for condition, trials in condition_results.items():
        if not trials:
            continue
        row = {"condition": condition, "n_trials": len(trials)}
        if all("success" in t for t in trials):
            row["success_rate"] = float(np.mean([bool(t["success"]) for t in trials]))
        for metric in ("position_rmse_m", "position_mean_m", "collisions",
                       "marker_visible_fraction"):
            values = np.asarray([t[metric] for t in trials
                                 if metric in t and np.isfinite(t[metric])], dtype=float)
            if values.size:
                row[metric + "_mean"] = float(values.mean())
                row[metric + "_std"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
        rows.append(row)
    return rows

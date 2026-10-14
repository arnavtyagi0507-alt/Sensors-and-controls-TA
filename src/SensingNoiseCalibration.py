"""Sensing, calibration, noise handling and frame utilities for 41014 Track A.

This module is deliberately independent of MuJoCo globals so it can be imported
by the team's simulation. Ground truth must NOT be passed to the controller
through this module; ground truth is for evaluation/logging only.

Coordinate conventions used here:
  world/map: x-y plane, heading theta measured counter-clockwise from +x
  base:      +x forward, +y left
  camera:    OpenCV convention (+x right, +y down, +z forward)

IMPORTANT: verify the actual MuJoCo camera extrinsics and the ArUco PnP
convention in your team's model before using camera_to_base(). The supplied
demo's camera-frame comments may not match OpenCV's standard convention.
"""


from dataclasses import dataclass
from typing import Optional, Sequence
import numpy as np


def wrap_angle(angle: float) -> float:
    """Wrap angle(s) to [-pi, pi)."""
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


@dataclass
class NoiseConfig:
    """1-sigma sensor noise values; units are radians, metres and pixels."""
    encoder_sigma_rad: float = 0.002
    lidar_sigma_m: float = 0.01
    camera_range_sigma_m: float = 0.015
    camera_bearing_sigma_rad: float = np.deg2rad(1.0)
    camera_pixel_sigma_px: float = 0.5
    encoder_bias_rad_per_sample: float = 0.0
    lidar_bias_m: float = 0.0
    camera_dropout_probability: float = 0.0
    lidar_dropout_probability: float = 0.0
    outlier_probability: float = 0.0
    outlier_scale: float = 5.0

    def validate(self):
        for name in (
            "encoder_sigma_rad", "lidar_sigma_m",
            "camera_range_sigma_m", "camera_bearing_sigma_rad",
            "camera_pixel_sigma_px", "camera_dropout_probability",
            "lidar_dropout_probability", "outlier_probability",
        ):
            value = float(getattr(self, name))
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        for name in (
            "camera_dropout_probability", "lidar_dropout_probability",
            "outlier_probability",
        ):
            if getattr(self, name) > 1:
                raise ValueError(f"{name} must be in [0, 1]")


class SensorProcessor:
    """Apply explicit sensor noise/disturbances and reject invalid measurements.

    The caller should either apply noise here OR in the original sensor reader,
    not both. To avoid double-counting noise, remove the original rng.normal(...)
    calls in read_encoders()/read_lidar() if using this class for noise.
    """

    def __init__(self, config: Optional[NoiseConfig] = None, seed=None):
        self.config = config or NoiseConfig()
        self.config.validate()
        self.rng = np.random.default_rng(seed)

    def encoder_angle(self, raw_angle: float, add_noise=True) -> float:
        """Noisy wheel angle in radians. Use consistently for both wheels."""
        value = float(raw_angle)
        if add_noise:
            value += self.config.encoder_bias_rad_per_sample
            value += self.rng.normal(0.0, self.config.encoder_sigma_rad)
        return value

    def lidar_scan(self, raw_ranges: Sequence[float], add_noise=True) -> np.ndarray:
        """Return a cleaned LiDAR scan; non-hits remain -1, invalid values become -1."""
        ranges = np.asarray(raw_ranges, dtype=float).copy()
        ranges[~np.isfinite(ranges)] = -1.0
        ranges[ranges <= 0.0] = -1.0
        valid = ranges > 0.0
        if add_noise and valid.any():
            ranges[valid] += self.config.lidar_bias_m
            ranges[valid] += self.rng.normal(
                0.0, self.config.lidar_sigma_m, int(valid.sum())
            )
            if self.config.outlier_probability > 0:
                outliers = self.rng.random(ranges.shape) < self.config.outlier_probability
                outliers &= valid
                ranges[outliers] += self.rng.normal(
                    0.0,
                    self.config.outlier_scale * self.config.lidar_sigma_m,
                    int(outliers.sum()),
                )
        # Reject impossible/negative readings after perturbation.
        ranges[ranges <= 0.0] = -1.0
        return ranges

    def camera_measurement(self, range_m: float, bearing_rad: float):
        """Validate and perturb a (range, bearing) camera measurement.

        Returns None for invalid measurements or simulated marker dropout.
        """
        r, b = float(range_m), float(bearing_rad)
        if not (np.isfinite(r) and np.isfinite(b) and r > 0.0):
            return None
        if self.rng.random() < self.config.camera_dropout_probability:
            return None
        r += self.rng.normal(0.0, self.config.camera_range_sigma_m)
        b += self.rng.normal(0.0, self.config.camera_bearing_sigma_rad)
        if self.config.outlier_probability and self.rng.random() < self.config.outlier_probability:
            r += self.rng.normal(0.0, self.config.outlier_scale * self.config.camera_range_sigma_m)
            b += self.rng.normal(0.0, self.config.outlier_scale * self.config.camera_bearing_sigma_rad)
        if r <= 0 or not np.isfinite(r) or not np.isfinite(b):
            return None
        return float(r), float(wrap_angle(b))

    def maybe_drop_lidar(self, scan: Sequence[float]) -> np.ndarray:
        """Simulate a complete LiDAR packet dropout by marking all beams as no-hit."""
        result = np.asarray(scan, dtype=float).copy()
        if self.rng.random() < self.config.lidar_dropout_probability:
            result[:] = -1.0
        return result


def camera_intrinsics_from_fov(width_px: int, height_px: int, fov_y_deg: float):
    """Pinhole camera matrix K and zero-distortion starting estimate.

    Use a calibration target and cv2.calibrateCamera for real camera calibration.
    In simulation this provides nominal intrinsics from the MuJoCo camera FOV.
    """
    if width_px <= 0 or height_px <= 0 or not 0 < fov_y_deg < 180:
        raise ValueError("Invalid image dimensions or vertical field of view")
    fy = (height_px / 2.0) / np.tan(np.deg2rad(fov_y_deg) / 2.0)
    # Square pixels are assumed; principal point is the image centre.
    fx = fy
    cx, cy = width_px / 2.0, height_px / 2.0
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    distortion = np.zeros(5, dtype=float)
    return K, distortion


def estimate_noise_sigma(residuals: Sequence[float], remove_bias=True) -> dict:
    """Estimate bias, sample standard deviation and RMSE from calibration residuals.

    Residuals should be measured minus reference/calibration values. Use repeated
    stationary readings or known reference distances; do not invent noise values.
    """
    x = np.asarray(residuals, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        raise ValueError("At least two finite calibration residuals are required")
    bias = float(np.mean(x))
    corrected = x - bias if remove_bias else x
    return {
        "n": int(x.size),
        "bias": bias,
        "sigma": float(np.std(corrected, ddof=1)),
        "rmse": float(np.sqrt(np.mean(x ** 2))),
    }


def calibrate_camera_range_bearing(range_samples, bearing_samples,
                                   reference_range, reference_bearing):
    """Estimate camera range/bearing bias and sigma from repeated known-pose trials."""
    ranges = np.asarray(range_samples, dtype=float)
    bearings = np.asarray(bearing_samples, dtype=float)
    if ranges.shape != bearings.shape:
        raise ValueError("Range and bearing samples must have the same shape")
    range_residual = ranges - float(reference_range)
    bearing_residual = wrap_angle(bearings - float(reference_bearing))
    return {
        "range": estimate_noise_sigma(range_residual),
        "bearing": estimate_noise_sigma(bearing_residual),
    }


def lidar_scan_to_points(scan, angles_rad, max_range=3.5):
    """Convert a planar scan to base-frame (x-forward, y-left) points.

    The angle array must match the actual beam directions in the robot frame.
    Invalid/no-hit ranges are omitted.
    """
    r = np.asarray(scan, dtype=float)
    a = np.asarray(angles_rad, dtype=float)
    if r.shape != a.shape:
        raise ValueError("scan and angles_rad must have the same shape")
    valid = np.isfinite(r) & (r > 0.0) & (r <= max_range)
    if not np.any(valid):
        return np.empty((0, 2), dtype=float)
    return np.column_stack((r[valid] * np.cos(a[valid]),
                            r[valid] * np.sin(a[valid])))


def base_to_world(point_base, pose_world_base):
    """Transform one or more base-frame xy points into the world/map frame."""
    points = np.asarray(point_base, dtype=float)
    pose = np.asarray(pose_world_base, dtype=float)
    if pose.shape != (3,) or points.shape[-1] != 2:
        raise ValueError("Expected pose [x,y,theta] and points ending in dimension 2")
    c, s = np.cos(pose[2]), np.sin(pose[2])
    rotation = np.array([[c, -s], [s, c]])
    return points @ rotation.T + pose[:2]


def world_to_base(point_world, pose_world_base):
    """Transform one or more world/map-frame xy points into the robot base frame."""
    points = np.asarray(point_world, dtype=float)
    pose = np.asarray(pose_world_base, dtype=float)
    if pose.shape != (3,) or points.shape[-1] != 2:
        raise ValueError("Expected pose [x,y,theta] and points ending in dimension 2")
    c, s = np.cos(pose[2]), np.sin(pose[2])
    rotation = np.array([[c, s], [-s, c]])
    return (points - pose[:2]) @ rotation.T


def camera_to_base(point_camera_xyz, camera_origin_base_xy, camera_yaw_rad=0.0):
    """Convert a camera-frame 3D point to base-frame xy using a planar extrinsic.

    This helper assumes camera point coordinates are expressed as:
      camera +z = forward, +x = right (OpenCV convention).
    It maps camera forward depth to base +x and camera-right to base -y,
    then applies the camera yaw and planar origin offset. Verify axes and yaw
    against the actual MuJoCo camera before relying on this conversion.
    """
    p = np.asarray(point_camera_xyz, dtype=float)
    origin = np.asarray(camera_origin_base_xy, dtype=float)
    if p.shape != (3,) or origin.shape != (2,):
        raise ValueError("Expected xyz camera point and xy camera origin")
    point_base_camera = np.array([p[2], -p[0]], dtype=float)
    c, s = np.cos(camera_yaw_rad), np.sin(camera_yaw_rad)
    rot = np.array([[c, -s], [s, c]])
    return rot @ point_base_camera + origin


def pose_error_metrics(estimated_poses, true_poses):
    """Evaluation-only metrics; never use these values as controller inputs."""
    est = np.asarray(estimated_poses, dtype=float)
    truth = np.asarray(true_poses, dtype=float)
    if est.shape != truth.shape or est.ndim != 2 or est.shape[1] != 3:
        raise ValueError("Expected matching Nx3 arrays [x, y, theta]")
    valid = np.all(np.isfinite(est), axis=1) & np.all(np.isfinite(truth), axis=1)
    if not valid.any():
        raise ValueError("No finite pose pairs available")
    delta = est[valid] - truth[valid]
    position_error = np.linalg.norm(delta[:, :2], axis=1)
    heading_error = np.abs(wrap_angle(delta[:, 2]))
    return {
        "valid_samples": int(valid.sum()),
        "position_rmse_m": float(np.sqrt(np.mean(position_error ** 2))),
        "position_mean_m": float(np.mean(position_error)),
        "position_p95_m": float(np.percentile(position_error, 95)),
        "heading_rmse_rad": float(np.sqrt(np.mean(heading_error ** 2))),
        "heading_mean_rad": float(np.mean(heading_error)),
    }


def summarize_trials(trial_results):
    """Aggregate per-trial dictionaries into success rate and metric summaries.

    Each item may contain 'success', 'position_rmse_m', 'position_mean_m',
    'collisions', and/or 'marker_visible_fraction'. Missing metrics are skipped.
    """
    trials = list(trial_results)
    if not trials:
        raise ValueError("At least one trial result is required")
    out = {"n_trials": len(trials)}
    if all("success" in t for t in trials):
        out["success_rate"] = float(np.mean([bool(t["success"]) for t in trials]))
    for key in ("position_rmse_m", "position_mean_m", "collisions",
                "marker_visible_fraction"):
        vals = np.asarray([t[key] for t in trials if key in t and np.isfinite(t[key])],
                          dtype=float)
        if vals.size:
            out[key + "_mean"] = float(vals.mean())
            out[key + "_std"] = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
            out[key + "_min"] = float(vals.min())
            out[key + "_max"] = float(vals.max())
    return out

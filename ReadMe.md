Tasks:
Track A, mobile robot: visual waypoint navigation

Christian: Sensing - calibration, noise handling and frame management, Integration and robustness - performance under disturbance and noise

Nivedha : modelling and estimation

# 41014 Track A — sensing and robustness starter modules

## Files
- `src/SensingNoiseCalibration.py`: sensor-noise model, nominal camera intrinsics,
  calibration residual statistics, sensor validation, 2D frame transforms and
  evaluation metrics.
- `src/robustness.py`: reproducible trial seeds, disturbance configuration and
  condition summaries.

## Important integration notes
1. These modules are utilities, not a complete end-to-end navigation pipeline.
   Connect them to the group's existing MuJoCo simulation and estimator/controller.
2. The uploaded demo already adds encoder and LiDAR noise in its reader functions.
   If using `SensorProcessor` to add noise, remove those existing noise additions
   or set `add_noise=False`; do not add the same noise twice.
3. `camera_intrinsics_from_fov()` supplies nominal pinhole intrinsics from the
   MuJoCo FOV, not a measured physical-camera calibration.
4. Confirm the camera optical axes and camera-to-base transform in the actual
   MuJoCo model. OpenCV PnP convention is x-right, y-down, z-forward; do not
   assume a variable named `z_c` means forward unless the frame has been verified.
5. Keep ground-truth pose out of the control and estimation update. Use it only
   for post-run evaluation, plots and reported metrics.
6. Report measured/estimated noise values honestly. Do not present the default
   values as empirically calibrated unless you have collected calibration data.

## Quick import check
From the repository root:
```python
from src.SensingNoiseCalibration import (
    NoiseConfig, SensorProcessor, camera_intrinsics_from_fov,
    pose_error_metrics, lidar_scan_to_points, base_to_world, world_to_base
)

processor = SensorProcessor(NoiseConfig(), seed=7)
K, distortion = camera_intrinsics_from_fov(640, 480, 60.0)
print(K)
```

## Dependencies
NumPy is required. The full MuJoCo demo additionally uses `mujoco`, `opencv-contrib-python`
and `matplotlib` as applicable to the existing team code.


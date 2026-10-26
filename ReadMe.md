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

## Integrated simulation
The Track A demo now routes encoder, LiDAR and ArUco range/bearing measurements
through `SensorProcessor`. The simulated noise is applied once in the readers;
the EKF and controller consume those processed measurements. LiDAR no-returns
use `-1` internally, and the controller stops if the complete scan is dropped.
The controller and EKF use the estimate only; true pose is used for evaluation.
`main.py` selects the noise configuration and trial disturbances, then calls the
simulation and aggregates repeated trials with the helpers in `src/robustness.py`.

Run one interactive trial from the repository root:
```bash
python main.py --seed 41014
```

Run repeatable headless trials comparing baseline and a documented synthetic
disturbance condition:
```bash
python main.py --headless --robustness --trials 10 --seed 41014
```
Each condition uses the same trial seeds. The summary reports docking success
rate, pose-error metrics, collision steps and marker visibility. The disturbed
case combines encoder bias, LiDAR bias, camera/LiDAR dropout, wheel-scale error
and a start-pose offset; these are stress-test settings, not measured hardware
calibration values. The current trial batch uses one fixed obstacle layout.

## Calibration and frame notes
1. `camera_intrinsics_from_fov()` supplies nominal pinhole intrinsics from the
   MuJoCo FOV, not a measured physical-camera calibration.
2. In the included MuJoCo XML, the camera origin is 0.07 m forward of the robot
   centre. Its `xyaxes` make camera-right point along robot -y and camera-up
   point along +z; the MuJoCo view direction is robot +x. This agrees with the
   PnP interpretation used by the demo: camera `t_z` is forward, `t_x` is right,
   and `atan2(-t_x, t_z)` is positive to the robot's left. Re-check this mapping
   if the camera pose or model changes.
3. The live plot shows the true pose, EKF estimate and a 95% position covariance
   ellipse. Report measured/estimated noise values honestly; do not present the default
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
Install the runtime dependencies from the repository root with
`python -m pip install -r requirements.txt`. The demo uses NumPy, MuJoCo,
OpenCV-contrib, SciPy and Matplotlib.

# 41014 Integrated Project — Track A: Visual Waypoint Navigation

**Project group:** 18

**Platform:** TurtleBot3-style differential-drive robot in MuJoCo

**Task:** Navigate around obstacles and visually dock at an ArUco-marked station.

**Video Link:** 

## Team and contributions

- **Arnav Tyagi:** Platform modelling and estimation.
- **Nivedha Suresh Balan:** Controller design and implementation.
- **Christian Puga:** Sensing, calibration, noise handling, frame management,
  integration and robustness.

## Project overview

The robot operates in a simulated room containing walls, randomly generated
obstacles and an ArUco marker at the docking station. The simulation includes
wheel encoders, a 72-beam 2D LiDAR and a forward-facing camera.

The intended pipeline is:

1. Read encoder, LiDAR and camera measurements.
2. Estimate the robot pose from motion and exteroceptive measurements.
3. Use the estimate and sensor measurements to navigate and dock.
4. Compare the estimate with simulator ground truth for evaluation only.

## Platform model

The planar robot state is
\[
x = [p_x,\ p_y,\ \theta]^T,
\]
where \(p_x,p_y\) are world-frame position and \(\theta\) is heading. For wheel
radius \(r\), wheel separation \(b\), and left/right wheel rates
\(\dot\phi_L,\dot\phi_R\), the differential-drive model is
\[
v = \frac{r}{2}(\dot\phi_R+\dot\phi_L), \qquad
\omega = \frac{r}{b}(\dot\phi_R-\dot\phi_L),
\]
\[
\dot p_x=v\cos\theta,\qquad
\dot p_y=v\sin\theta,\qquad
\dot\theta=\omega.
\]

The runnable demo uses nominal \(r=0.033\) m, \(b=0.160\) m, and a 0.05 s
control period. These are simulation parameters, not claims of hardware
calibration.

`trackA.py` contains separate local state-space, controllability and
observability analysis code. Near straight motion at speed \(v_0\), its
linearised unicycle model uses state \([p_x,p_y,\theta]\), with
\[
A=\begin{bmatrix}0&0&0\\0&0&v_0\\0&0&0\end{bmatrix},\qquad
B=\begin{bmatrix}1&0\\0&0\\0&1\end{bmatrix}.
\]
The file also contains a reduced lateral/heading tracking model. These analysis
sections are separate prototypes; confirm their outputs and assumptions against
the final integrated code before presenting them as end-to-end results.

## Sensors, calibration and frames

### Sensor configuration in the standalone demo

- **Wheel encoders:** synthetic Gaussian noise with standard deviation
  0.002 rad per reading.
- **LiDAR:** 72 beams around the robot; synthetic Gaussian noise with standard
  deviation 0.01 m on valid returns. A value of `-1` means no return.
- **Camera:** detects ArUco marker corners and uses PnP with the known marker
  size to estimate relative range and bearing.
- **Camera intrinsics:** calculated from the simulated 640 × 480 image and 60°
  vertical field of view. This is a nominal simulation model, not a physical
  camera calibration.

### Coordinate-frame convention

```text
World/map: x-y plane; heading theta is counter-clockwise from world +x
Robot base: +x forward, +y left
OpenCV camera: +x right, +y down, +z forward
Camera bearing: positive means the marker is left of the robot
```

The PnP result is converted to range using camera \(x,z\), with bearing
\(\operatorname{atan2}(-x_c,z_c)\). Re-check this sign and the camera extrinsics
if the camera mount or MuJoCo model changes.

`src/SensingNoiseCalibration.py` provides reusable noise configuration,
measurement validation, nominal intrinsics, calibration-residual statistics,
frame-transform helpers and evaluation metrics. Its default parameters are
assumptions until estimated from calibration data. **This processor is not
currently connected to the standalone demo's sensor readers.**

## Estimation and control in the runnable demo

The current `trackA_demo.py` updates the pose using wheel-encoder odometry
(dead reckoning). It does **not** maintain an EKF covariance or fuse camera
range/bearing into the pose estimate. The marker measurement is instead used
directly by the docking controller.

The controller has two modes:

- **Navigate:** choose a LiDAR-clear direction near the waypoint direction,
  reducing speed when obstacles are close.
- **Dock:** use camera range to set forward speed and camera bearing to steer.
  If the marker is not visible, turn toward its mapped direction; stop forward
  motion when an obstacle is very close ahead.

`trackA.py` includes separate EKF/LQR and modelling prototypes, but they are not
integrated into the standalone demo loop. The project brief requires a
Kalman filter/EKF or observer and control on the estimated state. Do not claim
this requirement is demonstrated by the standalone demo until an integrated
version is implemented and tested.

Ground truth is used in the demo for post-run pose-error plots and collision
logging. It is not passed into the controller.

## Integration and robustness

`src/robustness.py` supplies reproducible trial seeds, a disturbance settings
dataclass and condition-summary helpers. These helpers need to be called by the
simulation's actual trial loop for their settings to affect a run. At present,
they are **not wired into** the standalone demo. The current `main.py` expects
a different simulation API and should not be used as the entry point for this
demo.

A single successful run is a smoke test, not a repeated-trial robustness study.
For a robustness claim, run multiple baseline and disturbed trials with the
same seed list, and report success rate, pose error, collisions, and the exact
disturbances used.

## Example result from one run

One observed run of the standalone demo reported:

- Docked at 30.2 s.
- Final position error: 0.034 m.
- Mean position error: 0.020 m.
- Marker visible in 63% of control steps.
- Zero control steps reported as touching a wall or obstacle.

These are results from **one trial only**. They do not establish repeated-trial
robustness or performance under injected disturbance.

## Install and run

From the repository root, create and activate a virtual environment and install
the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run the standalone demo:

```bash
# macOS: use MuJoCo's Python launcher for the interactive viewer
mjpython trackA_demo.py --seed 41014

# Other platforms
python trackA_demo.py --seed 41014
```

The demo saves its result plot as `trackA_result.png`. To run without the live
viewer:

```bash
mjpython trackA_demo.py --headless --seed 41014
```

**Clean-clone check still required:** `trackA_demo.py` must be committed to this
repository, then the install and run commands above should be tested from a
fresh clone. `python main.py` is not currently a supported run command.

## Demo video

**Five-minute project demonstration:** _Add the public video link here._

The video should show successful task completion, at least one varied or
disturbed trial, and the estimate with its uncertainty beside ground truth.
The current standalone demo does not display EKF uncertainty, so update the
video and this description when the integrated estimator is ready.

## Repository

[Project GitHub repository](https://github.com/arnavtyagi0507-alt/Sensors-and-controls-TA)

## Known limitations and next integration work

> **Implementation status:** The standalone `trackA_demo.py` has been run
> successfully and demonstrates navigation and docking. It currently estimates
> pose by wheel odometry; it does not fuse camera measurements with an EKF.
> `main.py`, `trackA.py`, and the sensing/robustness helpers are not yet a single
> integrated, clean-start pipeline. See [Known limitations](#known-limitations)
> before treating this repository as the final assessed submission.


1. Connect the sensing/noise processor to the demo's sensor readers without
   applying noise twice.
2. Integrate the EKF with encoder motion and camera range/bearing measurements;
   ensure the controller uses the estimated state.
3. Connect the robustness disturbances and repeated-trial summaries to the
   actual simulation loop.
4. Reconcile `main.py` with the actual simulation API, then test from a clean
   clone.
5. Add the coordinate-frame diagram, verified model-analysis results,
   repeated-trial quantitative results, failure analysis and public video link.

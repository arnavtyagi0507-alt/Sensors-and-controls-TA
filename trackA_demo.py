import time
 
import numpy as np
import cv2
import mujoco
import mujoco.viewer
import scipy.linalg
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from src.SensingNoiseCalibration import (
    NoiseConfig,
    SensorProcessor,
    camera_intrinsics_from_fov,
    pose_error_metrics,
)
from src.robustness import Disturbance
 
# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
WHEEL_R, WHEEL_SEP = 0.033, 0.160          # TurtleBot3 Burger wheel radius / separation (m)
MARKER_XY = np.array([2.44, 0.0])          # ArUco marker position on the wall (m)
MARKER_SIZE = 0.20                         # marker side (m)
DOCK_DIST = 0.40                           # stop this far from the marker (m)
PRE_DOCK = np.array([1.54, 0.0])           # drive here first, then dock along y = 0
START_POSE = np.array([-1.5, -0.5, np.radians(30)])
T = 0.05                                   # control / EKF sample time (s)
N_LIDAR = 72                               # LiDAR beams (one every 5 deg)
CAM_W, CAM_H, CAM_FOVY = 640, 480, 60.0
 
SCENE_SEED = 41014
rng = np.random.default_rng(SCENE_SEED)
 
# ---------------------------------------------------------------------------
# 1. MODEL: seeded random world + robot (start pose is perturbed per trial)
# ---------------------------------------------------------------------------
def random_obstacles(n=4):
    start = START_POSE[:2]
    obs = [start + rng.uniform(0.35, 0.65) * (PRE_DOCK - start)]     # one always in the way
    while len(obs) < n:
        p = rng.uniform(-2.0, 1.6, 2)
        if (np.linalg.norm(p - start) > 0.7 and np.linalg.norm(p - PRE_DOCK) > 0.7
                and all(np.linalg.norm(p - o) > 0.7 for o in obs)):
            obs.append(p)
    return obs
 
 
OBSTACLES = random_obstacles()
TRUE_START = START_POSE + [*rng.uniform(-0.15, 0.15, 2), np.radians(rng.uniform(-10, 10))]
 
 
def marker_png():
    tag = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), 0, 400)
    img = cv2.flip(cv2.copyMakeBorder(tag, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=255), 1)
    return cv2.imencode(".png", img)[1].tobytes()
 
 
lidar_sites = "".join(f'<site name="l{k}" pos="0 0 .09" zaxis="{np.cos(2*np.pi*k/N_LIDAR):.4f} '
                      f'{np.sin(2*np.pi*k/N_LIDAR):.4f} 0"/>' for k in range(N_LIDAR))
lidar_sensors = "".join(f'<rangefinder name="r{k}" site="l{k}" cutoff="3.5"/>' for k in range(N_LIDAR))
obstacle_geoms = "".join(f'<geom name="obstacle{k}" type="cylinder" pos="{x:.2f} {y:.2f} .2" '
                         f'size=".18 .2" rgba=".8 .3 .3 1"/>'
                         for k, (x, y) in enumerate(OBSTACLES))
wheel = lambda s, y: (f'<body name="w{s}" pos="0 {y} 0"><joint name="w{s}" type="hinge" axis="0 1 0" damping=".001"/>'
                      f'<geom type="sphere" size="{WHEEL_R}" rgba="0 0 0 0" friction="1.5 .005 .0001" mass=".05"/></body>')
 
XML = f"""
<mujoco>
  <option timestep="0.002"/>
  <visual><global offwidth="{CAM_W}" offheight="{CAM_H}"/></visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".85 .85 .85" rgb2=".75 .75 .75" width="300" height="300"/>
    <material name="floor" texture="grid" texrepeat="8 8"/>
    <texture name="tag" type="2d" file="marker.png"/><material name="tag" texture="tag"/>
  </asset>
  <worldbody>
    <light pos="0 0 4" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="3 3 .1" material="floor"/>
    <geom type="box" pos=" 2.5 0 .25" size=".05 2.5 .25" rgba=".6 .65 .7 1"/>
    <geom type="box" pos="-2.5 0 .25" size=".05 2.5 .25" rgba=".6 .65 .7 1"/>
    <geom type="box" pos="0  2.5 .25" size="2.5 .05 .25" rgba=".6 .65 .7 1"/>
    <geom type="box" pos="0 -2.5 .25" size="2.5 .05 .25" rgba=".6 .65 .7 1"/>
    {obstacle_geoms}
    <geom type="box" pos="{MARKER_XY[0]} 0 .13" size=".005 .15 .15" material="tag" contype="0" conaffinity="0"/>
    <body name="robot" pos="{TRUE_START[0]} {TRUE_START[1]} .034" euler="0 0 {np.degrees(TRUE_START[2])}">
      <freejoint/>
      <geom type="box" size=".09 .09 .04" pos="-.03 0 .02" rgba=".15 .15 .15 1" mass="0" contype="0" conaffinity="0"/>
      <geom type="box" size=".07 .07 .03" pos="-.03 0 .025" rgba="0 0 0 0" mass=".8" contype="0" conaffinity="0"/>
      <geom type="sphere" size=".012" pos="-.06 0 -.021" friction="0 0 0" priority="1" condim="1" mass=".01" rgba="0 0 0 0"/>
      {wheel("l", WHEEL_SEP / 2)}{wheel("r", -WHEEL_SEP / 2)}
      <geom type="cylinder" size=".04 .015" pos="0 0 .075" rgba=".2 .4 .95 1" mass="0" contype="0" conaffinity="0"/>
      <geom type="box" size=".012 .025 .015" pos=".045 0 .08" rgba=".9 .15 .15 1" mass="0" contype="0" conaffinity="0"/>
      <camera name="cam" pos=".07 0 .095" xyaxes="0 -1 0 0 0 1" fovy="{CAM_FOVY}"/>
      {lidar_sites}
    </body>
  </worldbody>
  <actuator>
    <velocity joint="wl" kv=".5" ctrlrange="-6.6 6.6" forcerange="-.05 .05"/>
    <velocity joint="wr" kv=".5" ctrlrange="-6.6 6.6" forcerange="-.05 .05"/>
  </actuator>
  <sensor><jointpos name="el" joint="wl"/><jointpos name="er" joint="wr"/>{lidar_sensors}</sensor>
</mujoco>"""
 
model = mujoco.MjModel.from_xml_string(XML, assets={"marker.png": marker_png()})
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, CAM_H, CAM_W)
hide_rays = mujoco.MjvOption()
hide_rays.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False
 
# ---------------------------------------------------------------------------
# 2. SENSING: wheel encoders, LiDAR, camera (ArUco + PnP, Week 9)
# ---------------------------------------------------------------------------
LIDAR_ANGLES = np.arctan2(np.sin(2 * np.pi * np.arange(N_LIDAR) / N_LIDAR),
                          np.cos(2 * np.pi * np.arange(N_LIDAR) / N_LIDAR))
K_CAM, DIST_COEFFS = camera_intrinsics_from_fov(CAM_W, CAM_H, CAM_FOVY)
detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
corners_3d = MARKER_SIZE / 2 * np.array([[-1, 1, 0], [1, 1, 0], [1, -1, 0], [-1, -1, 0]], np.float32)
SENSOR_PROCESSOR = SensorProcessor(NoiseConfig(), seed=SCENE_SEED)
ACTIVE_WHEEL_SCALES = (1.0, 1.0)
 
 
def read_encoders():
    """Wheel angles (rad) after the configured sensor model."""
    raw = np.array([data.sensor("el").data[0], data.sensor("er").data[0]])
    scaled = raw * np.asarray(ACTIVE_WHEEL_SCALES)
    return np.array([SENSOR_PROCESSOR.encoder_angle(angle) for angle in scaled])
 
 
def read_lidar():
    """Configured noisy scan; missing returns use the sensor's -1 sentinel."""
    raw = np.array([data.sensor(f"r{k}").data[0] for k in range(N_LIDAR)])
    return SENSOR_PROCESSOR.maybe_drop_lidar(SENSOR_PROCESSOR.lidar_scan(raw))
 
 
def read_marker():
    """Range (m) and bearing (rad, + = left) to the marker from the camera, or None."""
    renderer.update_scene(data, camera="cam", scene_option=hide_rays)
    gray = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return None
    ok, _, t = cv2.solvePnP(
        corners_3d, corners[0].reshape(4, 2), K_CAM, DIST_COEFFS,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not ok or not np.all(np.isfinite(t)):
        return None
    r, b = np.hypot(t[0, 0], t[2, 0]), np.arctan2(-t[0, 0], t[2, 0])
    if r >= 3.0:                                      # ignore far readings (marker too small)
        return None
    measurement = SENSOR_PROCESSOR.camera_measurement(r, b)
    return np.asarray(measurement) if measurement is not None else None
 
 
# ---------------------------------------------------------------------------
# 3. ESTIMATION: EKF (Week 7)
# ---------------------------------------------------------------------------
def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi
 
 
def f_uni(x, u):
    """Unicycle motion over one sample. x = [x, y, th], u = [v, w]."""
    v, w = u
    th = x[2] + w * T / 2
    return np.array([x[0] + v * T * np.cos(th), x[1] + v * T * np.sin(th), wrap(x[2] + w * T)])
 
 
def F_uni(x, u):
    """Jacobian of f_uni with respect to x."""
    v, w = u
    th = x[2] + w * T / 2
    return np.array([[1, 0, -v * T * np.sin(th)], [0, 1, v * T * np.cos(th)], [0, 0, 1]])
 
 
CAM_OFFSET = 0.07                       # camera is 7 cm in front of the robot centre
MARKER_FACE = MARKER_XY - [0.005, 0]
 
 
def h_rb(x):
    """Expected range and bearing to the marker from pose x."""
    dx = MARKER_FACE[0] - (x[0] + CAM_OFFSET * np.cos(x[2]))
    dy = MARKER_FACE[1] - (x[1] + CAM_OFFSET * np.sin(x[2]))
    return np.array([np.hypot(dx, dy), wrap(np.arctan2(dy, dx) - x[2])])
 
 
def H_rb(x):
    """Jacobian of h_rb with respect to x (numerical, 2x3)."""
    return np.array([(h_rb(x + d) - h_rb(x - d)) / 2e-6 for d in 1e-6 * np.eye(3)]).T
 
 
Q_D = np.diag([0.001 ** 2, 0.001 ** 2, np.radians(0.3) ** 2])   # wheel slip per sample
 
 
def ekf_step(xh, P, u, z):
    # predict with the wheel encoders
    F = F_uni(xh, u)
    xh = f_uni(xh, u)
    P = F @ P @ F.T + Q_D
    # correct with the camera (only when the marker is seen)
    if z is not None:
        H = H_rb(xh)
        nu = z - h_rb(xh)
        nu[1] = wrap(nu[1])
        R = np.diag([
            SENSOR_PROCESSOR.config.camera_range_sigma_m ** 2,
            SENSOR_PROCESSOR.config.camera_bearing_sigma_rad ** 2,
        ])
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        xh = xh + K @ nu
        xh[2] = wrap(xh[2])
        P = (np.eye(3) - K @ H) @ P
    return xh, P
 
 
# ---------------------------------------------------------------------------
# 4. CONTROL: LQR on the path error model (Weeks 5-6) + LiDAR obstacle check
# ---------------------------------------------------------------------------
def lqr_gain(V):
    Ad = np.array([[1, V * T], [0, 1]])                  # error model e = [e_y, e_th]
    Bd = np.array([[V * T ** 2 / 2], [T]])
    Q = np.diag([1 / 0.05 ** 2, 1 / np.radians(10) ** 2])   # Bryson's rule: 5 cm, 10 deg
    R = np.array([[1.0]])                                    # 1 rad/s steering
    S = scipy.linalg.solve_discrete_are(Ad, Bd, Q, R)
    return np.linalg.solve(R + Bd.T @ S @ Bd, Bd.T @ S @ Ad).ravel()
 
 
K_LQR = lqr_gain(0.15)
 
 
def is_free(scan, angle, width=0.23, length=0.7):
    """True if no LiDAR hit lies in a robot-wide corridor in this direction."""
    px, py = scan * np.cos(LIDAR_ANGLES), scan * np.sin(LIDAR_ANGLES)
    along = px * np.cos(angle) + py * np.sin(angle)
    across = np.abs(px * np.sin(angle) - py * np.cos(angle))
    return not np.any((along > 0) & (along < length) & (across < width))
 
 
def controller(xh, scan, mode):
    """Returns v, w, mode. Uses only the EKF estimate and the LiDAR."""
    x, y, th = xh
    valid_scan = np.isfinite(scan) & (scan > 0.0)
    if not valid_scan.any():
        return 0.0, 0.0, mode
    scan = np.where(valid_scan, scan, 3.5)
    front = scan[np.abs(LIDAR_ANGLES) < np.radians(30)].min()
    if mode == "NAVIGATE" and np.hypot(*(PRE_DOCK - xh[:2])) < 0.25:
        mode = "DOCK"
    if mode == "NAVIGATE":
        goal = wrap(np.arctan2(PRE_DOCK[1] - y, PRE_DOCK[0] - x) - th)    # goal direction, robot frame
        free = [a for a in np.r_[goal, LIDAR_ANGLES] if is_free(scan, a)]
        if not free:
            return 0.0, 0.6, mode
        heading = min(free, key=lambda a: abs(wrap(a - goal)))          # free direction closest to goal
        e = np.array([0.0, -heading])
        v = 0.15 * np.cos(heading) ** 2 * np.clip((front - 0.2) / 0.4, 0, 1)
    else:                                                                # DOCK: follow the line y = 0
        e = np.array([y, th])
        v = np.clip(0.5 * (h_rb(xh)[0] - DOCK_DIST), 0, 0.1) * np.cos(th) ** 2
    w = np.clip(-K_LQR @ e, -1.2, 1.2)
    return v, w, mode
 
 
# ---------------------------------------------------------------------------
# 5. MAIN LOOP
# ---------------------------------------------------------------------------
def set_wheels(v, w):
    data.ctrl[:] = [(v - w * WHEEL_SEP / 2) / WHEEL_R, (v + w * WHEEL_SEP / 2) / WHEEL_R]
 
 
def true_pose():
    R = data.body("robot").xmat.reshape(3, 3)
    return np.array([*data.body("robot").xpos[:2], np.arctan2(R[1, 0], R[0, 0])])
 

def _robot_body_ids():
    root_id = model.body("robot").id
    body_ids = set()
    for body_id in range(model.nbody):
        parent_id = body_id
        while parent_id != 0 and parent_id != root_id:
            parent_id = int(model.body_parentid[parent_id])
        if parent_id == root_id:
            body_ids.add(body_id)
    return body_ids


def _touching_obstacle(robot_body_ids, obstacle_geom_ids):
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        if ((geom1 in obstacle_geom_ids and int(model.geom_bodyid[geom2]) in robot_body_ids)
                or (geom2 in obstacle_geom_ids and int(model.geom_bodyid[geom1]) in robot_body_ids)):
            return True
    return False


def run(live=True, seed=SCENE_SEED, disturbance=None, noise_config=None):
    """Run one simulation trial; truth is recorded only for evaluation."""
    global SENSOR_PROCESSOR, ACTIVE_WHEEL_SCALES

    disturbance = disturbance or Disturbance()
    disturbance.validate()
    base_noise = noise_config or NoiseConfig()
    trial_noise = NoiseConfig(
        encoder_sigma_rad=base_noise.encoder_sigma_rad,
        lidar_sigma_m=base_noise.lidar_sigma_m,
        camera_range_sigma_m=base_noise.camera_range_sigma_m,
        camera_bearing_sigma_rad=base_noise.camera_bearing_sigma_rad,
        camera_pixel_sigma_px=base_noise.camera_pixel_sigma_px,
        encoder_bias_rad_per_sample=base_noise.encoder_bias_rad_per_sample,
        lidar_bias_m=base_noise.lidar_bias_m + disturbance.lidar_bias_m,
        camera_dropout_probability=max(
            base_noise.camera_dropout_probability,
            disturbance.camera_dropout_probability,
        ),
        lidar_dropout_probability=max(
            base_noise.lidar_dropout_probability,
            disturbance.lidar_dropout_probability,
        ),
        outlier_probability=base_noise.outlier_probability,
        outlier_scale=base_noise.outlier_scale,
    )
    SENSOR_PROCESSOR = SensorProcessor(trial_noise, seed=seed)
    ACTIVE_WHEEL_SCALES = (
        disturbance.wheel_scale_left,
        disturbance.wheel_scale_right,
    )
    mujoco.mj_resetData(model, data)
    start = TRUE_START + np.array([
        disturbance.start_x_offset_m,
        disturbance.start_y_offset_m,
        disturbance.start_heading_offset_rad,
    ])
    data.qpos[:3] = [start[0], start[1], 0.034]
    data.qpos[3:7] = [np.cos(start[2] / 2), 0.0, 0.0, np.sin(start[2] / 2)]
    mujoco.mj_forward(model, data)

    for _ in range(250):                                   # let the robot settle on its wheels
        mujoco.mj_step(model, data)
    xh, P = START_POSE.copy(), np.diag([0.1 ** 2, 0.1 ** 2, np.radians(7) ** 2])
    enc_prev, mode = read_encoders(), "NAVIGATE"
    log_est, log_true, log_covariance = [], [], []
    marker_detections = 0
    collision_steps = 0
    docked = False
    robot_body_ids = _robot_body_ids()
    obstacle_geom_ids = {
        model.geom(f"obstacle{index}").id for index in range(len(OBSTACLES))
    }
 
    if live:
        viewer = mujoco.viewer.launch_passive(model, data)
        viewer.cam.lookat[:], viewer.cam.distance, viewer.cam.elevation, viewer.cam.azimuth = [0, 0, 0], 7.5, -90, 90
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False
        plt.ion()
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot([-2.5, 2.5, 2.5, -2.5, -2.5], [-2.5, -2.5, 2.5, 2.5, -2.5], "k-")
        ax.plot(*MARKER_XY, "gs", ms=10, label="marker")
        dots, = ax.plot([], [], "r.", ms=3, label="LiDAR")
        est_dot, = ax.plot([], [], "bo", label="EKF estimate")
        true_dot, = ax.plot([], [], "o", color="0.5", label="true pose")
        uncertainty = Ellipse((xh[0], xh[1]), 0, 0, fill=False, color="b",
                              linestyle=":", label="EKF 95% position")
        ax.add_patch(uncertainty)
        ax.set_xlim(-2.6, 2.6); ax.set_ylim(-2.6, 2.6); ax.set_aspect("equal"); ax.legend(loc="lower right")
 
    for k in range(1800):                                  # up to 90 s
        t0 = time.time()
        # sense
        enc = read_encoders()
        d_l, d_r = (enc - enc_prev) * WHEEL_R
        d_l += disturbance.encoder_bias_rad_per_sample * WHEEL_R
        d_r += disturbance.encoder_bias_rad_per_sample * WHEEL_R
        enc_prev = enc
        u = np.array([(d_l + d_r) / 2 / T, (d_r - d_l) / WHEEL_SEP / T])   # measured v, w
        scan = read_lidar()
        z = read_marker()
        marker_detections += int(z is not None)
        # estimate
        xh, P = ekf_step(xh, P, u, z)
        # control
        v, w, mode = controller(xh, scan, mode)
        set_wheels(v, w)
        collided_this_step = False
        for _ in range(int(T / model.opt.timestep)):
            mujoco.mj_step(model, data)
            collided_this_step |= _touching_obstacle(robot_body_ids, obstacle_geom_ids)
        collision_steps += int(collided_this_step)
        log_est.append(xh.copy()); log_true.append(true_pose()); log_covariance.append(np.diag(P).copy())
 
        if k % 20 == 0:
            print(f"t={k*T:4.1f}s  {mode:8s}  EKF x={xh[0]:+.2f} y={xh[1]:+.2f}  "
                  f"marker={'seen' if z is not None else '-':4s}  "
                  f"closest obstacle {np.where(scan > 0, scan, 3.5).min():.2f} m")
        if live:
            if not viewer.is_running():
                break
            viewer.sync()
            if k % 5 == 0:
                display_scan = np.where(scan > 0, scan, 3.5)
                dots.set_data(xh[0] + display_scan * np.cos(LIDAR_ANGLES + xh[2]),
                              xh[1] + display_scan * np.sin(LIDAR_ANGLES + xh[2]))
                est_dot.set_data([xh[0]], [xh[1]]); true_dot.set_data([log_true[-1][0]], [log_true[-1][1]])
                eigenvalues, eigenvectors = np.linalg.eigh(P[:2, :2])
                order = np.argsort(eigenvalues)[::-1]
                major = eigenvectors[:, order[0]]
                uncertainty.center = (xh[0], xh[1])
                uncertainty.width = 2 * np.sqrt(5.991 * max(eigenvalues[order[0]], 0.0))
                uncertainty.height = 2 * np.sqrt(5.991 * max(eigenvalues[order[1]], 0.0))
                uncertainty.angle = np.degrees(np.arctan2(major[1], major[0]))
                ax.set_title(f"{mode}   t = {k*T:.1f} s"); plt.pause(0.001)
            time.sleep(max(0, T - (time.time() - t0)))
        if mode == "DOCK" and z is not None and abs(z[0] - DOCK_DIST) < 0.03 and abs(z[1]) < 0.05:
            print(f"DOCKED at t={k*T:.1f}s")
            docked = True
            break
 
    if live:
        viewer.close(); plt.ioff(); plt.close(fig)
    estimated = np.asarray(log_est)
    truth = np.asarray(log_true)
    metrics = pose_error_metrics(estimated, truth)
    metrics.update({
        "success": docked,
        "collisions": collision_steps,
        "marker_visible_fraction": marker_detections / max(len(log_est), 1),
    })
    return {
        **metrics,
        "estimated_poses": estimated,
        "true_poses": truth,
        "covariances": np.asarray(log_covariance),
        "docked": docked,
    }
 
"""
41014 Integrated Project - Track A basic demo (TurtleBot3-style robot in MuJoCo)

What this shows (the "basic simulation" milestone):
  * a 5 m x 5 m room with walls, RANDOM obstacles (new layout every run) and an ArUco
    charging-station marker; one obstacle is always placed in the robot's way
  * a differential-drive robot with TurtleBot3 Burger dimensions
  * all three sensors working: wheel encoders, front camera (ArUco + PnP), 72-beam LiDAR
  * the loop: SENSE -> ESTIMATE (EKF: odometry + camera) -> CONTROL -> DRIVE
  * the robot's real start pose is randomised; the EKF only knows the nominal one
  * live top-down plot: LiDAR scan, EKF estimate + uncertainty ellipse, odometry-only, truth
  * control has two modes:
      NAVIGATE: head for a point in front of the dock, steering around obstacles with the LiDAR
      DOCK:     final approach using the camera (Week 9 D3 law), stop 0.40 m from the marker
  * top-down view of the room in the MuJoCo window
  * plots of estimated vs true position at the end

Rule from the brief: the controller uses only sensor readings / the estimate.
Ground truth (true_pose) is used ONLY for logging and plots.

How to run (Windows, inside your .venv):
    pip install mujoco numpy opencv-contrib-python matplotlib
    python trackA_demo.py              # live 3D window + camera window + plots
    python trackA_demo.py --headless   # no windows during the run, plots only
    python trackA_demo.py --seed 7     # repeat a specific random obstacle layout

Mac users: run with  mjpython trackA_demo.py  (MuJoCo's viewer needs it on macOS).

Next steps (TODO markers below): LQR, tuning Q/R from measured sensor noise.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import cv2
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Shared facts (keep these the same across the team)
# ---------------------------------------------------------------------------
WHEEL_R = 0.033            # wheel radius (m), TurtleBot3 Burger
WHEEL_SEP = 0.160          # distance between wheels (m)
MARKER_ID = 0              # ArUco DICT_4X4_50, id 0
MARKER_SIZE = 0.20         # black square side (m)
MARKER_XY = np.array([2.44, 0.0])   # marker position in the world (for the EKF later)
DOCK_DIST = 0.40           # stop this far in front of the marker (m)
PRE_DOCK = MARKER_XY - np.array([DOCK_DIST + 0.5, 0.0])   # navigate here first, then dock
START_POSE = np.array([-1.5, -0.5, np.radians(30)])   # x, y, theta
DT_CTRL = 0.05             # control/estimation period (s), 20 Hz
CAM_W, CAM_H, CAM_FOVY = 640, 480, 60.0
N_LIDAR = 72               # one beam every 5 degrees

# Sensor noise (perception teammate: replace with measured values)
NOISE_ENC = 0.002          # rad
NOISE_LIDAR = 0.01         # m

N_OBSTACLES = 4           # how many random obstacles to place

HERE = Path(__file__).resolve().parent

_ap = argparse.ArgumentParser()
_ap.add_argument("--headless", action="store_true", help="no windows during the run")
_ap.add_argument("--seed", type=int, default=None,
                 help="random seed for obstacle placement (same seed = same layout)")
ARGS, _ = _ap.parse_known_args()
SEED = ARGS.seed if ARGS.seed is not None else int(np.random.SeedSequence().entropy % 10000)
rng = np.random.default_rng(SEED)


# ---------------------------------------------------------------------------
# 1. WORLD: marker image + MuJoCo scene
# ---------------------------------------------------------------------------
def make_marker_png(path, px=400, border_frac=0.25):
    """Write the ArUco marker (with white border) as a texture image."""
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    tag = cv2.aruco.generateImageMarker(d, MARKER_ID, px)
    b = int(px * border_frac)
    img = cv2.copyMakeBorder(tag, b, b, b, b, cv2.BORDER_CONSTANT, value=255)
    img = cv2.flip(img, 1)   # MuJoCo maps this texture mirrored on the box face, so pre-flip it
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    return MARKER_SIZE * (1 + 2 * border_frac)   # full panel side (m)


def random_obstacles(n=N_OBSTACLES, seed=SEED):
    """Place n obstacles (cylinders or boxes) at random, keeping clear of the walls,
    the robot's start, the dock and each other. The FIRST obstacle is always put
    roughly on the straight line from the start to the dock, so the robot has to
    steer around it.
    """
    KEEP_PATH_CLEAR = False
    g = np.random.default_rng(seed)
    start, dock = START_POSE[:2], MARKER_XY - np.array([DOCK_DIST, 0.0])
    obstacles = []
    # obstacle in the way: somewhere in the middle of the start -> pre-dock line
    t = g.uniform(0.35, 0.65)
    p = start + t * (PRE_DOCK - start) + g.uniform(-0.1, 0.1, size=2)
    obstacles.append({"kind": str(g.choice(["cylinder", "box"])), "pos": p,
                      "size": g.uniform(0.15, 0.22), "yaw": g.uniform(0, 180),
                      "height": g.uniform(0.15, 0.3)})
    tries = 0
    while len(obstacles) < n and tries < 5000:
        tries += 1
        kind = g.choice(["cylinder", "box"])
        size = g.uniform(0.10, 0.22)                     # radius / half-width (m)
        x, y = g.uniform(-2.5 + 0.3 + size, 2.5 - 0.3 - size, size=2)
        p = np.array([x, y])
        if np.linalg.norm(p - start) < size + 0.5:       # clear of the start
            continue
        if np.linalg.norm(p - dock) < size + 0.6 or x > 1.6:   # clear of the dock area
            continue
        if KEEP_PATH_CLEAR:                              # clear of the start->dock line
            d = dock - start
            t = np.clip(np.dot(p - start, d) / np.dot(d, d), 0, 1)
            if np.linalg.norm(p - (start + t * d)) < size + 0.35:
                continue
        if any(np.linalg.norm(p - o["pos"]) < size + o["size"] + 0.3 for o in obstacles):
            continue
        obstacles.append({"kind": str(kind), "pos": p, "size": size,
                          "yaw": g.uniform(0, 180), "height": g.uniform(0.15, 0.3)})
    return obstacles


OBSTACLES = random_obstacles()

# Randomise where the robot REALLY starts. The EKF only knows the nominal START_POSE,
# so it starts with an error that the camera measurements must correct.
_g_start = np.random.default_rng(SEED + 1000)
TRUE_START = START_POSE + np.array([*_g_start.uniform(-0.15, 0.15, 2), np.radians(_g_start.uniform(-10, 10))])


def obstacles_xml(obstacles):
    colours = [".8 .3 .3 1", ".8 .5 .2 1", ".3 .6 .3 1", ".5 .4 .7 1", ".3 .5 .8 1", ".7 .7 .3 1"]
    out = []
    for i, o in enumerate(obstacles):
        x, y = o["pos"]; h = o["height"]; c = colours[i % len(colours)]
        if o["kind"] == "cylinder":
            out.append(f'<geom name="obs{i}" type="cylinder" pos="{x:.3f} {y:.3f} {h:.3f}" '
                       f'size="{o["size"]:.3f} {h:.3f}" rgba="{c}"/>')
        else:
            out.append(f'<geom name="obs{i}" type="box" pos="{x:.3f} {y:.3f} {h:.3f}" '
                       f'size="{o["size"]:.3f} {o["size"]:.3f} {h:.3f}" euler="0 0 {o["yaw"]:.1f}" rgba="{c}"/>')
    return "\n    ".join(out)


def build_xml(marker_png, panel):
    lidar_sites = "\n      ".join(
        f'<site name="lidar{k}" pos="0 0 0.09" '
        f'zaxis="{np.cos(2*np.pi*k/N_LIDAR):.4f} {np.sin(2*np.pi*k/N_LIDAR):.4f} 0" '
        f'size="0.005" rgba="1 0 0 0"/>' for k in range(N_LIDAR))
    lidar_sensors = "\n    ".join(
        f'<rangefinder name="r{k}" site="lidar{k}" cutoff="3.5"/>' for k in range(N_LIDAR))
    wheel = lambda side, y: f"""
      <body name="wheel_{side}" pos="0 {y} 0">
        <joint name="wheel_{side}" type="hinge" axis="0 1 0" damping="0.001"/>
        <geom type="sphere" size="{WHEEL_R}" rgba="0 0 0 0" friction="1.5 .005 .0001" mass="0.05"/>
      </body>"""
    return f"""
<mujoco model="trackA">
  <option timestep="0.002"/>
  <visual><global offwidth="{CAM_W}" offheight="{CAM_H}"/></visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".85 .85 .85" rgb2=".75 .75 .75"
             width="300" height="300"/>
    <material name="floor" texture="grid" texrepeat="8 8"/>
    <texture name="aruco" type="2d" file="{marker_png.as_posix()}"/>
    <material name="aruco" texture="aruco"/>
    <material name="wall" rgba=".6 .65 .7 1"/>
  </asset>

  <worldbody>
    <light pos="0 0 4" dir="0 0 -1" diffuse=".8 .8 .8"/>
    <light pos="2 2 3" dir="-1 -1 -1" diffuse=".3 .3 .3"/>
    <geom name="floor" type="plane" size="3 3 .1" material="floor"/>

    <!-- room 5 m x 5 m -->
    <geom type="box" pos=" 2.5 0 .25" size=".05 2.5 .25" material="wall"/>
    <geom type="box" pos="-2.5 0 .25" size=".05 2.5 .25" material="wall"/>
    <geom type="box" pos="0  2.5 .25" size="2.5 .05 .25" material="wall"/>
    <geom type="box" pos="0 -2.5 .25" size="2.5 .05 .25" material="wall"/>

    <!-- obstacles (random, see random_obstacles) -->
    {obstacles_xml(OBSTACLES)}

    <!-- charging station: ArUco panel on the +x wall, centre at camera height -->
    <body name="dock" pos="{MARKER_XY[0]} {MARKER_XY[1]} 0.13">
      <geom type="box" size="0.005 {panel/2:.4f} {panel/2:.4f}" material="aruco"
            contype="0" conaffinity="0"/>
    </body>

    <!-- robot -->
    <body name="base" pos="{TRUE_START[0]} {TRUE_START[1]} 0.034" euler="0 0 {np.degrees(TRUE_START[2]):.3f}">
      <freejoint name="root"/>
      <!-- what you SEE: one plain rectangle (massless, no collisions) -->
      <geom name="robot_box" type="box" size=".09 .09 .04" pos="-.03 0 .02" rgba=".15 .15 .15 1"
            mass="0" contype="0" conaffinity="0"/>
      <!-- physics only (invisible): body mass, caster and wheels below -->
      <geom name="chassis" type="box" size=".07 .07 .03" pos="-.03 0 .025" rgba="0 0 0 0"
            mass="0.8" contype="0" conaffinity="0"/>
      <!-- priority=1: MuJoCo uses the caster's zero friction instead of the floor's -->
      <geom name="caster" type="sphere" size=".012" pos="-.06 0 -.021" friction="0 0 0"
            priority="1" condim="1" mass="0.01" rgba="0 0 0 0"/>
      {wheel("l", WHEEL_SEP/2)}
      {wheel("r", -WHEEL_SEP/2)}
      <camera name="front" pos=".07 0 .095" xyaxes="0 -1 0 0 0 1" fovy="{CAM_FOVY}"/>
      {lidar_sites}
      <!-- sensors you can see: LiDAR (blue puck on top) and camera (red block at the front) -->
      <geom name="lidar_vis" type="cylinder" size=".04 .015" pos="0 0 .075" rgba=".2 .4 .95 1"
            contype="0" conaffinity="0" mass="0.01"/>
      <geom name="camera_vis" type="box" size=".012 .025 .015" pos=".045 0 .08" rgba=".9 .15 .15 1"
            contype="0" conaffinity="0" mass="0"/>
    </body>
  </worldbody>

  <actuator>
    <velocity name="v_l" joint="wheel_l" kv="0.5" ctrlrange="-6.6 6.6" forcerange="-0.05 0.05"/>
    <velocity name="v_r" joint="wheel_r" kv="0.5" ctrlrange="-6.6 6.6" forcerange="-0.05 0.05"/>
  </actuator>

  <sensor>
    <jointpos name="enc_l" joint="wheel_l"/>
    <jointpos name="enc_r" joint="wheel_r"/>
    {lidar_sensors}
  </sensor>
</mujoco>
"""


marker_png = HERE / "marker.png"
panel = make_marker_png(marker_png)
model = mujoco.MjModel.from_xml_string(build_xml(marker_png, panel))
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, CAM_H, CAM_W)
N_SUB = int(round(DT_CTRL / model.opt.timestep))   # physics steps per control step

# camera intrinsics from the field of view (pinhole model)
f_px = (CAM_H / 2) / np.tan(np.radians(CAM_FOVY) / 2)
K = np.array([[f_px, 0, CAM_W / 2], [0, f_px, CAM_H / 2], [0, 0, 1]])
DIST = np.zeros(5)


# ---------------------------------------------------------------------------
# 2. SENSORS  (perception)  - the only things control is allowed to use
# ---------------------------------------------------------------------------
def read_encoders():
    """Wheel angles (rad), with noise."""
    return (data.sensor("enc_l").data[0] + rng.normal(0, NOISE_ENC),
            data.sensor("enc_r").data[0] + rng.normal(0, NOISE_ENC))


def read_lidar():
    """Ranges (m) at 0, 10, 20 ... 350 deg in the robot frame. -1 = nothing hit."""
    r = np.array([data.sensor(f"r{k}").data[0] for k in range(N_LIDAR)])
    hit = r > 0
    r[hit] += rng.normal(0, NOISE_LIDAR, hit.sum())
    return r


_scene_opt = mujoco.MjvOption()
_scene_opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False   # keep LiDAR rays out of the camera image


def read_camera():
    """RGB image from the robot's front camera."""
    renderer.update_scene(data, camera="front", scene_option=_scene_opt)
    return renderer.render()


_detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
_obj = 0.5 * MARKER_SIZE * np.array([[-1, 1, 0], [1, 1, 0], [1, -1, 0], [-1, -1, 0]], dtype=np.float32)


def detect_marker(img):
    """Return (range, bearing, corners) of the marker, or None if not seen.
    range in m, bearing in rad (positive = marker to the LEFT of the robot)."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    corners, ids, _ = _detector.detectMarkers(gray)
    if ids is None or MARKER_ID not in ids.ravel():
        return None
    c = corners[list(ids.ravel()).index(MARKER_ID)].reshape(4, 2).astype(np.float32)
    ok, rvec, tvec = cv2.solvePnP(_obj, c, K, DIST, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok or not np.all(np.isfinite(tvec)):    # PnP occasionally fails: treat as not seen
        return None
    x_c, z_c = tvec[0, 0], tvec[2, 0]               # camera frame: x right, z forward
    return np.hypot(x_c, z_c), np.arctan2(-x_c, z_c), c


# ---------------------------------------------------------------------------
# 3. ESTIMATION  (odometry motion model + EKF)
# ---------------------------------------------------------------------------
def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def odometry_update(pose, d_l, d_r):
    """Motion model: pose = [x, y, theta]; d_l, d_r = wheel angle change (rad)."""
    s_l, s_r = d_l * WHEEL_R, d_r * WHEEL_R
    ds, dth = (s_l + s_r) / 2, (s_r - s_l) / WHEEL_SEP
    x, y, th = pose
    th_new = (th + dth + np.pi) % (2 * np.pi) - np.pi
    return np.array([x + ds * np.cos(th + dth / 2), y + ds * np.sin(th + dth / 2), th_new])


CAM_OFFSET = 0.07                                  # camera is 7 cm in front of the robot centre
MARKER_FACE = MARKER_XY - np.array([0.005, 0.0])   # front face of the marker panel


class EKF:
    """Extended Kalman Filter, state = [x, y, theta].
       predict: wheel-encoder odometry (motion model above)
       update : marker range & bearing from the camera (known marker position in the map)"""

    def __init__(self, x0, P0):
        self.x = np.array(x0, float)
        self.P = np.array(P0, float)
        self.n_updates = 0
        self.n_rejected = 0

    # ---- tuning (rough) ----
    SLIP = 0.03                                  # wheel slip: 3 % of distance travelled
    ENC_SIGMA = NOISE_ENC * WHEEL_R * np.sqrt(2)  # encoder noise on one step's wheel distance (m)

    def R_meas(self, r):
        """Camera noise: range error grows with distance, bearing about 1 degree."""
        return np.diag([(0.02 + 0.02 * r) ** 2, np.radians(1.0) ** 2])

    def predict(self, d_l, d_r):
        s_l, s_r = d_l * WHEEL_R, d_r * WHEEL_R
        ds, dth = (s_l + s_r) / 2, (s_r - s_l) / WHEEL_SEP
        th = self.x[2]
        phi = th + dth / 2
        b = WHEEL_SEP
        # Jacobian wrt the state
        F = np.array([[1, 0, -ds * np.sin(phi)],
                      [0, 1,  ds * np.cos(phi)],
                      [0, 0,  1]])
        # Jacobian wrt the two wheel distances (s_l, s_r)
        G = np.array([[0.5 * np.cos(phi) + ds * np.sin(phi) / (2 * b), 0.5 * np.cos(phi) - ds * np.sin(phi) / (2 * b)],
                      [0.5 * np.sin(phi) - ds * np.cos(phi) / (2 * b), 0.5 * np.sin(phi) + ds * np.cos(phi) / (2 * b)],
                      [-1 / b, 1 / b]])
        M = np.diag([(self.ENC_SIGMA + self.SLIP * abs(s_l)) ** 2,
                     (self.ENC_SIGMA + self.SLIP * abs(s_r)) ** 2])
        self.x = odometry_update(self.x, d_l, d_r)
        self.P = F @ self.P @ F.T + G @ M @ G.T

    def predicted_measurement(self, x=None):
        """What the camera SHOULD see from state x: (range, bearing) and the Jacobian H."""
        x, y, th = self.x if x is None else x
        cx, cy = x + CAM_OFFSET * np.cos(th), y + CAM_OFFSET * np.sin(th)   # camera position
        dx, dy = MARKER_FACE[0] - cx, MARKER_FACE[1] - cy
        q = dx * dx + dy * dy
        r = np.sqrt(q)
        ddx_dth, ddy_dth = CAM_OFFSET * np.sin(th), -CAM_OFFSET * np.cos(th)
        H = np.array([[-dx / r, -dy / r, (dx * ddx_dth + dy * ddy_dth) / r],
                      [ dy / q, -dx / q, (dx * ddy_dth - dy * ddx_dth) / q - 1]])
        return np.array([r, wrap(np.arctan2(dy, dx) - th)]), H

    def update(self, z):
        """z = (range, bearing) from the camera."""
        z_hat, H = self.predicted_measurement()
        nu = np.array([z[0] - z_hat[0], wrap(z[1] - z_hat[1])])   # innovation
        S = H @ self.P @ H.T + self.R_meas(z[0])
        if nu @ np.linalg.solve(S, nu) > 13.8:                  # chi-square gate (99.9 %, 2 dof)
            self.n_rejected += 1
            return False                                        # outlier: ignore it
        K_gain = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K_gain @ nu
        self.x[2] = wrap(self.x[2])
        self.P = (np.eye(3) - K_gain @ H) @ self.P
        self.n_updates += 1
        return True


# ---------------------------------------------------------------------------
# 4. CONTROL  (LiDAR obstacle avoidance + camera docking)
# ---------------------------------------------------------------------------
LIDAR_ANGLES = (2 * np.pi * np.arange(N_LIDAR) / N_LIDAR + np.pi) % (2 * np.pi) - np.pi
ROBOT_RADIUS = 0.11        # m, robot footprint radius
CLEARANCE = 0.12           # m, extra gap to keep from obstacles
LOOKAHEAD = 0.70           # m, how far ahead a direction must be free
V_MAX, W_MAX = 0.15, 1.2   # m/s, rad/s


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def direction_is_free(scan_xy, angle):
    """True if a corridor as wide as the robot, LOOKAHEAD long, in this direction has no LiDAR hits."""
    u = np.array([np.cos(angle), np.sin(angle)])
    along = scan_xy @ u                                          # distance along the corridor
    across = np.abs(scan_xy[:, 0] * u[1] - scan_xy[:, 1] * u[0])  # distance sideways
    return not np.any((along > 0) & (along < LOOKAHEAD) & (across < ROBOT_RADIUS + CLEARANCE))


class Controller:
    """Two modes:
       NAVIGATE - drive towards PRE_DOCK (using the ESTIMATED pose) and pick the free
                  LiDAR direction closest to the goal direction, so the robot goes
                  around obstacles.
       DOCK     - final approach: D3-style state feedback on the EKF estimate.
       Uses only the estimate and sensor data - never ground truth."""

    def __init__(self):
        self.mode = "NAVIGATE"
        self.prev_dir = 0.0

    def __call__(self, est, meas, scan):
        r = np.where(scan > 0, scan, 3.5)                       # no hit = far away
        scan_xy = r[:, None] * np.c_[np.cos(LIDAR_ANGLES), np.sin(LIDAR_ANGLES)]
        front = r[np.abs(LIDAR_ANGLES) < np.radians(30)].min()  # closest thing ahead
        x, y, th = est

        # switch to docking once we are near the pre-dock point
        if self.mode == "NAVIGATE" and np.hypot(*(PRE_DOCK - est[:2])) < 0.25:
            self.mode = "DOCK"
            print("  -> switching to DOCK mode (camera)")

        if self.mode == "NAVIGATE":
            goal_dir = wrap(np.arctan2(PRE_DOCK[1] - y, PRE_DOCK[0] - x) - th)   # robot frame
            candidates = np.r_[goal_dir, LIDAR_ANGLES]
            free = [c for c in candidates if direction_is_free(scan_xy, c)]
            if not free:                                         # boxed in: turn on the spot
                return 0.0, 0.6
            # closest free direction to the goal, with a small preference for the previous choice
            cost = [abs(wrap(c - goal_dir)) + 0.3 * abs(wrap(c - self.prev_dir)) for c in free]
            heading = free[int(np.argmin(cost))]
            self.prev_dir = heading
            w = np.clip(1.8 * heading, -W_MAX, W_MAX)
            v = V_MAX * max(0.0, np.cos(heading)) ** 2 * np.clip((front - 0.2) / 0.4, 0, 1)
            return v, w

        # DOCK mode: state feedback on the EKF estimate (range & bearing to the dock
        # computed from the estimated pose, not from the raw camera reading)
        (rng_, b), _ = EKF.predicted_measurement(None, est)
        if abs(b) > np.radians(25):
            v, w = 0.0, np.clip(1.5 * b, -0.6, 0.6)             # first turn to face the dock
        else:
            v = np.clip(0.5 * (rng_ - DOCK_DIST), -0.1, 0.12)    # drive until DOCK_DIST away
            w = np.clip(1.5 * b, -1.0, 1.0)                     # keep facing the marker
        if front < 0.15 and v > 0:
            v = 0.0                                             # safety stop
        return v, w
# TODO (control, next): replace the DOCK law with LQR, so the robot also lines up
#   straight with the dock (this D3-style law cannot remove a sideways offset).


# ---------------------------------------------------------------------------
# 5. SIMULATION helpers
# ---------------------------------------------------------------------------
_wheel_target = np.zeros(2)
MAX_WHEEL_ACC = 20.0   # rad/s^2, motor driver ramps speed like a real robot


def set_wheel_speeds(v, w):
    """Unicycle command (v m/s, w rad/s) -> wheel speeds (rad/s)."""
    if not (np.isfinite(v) and np.isfinite(w)):
        v, w = 0.0, 0.0
    _wheel_target[:] = [(v - w * WHEEL_SEP / 2) / WHEEL_R, (v + w * WHEEL_SEP / 2) / WHEEL_R]


def step_physics():
    dmax = MAX_WHEEL_ACC * model.opt.timestep
    for _ in range(N_SUB):
        data.ctrl[:] += np.clip(_wheel_target - data.ctrl, -dmax, dmax)
        mujoco.mj_step(model, data)


def true_pose():
    """Ground truth - ONLY for logging and plots."""
    R = data.body("base").xmat.reshape(3, 3)
    return np.array([*data.body("base").xpos[:2], np.arctan2(R[1, 0], R[0, 0])])


_ROBOT_BODIES = {model.body(n).id for n in ["base", "wheel_l", "wheel_r"]}
_FLOOR = model.geom("floor").id


def robot_touching_something():
    """Ground truth check for evaluation only: is the robot touching a wall or obstacle?"""
    for c in data.contact[:data.ncon]:
        g1, g2 = c.geom1, c.geom2
        if _FLOOR in (g1, g2):
            continue
        if (model.geom_bodyid[g1] in _ROBOT_BODIES) != (model.geom_bodyid[g2] in _ROBOT_BODIES):
            return True
    return False


def set_top_view(cam):
    """Look straight down on the room: +x to the right, +y up the screen."""
    cam.lookat[:] = [0.0, 0.0, 0.0]
    cam.distance = 7.5
    cam.elevation = -90
    cam.azimuth = 90


def show_camera(img, meas):
    """Camera window with the detected marker outlined (press nothing; it updates live)."""
    frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if meas is not None:
        r, b, c = meas
        cv2.polylines(frame, [c.astype(int)], True, (0, 255, 0), 2)
        txt = f"marker: range {r:.2f} m, bearing {np.degrees(b):+.1f} deg"
    else:
        txt = "marker: not visible"
    cv2.putText(frame, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    try:
        cv2.imshow("robot camera", frame)
        cv2.waitKey(1)
    except cv2.error:
        pass   # OpenCV built without window support (e.g. the "headless" package): skip the camera window


def cov_ellipse(P2, n_sigma=2.0, npts=40):
    vals, vecs = np.linalg.eigh(P2)
    t = np.linspace(0, 2 * np.pi, npts)
    return (vecs @ np.diag(n_sigma * np.sqrt(np.maximum(vals, 0))) @ np.vstack([np.cos(t), np.sin(t)])).T


class LiveView:
    """Live top-down plot: map, LiDAR scan, EKF estimate + 2-sigma ellipse,
       odometry-only estimate, and the true pose (display only)."""

    def __init__(self):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(6.5, 6.5))
        ax = self.ax
        ax.plot([-2.5, 2.5, 2.5, -2.5, -2.5], [-2.5, -2.5, 2.5, 2.5, -2.5], "k-", lw=2)
        ax.plot(*MARKER_XY, "gs", ms=10, label="ArUco dock")
        for o in OBSTACLES:
            ax.add_patch(plt.Circle(o["pos"], o["size"] * (1.42 if o["kind"] == "box" else 1),
                                    color="0.8", zorder=0))
        self.scan, = ax.plot([], [], "r.", ms=4, label="LiDAR hits")
        self.truth, = ax.plot([], [], "o", color="0.4", ms=7, label="true pose (display only)")
        self.odo, = ax.plot([], [], "x", color="C1", ms=8, label="odometry only")
        self.est, = ax.plot([], [], "o", color="C0", ms=7, label="EKF estimate")
        self.head, = ax.plot([], [], "-", color="C0", lw=2)
        self.ell, = ax.plot([], [], "-", color="C0", lw=1, label="EKF 2-sigma")
        self.ray, = ax.plot([], [], "g--", lw=1)
        self.path, = ax.plot([], [], "-", color="C0", lw=0.8, alpha=.5)
        ax.set_xlim(-2.6, 2.6); ax.set_ylim(-2.6, 2.6); ax.set_aspect("equal"); ax.grid(alpha=.3)
        ax.legend(loc="lower right", fontsize=7)
        self.trail = []

    def update(self, t, ekf, odo, truth, scan, meas, mode):
        x, y, th = ekf.x
        r = np.where(scan > 0, scan, np.nan)
        a = LIDAR_ANGLES + th
        self.scan.set_data(x + r * np.cos(a), y + r * np.sin(a))   # scan drawn at the ESTIMATED pose
        self.truth.set_data([truth[0]], [truth[1]])
        self.odo.set_data([odo[0]], [odo[1]])
        self.est.set_data([x], [y])
        self.head.set_data([x, x + 0.25 * np.cos(th)], [y, y + 0.25 * np.sin(th)])
        e = cov_ellipse(ekf.P[:2, :2]) + [x, y]
        self.ell.set_data(e[:, 0], e[:, 1])
        if meas is not None:
            self.ray.set_data([x, MARKER_XY[0]], [y, MARKER_XY[1]])
        else:
            self.ray.set_data([], [])
        self.trail.append((x, y))
        self.path.set_data(*np.array(self.trail).T)
        sd = np.sqrt(np.diag(ekf.P))
        self.ax.set_title(f"t={t:4.1f}s  mode={mode}  marker {'SEEN' if meas else 'not seen'}\n"
                          f"EKF sigma: x {sd[0]*100:.1f} cm, y {sd[1]*100:.1f} cm, th {np.degrees(sd[2]):.1f} deg",
                          fontsize=9)
        self.fig.canvas.draw_idle()
        plt.pause(0.001)


def draw_estimate_in_viewer(viewer, est):
    """Draw a see-through green box at the EKF estimate inside the MuJoCo window."""
    try:
        x, y, th = est
        c, s_ = np.cos(th), np.sin(th)
        mat = np.array([c, -s_, 0, s_, c, 0, 0, 0, 1], float)
        with viewer.lock():
            mujoco.mjv_initGeom(viewer.user_scn.geoms[0], mujoco.mjtGeom.mjGEOM_BOX,
                                np.array([.09, .09, .045]), np.array([x - .03 * c, y - .03 * s_, .06]),
                                mat, np.array([0.1, 0.9, 0.2, 0.35], np.float32))
            viewer.user_scn.ngeom = 1
    except Exception:
        pass


# ---------------------------------------------------------------------------
# 6. MAIN LOOP
# ---------------------------------------------------------------------------
def run(T=90.0, live=True):
    mujoco.mj_resetData(model, data)
    set_wheel_speeds(0, 0)
    data.ctrl[:] = 0
    for _ in range(250):                       # let the robot settle onto its wheels
        mujoco.mj_step(model, data)

    viewer = None
    if live:
        viewer = mujoco.viewer.launch_passive(model, data)
        set_top_view(viewer.cam)
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False   # hide the yellow LiDAR rays

    ctrl = Controller()
    # The filter only knows the NOMINAL start pose; the robot really starts at TRUE_START.
    ekf = EKF(START_POSE, np.diag([0.15 ** 2, 0.15 ** 2, np.radians(10) ** 2]))
    odo = START_POSE.copy()                    # odometry-only estimate, kept for comparison
    est = ekf.x
    enc_prev = read_encoders()
    live_plot = LiveView() if live else None
    log = {k: [] for k in ["t", "est", "odo", "P", "true", "range", "bearing", "v", "w",
                           "lidar_min", "collision", "updated"]}
    docked_for = 0.0

    for k in range(int(T / DT_CTRL)):
        t0 = time.time()
        # ---- SENSE ----
        enc = read_encoders()
        d_l, d_r = enc[0] - enc_prev[0], enc[1] - enc_prev[1]
        enc_prev = enc
        scan = read_lidar()
        img = read_camera()
        meas = detect_marker(img)

        # ---- ESTIMATE (EKF) ----
        ekf.predict(d_l, d_r)
        updated = ekf.update(meas[:2]) if meas is not None else False
        est = ekf.x.copy()
        odo = odometry_update(odo, d_l, d_r)

        # ---- CONTROL ----
        v, w = ctrl(est, meas, scan)
        set_wheel_speeds(v, w)

        # ---- LOG (ground truth only here) ----
        valid = scan[scan > 0]
        log["t"].append(k * DT_CTRL); log["est"].append(est.copy()); log["true"].append(true_pose())
        log["odo"].append(odo.copy()); log["P"].append(ekf.P.copy()); log["updated"].append(updated)
        log["range"].append(meas[0] if meas else np.nan); log["bearing"].append(meas[1] if meas else np.nan)
        log["v"].append(v); log["w"].append(w)
        log["lidar_min"].append(valid.min() if valid.size else np.nan)
        log["collision"].append(robot_touching_something())

        if k % 20 == 0:                        # print sensor readouts once per second
            mtxt = f"range {meas[0]:.2f} m, bearing {np.degrees(meas[1]):+5.1f} deg" if meas else "not seen"
            print(f"t={k*DT_CTRL:5.1f}s | encoders L={enc[0]:7.2f} R={enc[1]:7.2f} rad | "
                  f"LiDAR closest {log['lidar_min'][-1]:.2f} m | marker {mtxt} | "
                  f"cmd v={v:.2f} w={w:+.2f} | {ctrl.mode} | EKF x={est[0]:+.2f} y={est[1]:+.2f}")

        # ---- ACT ----
        step_physics()
        if viewer is not None:
            if not viewer.is_running():
                break
            draw_estimate_in_viewer(viewer, est)
            viewer.sync()
            show_camera(img, meas)
            if k % 4 == 0:
                live_plot.update(k * DT_CTRL, ekf, odo, log["true"][-1], scan, meas, ctrl.mode)
            time.sleep(max(0.0, DT_CTRL - (time.time() - t0)))   # run in real time

        # ---- DONE? docked = at the right distance and facing the marker for 1 s ----
        if ctrl.mode == "DOCK" and meas and abs(meas[0] - DOCK_DIST) < 0.03 and abs(meas[1]) < 0.05:
            docked_for += DT_CTRL
            if docked_for >= 1.0:
                print(f"\nDOCKED at t={k*DT_CTRL:.1f}s")
                break
        else:
            docked_for = 0.0

    print(f"EKF: {ekf.n_updates} camera updates used, {ekf.n_rejected} rejected as outliers")
    if live_plot is not None:
        plt.ioff(); plt.close(live_plot.fig)
    if viewer is not None:
        viewer.close()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
    return {k: np.array(v) for k, v in log.items()}


def plot_results(log):
    err = np.linalg.norm(log["est"][:, :2] - log["true"][:, :2], axis=1)
    err_odo = np.linalg.norm(log["odo"][:, :2] - log["true"][:, :2], axis=1)
    e = log["est"] - log["true"]; e[:, 2] = wrap(e[:, 2])
    sd = np.sqrt(np.array([np.diag(P) for P in log["P"]]))
    inside = np.mean(np.all(np.abs(e) <= 3 * sd, axis=1)) * 100
    print(f"\nstart: robot really at x={TRUE_START[0]:.2f} y={TRUE_START[1]:.2f} "
          f"th={np.degrees(TRUE_START[2]):.1f} deg; EKF told x={START_POSE[0]:.2f} y={START_POSE[1]:.2f} "
          f"th={np.degrees(START_POSE[2]):.1f} deg")
    print(f"final true pose : x={log['true'][-1,0]:.3f}  y={log['true'][-1,1]:.3f}  "
          f"theta={np.degrees(log['true'][-1,2]):.1f} deg")
    print(f"final EKF pose  : x={log['est'][-1,0]:.3f}  y={log['est'][-1,1]:.3f}  "
          f"theta={np.degrees(log['est'][-1,2]):.1f} deg")
    print(f"position RMSE   : EKF {np.sqrt(np.mean(err**2)):.3f} m   vs odometry only {np.sqrt(np.mean(err_odo**2)):.3f} m")
    print(f"final error     : EKF {err[-1]:.3f} m   vs odometry only {err_odo[-1]:.3f} m")
    print(f"EKF consistency : error inside 3-sigma bounds {inside:.0f}% of the time")
    print(f"marker seen in {np.isfinite(log['range']).mean()*100:.0f}% of steps")
    print(f"collisions: {int(log['collision'].sum())} control steps touching a wall/obstacle")

    t = log["t"]
    fig, ax = plt.subplots(2, 3, figsize=(17, 9))
    a = ax[0, 0]
    a.plot(*log["true"][:, :2].T, "k-", label="ground truth")
    a.plot(*log["est"][:, :2].T, "C0--", label="EKF estimate")
    a.plot(*log["odo"][:, :2].T, "C1:", label="odometry only")
    a.plot(*MARKER_XY, "gs", ms=10, label="marker")
    a.plot(*PRE_DOCK, "g+", ms=12, label="pre-dock point")
    for o in OBSTACLES:
        if o["kind"] == "cylinder":
            a.add_patch(plt.Circle(o["pos"], o["size"], color="C3", alpha=.5))
        else:
            sq = plt.Rectangle(o["pos"] - o["size"], 2 * o["size"], 2 * o["size"], color="C1", alpha=.5)
            sq.set_transform(plt.matplotlib.transforms.Affine2D().rotate_deg_around(*o["pos"], o["yaw"])
                             + a.transData)
            a.add_patch(sq)
    a.set_xlim(-2.5, 2.5); a.set_ylim(-2.5, 2.5); a.set_aspect("equal"); a.grid()
    a.legend(fontsize=7); a.set_title(f"path (top view), layout seed {SEED}")

    a = ax[0, 1]
    a.plot(t, err, "C0", label="EKF"); a.plot(t, err_odo, "C1:", label="odometry only")
    a.set_title("position error |estimate - truth| (m)"); a.set_xlabel("time (s)"); a.grid(); a.legend()

    a = ax[0, 2]
    a.plot(t, log["range"], label="range to marker (m)")
    a.plot(t, log["bearing"], label="bearing (rad)")
    a.plot(t, log["lidar_min"], label="LiDAR closest (m)", alpha=.6)
    a.axhline(DOCK_DIST, color="k", ls=":")
    a.legend(fontsize=8); a.set_xlabel("time (s)"); a.grid(); a.set_title("sensors")

    for i, (name, scale) in enumerate([("x error (m)", 1), ("y error (m)", 1), ("theta error (deg)", np.degrees(1))]):
        a = ax[1, i]
        a.plot(t, e[:, i] * scale, "C0", label="EKF error")
        a.fill_between(t, -3 * sd[:, i] * scale, 3 * sd[:, i] * scale, color="C0", alpha=.15, label="EKF +-3 sigma")
        a.set_title(name); a.set_xlabel("time (s)"); a.grid(); a.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(HERE / "trackA_result.png", dpi=90)
    print(f"plot saved to {HERE / 'trackA_result.png'}")
    plt.show()


if __name__ == "__main__":
    print(f"Obstacle layout seed: {SEED}   (rerun this layout with:  python trackA_demo.py --seed {SEED})")
    for i, o in enumerate(OBSTACLES):
        print(f"  obstacle {i}: {o['kind']:8s} at ({o['pos'][0]:+.2f}, {o['pos'][1]:+.2f}), size {o['size']:.2f} m")
    plot_results(run(live=not ARGS.headless))
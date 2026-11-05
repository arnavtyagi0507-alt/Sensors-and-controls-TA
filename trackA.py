import argparse
import time
 
import numpy as np
import cv2
import mujoco
import mujoco.viewer
import scipy.linalg
import matplotlib.pyplot as plt
import sympy as sp
import control as ct
 
# --- settings ---
WHEEL_R, WHEEL_SEP = 0.033, 0.160     #turtle bit wheel radius and seperation (m)
MARKER_XY = np.array([2.44, 0.0])          # ArUco marker position (m)
MARKER_SIZE = 0.20                         
DOCK_DIST = 0.40                           # distance stopped from marker (m)
PRE_DOCK = np.array([1.54, 0.0])           
START_POSE = np.array([-1.5, -0.5, np.radians(30)])
T = 0.05                                   #EKF sample time (s)
N_LIDAR = 72                               # LiDAR beams (one every 5 deg)
CAM_W, CAM_H, CAM_FOVY = 640, 480, 60.0


ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=None) # --seed makes random noise sequences repeatable
ap.add_argument("--headless", action="store_true") #headless skips the live viewers and runs sim in the background
ARGS = ap.parse_args()
SEED = ARGS.seed if ARGS.seed is not None else np.random.randint(10000)
rng = np.random.default_rng(SEED)
 
# --- Model world and robot ---

#generate n obst. positions between start and predock goal - remaining obstacles are sampled randomly while respecting gaps
def random_obstacles(n=4):
    start = START_POSE[:2]
    obs = [start + rng.uniform(0.35, 0.65) * (PRE_DOCK - start)]     # one always in the way
    while len(obs) < n:
        p = rng.uniform(-2.0, 1.6, 2)
        if (np.linalg.norm(p - start) > 0.7 and np.linalg.norm(p - PRE_DOCK) > 0.7
                and all(np.linalg.norm(p - o) > 0.7 for o in obs)):
            obs.append(p)
    return obs
 

 #gen env. once per run 
OBSTACLES = random_obstacles()
TRUE_START = START_POSE + [*rng.uniform(-0.15, 0.15, 2), np.radians(rng.uniform(-10, 10))]
 
 # create aruco marker for the wall
def marker_png():
    tag = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), 0, 400)
    img = cv2.flip(cv2.copyMakeBorder(tag, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=255), 1)
    return cv2.imencode(".png", img)[1].tobytes()
 
 # creates one mujoco site per LIDar beam. N_LIDAR sets the angular resolution
lidar_sites = "".join(f'<site name="l{k}" pos="0 0 .09" zaxis="{np.cos(2*np.pi*k/N_LIDAR):.4f} '
                      f'{np.sin(2*np.pi*k/N_LIDAR):.4f} 0"/>' for k in range(N_LIDAR))
lidar_sensors = "".join(f'<rangefinder name="r{k}" site="l{k}" cutoff="3.5"/>' for k in range(N_LIDAR))
obstacle_geoms = "".join(f'<geom type="cylinder" pos="{x:.2f} {y:.2f} .2" size=".18 .2" rgba="0 0 .5 1"/>'
                         for x, y in OBSTACLES)
wheel = lambda s, y: (f'<body name="w{s}" pos="0 {y} 0"><joint name="w{s}" type="hinge" axis="0 1 0" damping=".001"/>'
                      f'<geom type="sphere" size="{WHEEL_R}" rgba="0 0 0 0" friction="1.5 .005 .0001" mass=".05"/></body>')

 #mujoco model description in XML
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

 #compile XML into a mujoco model 
model = mujoco.MjModel.from_xml_string(XML, assets={"marker.png": marker_png()})
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, CAM_H, CAM_W)
hide_rays = mujoco.MjvOption()
hide_rays.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False
 
# --- SENSING: LiDAR and camera ---
#lidar gives obstacle distances - camera detects ARuCo tag and estimates range and bearing
LIDAR_ANGLES = np.arctan2(np.sin(2 * np.pi * np.arange(N_LIDAR) / N_LIDAR),
                          np.cos(2 * np.pi * np.arange(N_LIDAR) / N_LIDAR))
f_px = (CAM_H / 2) / np.tan(np.radians(CAM_FOVY) / 2)
K_CAM = np.array([[f_px, 0, CAM_W / 2], [0, f_px, CAM_H / 2], [0, 0, 1]])
detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
corners_3d = MARKER_SIZE / 2 * np.array([[-1, 1, 0], [1, 1, 0], [1, -1, 0], [-1, -1, 0]], np.float32)
 
 # read wheel joint angles (radians) and add encoder noise (0.002 rad stddev)
def read_encoders():
    return np.array([data.sensor("el").data[0], data.sensor("er").data[0]]) + rng.normal(0, 0.002, 2)
 
 
def read_lidar():
    r = np.array([data.sensor(f"r{k}").data[0] for k in range(N_LIDAR)])
    return np.where(r > 0, r + rng.normal(0, 0.01, N_LIDAR), 3.5)
 
 
def read_marker():
    renderer.update_scene(data, camera="cam", scene_option=hide_rays)
    img = renderer.render()
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    result = None
    if ids is not None:
        ok, _, t = cv2.solvePnP(corners_3d, corners[0].reshape(4, 2), K_CAM, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if ok and np.all(np.isfinite(t)):
            r, b = np.hypot(t[0, 0], t[2, 0]), np.arctan2(-t[0, 0], t[2, 0])
            if r < 3.0:                                   # ignore far readings (marker too small)
                result = np.array([r, b])
    if not ARGS.headless:                                 # show the robot camera
        frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if ids is not None:
            cv2.polylines(frame, [corners[0].reshape(4, 2).astype(int)], True, (0, 255, 0), 2)
        text = f"range {result[0]:.2f} m, bearing {np.degrees(result[1]):+.1f} deg" if result is not None else "marker not seen"
        cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow("robot camera", frame)
        cv2.waitKey(1)
    return result
 
 
# --- estimation: EKF for unicycle + range/bearing to marker ---
def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi
 
 
def f_uni(x, u):
    v, w = u
    th = x[2] + w * T / 2
    return np.array([x[0] + v * T * np.cos(th), x[1] + v * T * np.sin(th), wrap(x[2] + w * T)])
 
 
def F_uni(x, u):
    v, w = u
    th = x[2] + w * T / 2
    return np.array([[1, 0, -v * T * np.sin(th)], [0, 1, v * T * np.cos(th)], [0, 0, 1]])
 
 
CAM_OFFSET = 0.07                       # camera is 7 cm in front of the robot centre
MARKER_FACE = MARKER_XY - [0.005, 0]
 
 
def h_rb(x):
    dx = MARKER_FACE[0] - (x[0] + CAM_OFFSET * np.cos(x[2]))
    dy = MARKER_FACE[1] - (x[1] + CAM_OFFSET * np.sin(x[2]))
    return np.array([np.hypot(dx, dy), wrap(np.arctan2(dy, dx) - x[2])])
 
 
def H_rb(x):
    return np.array([(h_rb(x + d) - h_rb(x - d)) / 2e-6 for d in 1e-6 * np.eye(3)]).T
 
 
process_noise = np.diag([0.001 ** 2, 0.001 ** 2, np.radians(0.3) ** 2])   # wheel slip per sample
meas_noise = np.diag([0.05 ** 2, np.radians(0.5) ** 2])                # camera: 5 cm range, 0.5 deg bearing
 
 
def ekf_step(xh, P, u, z):
    # predict with the wheel encoders
    F = F_uni(xh, u)
    xh = f_uni(xh, u)
    P = F @ P @ F.T + process_noise
    # correct with the camera (only when the marker is seen)
    if z is not None:
        H = H_rb(xh)
        nu = z - h_rb(xh)
        nu[1] = wrap(nu[1])
        S = H @ P @ H.T + meas_noise
        K = P @ H.T @ np.linalg.inv(S)
        xh = xh + K @ nu
        xh[2] = wrap(xh[2])
        P = (np.eye(3) - K @ H) @ P
    return xh, P
 
 
# --- Control ---
def lqr_gain(V):
    Ad = np.array([[1, V * T], [0, 1]])                  # error model e = [e_y, e_th]
    Bd = np.array([[V * T ** 2 / 2], [T]])
    Q = np.diag([1 / 0.05 ** 2, 1 / np.radians(10) ** 2])   # Bryson's rule: 5 cm, 10 deg
    R = np.array([[1.0]])                                    # 1 rad/s steering
    S = scipy.linalg.solve_discrete_are(Ad, Bd, Q, R)
    return np.linalg.solve(R + Bd.T @ S @ Bd, Bd.T @ S @ Ad).ravel()
 
 
K_LQR = lqr_gain(0.15)
 
 
def is_free(scan, angle, width=0.23, length=0.7):
    px, py = scan * np.cos(LIDAR_ANGLES), scan * np.sin(LIDAR_ANGLES)
    along = px * np.cos(angle) + py * np.sin(angle)
    across = np.abs(px * np.sin(angle) - py * np.cos(angle))
    return not np.any((along > 0) & (along < length) & (across < width))
 
 
def controller(xh, scan, mode):
    x, y, th = xh
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
 
 
# --- Main loop ---
def set_wheels(v, w):
    data.ctrl[:] = [(v - w * WHEEL_SEP / 2) / WHEEL_R, (v + w * WHEEL_SEP / 2) / WHEEL_R]
 
 
def true_pose():
    R = data.body("robot").xmat.reshape(3, 3)
    return np.array([*data.body("robot").xpos[:2], np.arctan2(R[1, 0], R[0, 0])])
 
 
def run(live=True):
    for _ in range(250):                                   # let the robot settle on its wheels
        mujoco.mj_step(model, data)
    xh, P = START_POSE.copy(), np.diag([0.1 ** 2, 0.1 ** 2, np.radians(7) ** 2])
    enc_prev, mode = read_encoders(), "NAVIGATE"
    log_est, log_true = [], []
 
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
        ax.set_xlim(-2.6, 2.6); ax.set_ylim(-2.6, 2.6); ax.set_aspect("equal"); ax.legend(loc="lower right")
 
    for k in range(1800):                                  # up to 90 s
        t0 = time.time()
        # sense
        enc = read_encoders()
        d_l, d_r = (enc - enc_prev) * WHEEL_R
        enc_prev = enc
        u = np.array([(d_l + d_r) / 2 / T, (d_r - d_l) / WHEEL_SEP / T])   # measured v, w
        scan = read_lidar()
        z = read_marker()
        # estimate
        xh, P = ekf_step(xh, P, u, z)
        # control
        v, w, mode = controller(xh, scan, mode)
        set_wheels(v, w)
        for _ in range(int(T / model.opt.timestep)):
            mujoco.mj_step(model, data)
        log_est.append(xh.copy()); log_true.append(true_pose())
 
        if k % 20 == 0:
            print(f"t={k*T:4.1f}s  {mode:8s}  EKF x={xh[0]:+.2f} y={xh[1]:+.2f}  "
                  f"marker={'seen' if z is not None else '-':4s}  closest obstacle {scan.min():.2f} m")
        if live:
            if not viewer.is_running():
                break
            viewer.sync()
            if k % 5 == 0:
                dots.set_data(xh[0] + scan * np.cos(LIDAR_ANGLES + xh[2]), xh[1] + scan * np.sin(LIDAR_ANGLES + xh[2]))
                est_dot.set_data([xh[0]], [xh[1]]); true_dot.set_data([log_true[-1][0]], [log_true[-1][1]])
                ax.set_title(f"{mode}   t = {k*T:.1f} s"); plt.pause(0.001)
            time.sleep(max(0, T - (time.time() - t0)))
        if mode == "DOCK" and z is not None and abs(z[0] - DOCK_DIST) < 0.03 and abs(z[1]) < 0.05:
            print(f"DOCKED at t={k*T:.1f}s")
            break
 
    if live:
        viewer.close(); plt.ioff(); plt.close(fig)
    return np.array(log_est), np.array(log_true)
 
 
if __name__ == "__main__":
    print(f"seed {SEED}  (repeat with --seed {SEED})")
    est, true = run(live=not ARGS.headless)
    err = np.linalg.norm(est[:, :2] - true[:, :2], axis=1)
    print(f"final position error {err[-1]*100:.1f} cm, mean {err.mean()*100:.1f} cm; "
          f"dock offset {true[-1, 1]*100:+.1f} cm")
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(true[:, 0], true[:, 1], "k-", label="true path")
    ax[0].plot(est[:, 0], est[:, 1], "b--", label="EKF estimate")
    for o in OBSTACLES:
        ax[0].add_patch(plt.Circle(o, 0.18, color="r", alpha=.5))
    ax[0].plot(*MARKER_XY, "gs", ms=10, label="marker")
    ax[0].set_xlim(-2.5, 2.5); ax[0].set_ylim(-2.5, 2.5); ax[0].set_aspect("equal"); ax[0].legend(); ax[0].grid()
    ax[1].plot(np.arange(len(err)) * T, err * 100)
    ax[1].set_xlabel("time (s)"); ax[1].set_ylabel("EKF position error (cm)"); ax[1].grid()
    plt.tight_layout(); plt.show()


T = 0.05
WHEEL_R = 0.033
WHEEL_SEP = 0.16
CAM_OFFSET = 0.07
MARKER = np.array([2.435,0.0])

def wrap(a): # keeping the angles between -pi and pi 
    return(a + np.pi) % (2 * np.pi) - np.pi


def encoders_to_u(dphi_l, dphi_r, r=WHEEL_R, b = WHEEL_SEP, dt = T): #Forward speed is the average of the two wheels; turn rate is their
    d_l = r * dphi_l
    d_r = r * dphi_r
    # difference divided by the wheel separation.
    return np.array([(d_r + d_l) / 2 / dt, (d_r - d_l) / b / dt]) 

def f_uni(xh, u):
    v, w = u
    x, y, th = xh
    if abs(w) < 1e-6:
        return np.array([x + v * T * np.cos(th), y + v * T * np.sin(th), wrap(th)])
    th_n = th + w * T
    return np.array([x + v / w * (np.sin(th_n) - np.sin(th)), y - v / w * (np.cos(th_n) - np.cos(th)), wrap(th_n)])

def h_rb(xh, mk = MARKER):
    cx = xh[0] + CAM_OFFSET * np.cos(xh[2])
    cy = xh[1] + CAM_OFFSET * np.sin(xh[2])
    dx,dy = mk[0] - cx, mk[1] - cy
    return np.array([np.sqrt(dx**2 + dy**2), wrap(np.arctan2(dy, dx) - xh[2])])

def F_uni(xh, u):
    v, w = u
    th = xh[2]
    F = np.eye(3)
    if abs(w) < 1e-6:
        F[0,2] = -v * T * np.sin(th)
        F[1,2] = v * T * np.cos(th)
    else:
        th_n = th + w * T
        F[0,2] = v / w * (np.cos(th_n) - np.cos(th))
        F[1,2] = v / w * (np.sin(th_n) - np.sin(th))
    return F




def h_range(xh, mk = MARKER):
    return h_rb(xh, mk)[:1]

def num_jac(fn, x, eps = 1e-6):
    x = np.asarray(x,float)
    f0 = np.atleast_1d(fn(x))
    J = np.zeros((len(f0), len(x)))
    for i in range(len(x)):
        dx = np.zeros_like(x); dx[i] = eps
        J[:,i] = (np.atleast_1d(fn(x + dx)) - np.atleast_1d(fn(x-dx)))/(2*eps)
    return J

def H_rb(xh, mk = MARKER):
    return num_jac(lambda s: h_rb(s, mk), xh)

def H_range(xh, mk = MARKER):
    return num_jac(lambda s: h_range(s, mk), xh)

#linearised unicycle about driving straight along x (theta = 0) at speed v0
def unicycle_lin(v0):
    Au = np.array([[0.0,0,0],[0,0,v0],[0,0,0]])
    Bu = np.array([[1.0,0],[0,0],[0,1.0]]) # sideways error while the robot is moving.
    return Au, Bu

def pbh_stabilisable(A,B, tol = 1e-8):
    n = A.shape[0]
    return all(np.linalg.matrix_rank(np.hstack([A - lam * np.eye(n), B]), tol) == n
               for lam in np.linalg.eigvals(A) if lam.real >= -tol)


#letting sympy take the Jacobians of the continuous model
def linearise_sympy():
    x,y,th,v,w,v0 = sp.symbols("x y theta v w v0", real = True) 
    X, U = sp.Matrix([x,y,th]), sp.Matrix([v,w]) # evaluating them at straight driving. Should match unicycle_lin.
    f = sp.Matrix([v*sp.cos(th), v*sp.sin(th), w])
    op = {th: 0, v: v0, w: 0}
    return f.jacobian(X).subs(op), f.jacobian(U).subs(op)

def pbh_detectable(A,C, tol = 1e-8):
    n = A.shape[0]
    return all(np.linalg.matrix_rank(np.vstack([A - lam * np.eye(n), C]), tol) == n
               for lam in np.linalg.eigvals(A) if lam.real >= -tol)



def tracking_error_ss(v0):
    return np.array([[0.0, v0], [0.0, 0.0]]), np.array([[0.0], [1.0]])





def main():
    np.set_printoptions(precision=3, suppress=True)

    xt, ut = np.array([-0.3, 1.5, 0.5]), np.array([0.13, -0.2])
    print("Largest difference between analytic and numerical F: =",
          np.abs(F_uni(xt, ut) - num_jac(lambda s: f_uni(s, ut), xt)).max())

    A_lin, B_lin = linearise_sympy()
    print("A = ", A_lin.tolist(), "\nB = ", B_lin.tolist())

    print("  speed   rank (continuous)   rank (discrete)")

    for v0 in [0.15,0.17,0.08,0.03,0.0]:
        A, B = tracking_error_ss(v0)
        sysd = ct.c2d(ct.ss(A, B, np.eye(2), np.zeros((2,1))), T, method = "zoh")
        print(f"  {v0:4.2f}   {np.linalg.matrix_rank(ct.ctrb(A, B))}                   "
              f"{np.linalg.matrix_rank(ct.ctrb(sysd.A, sysd.B))}")

    sysd = ct.c2d(ct.ss(*tracking_error_ss(0.15), np.eye(2), np.zeros((2,1))),T,method = 'zoh')
    print("Discrete model at 0.15 m/s, the one the LQR is designed on:")
    print("Ad =\n", sysd.A, "\nBd =\n", sysd.B)

    A_full, B_full = unicycle_lin(0.15)

    operating_point = np.array([0.20,0.04,0.0])

    for name, Hf in (( "range only", H_range), ("range + bearing", H_rb)):
        C = Hf(operating_point)
        for v0 in (0.15,0.0):
            A, _ = unicycle_lin(v0)
            O = ct.obsv(A,C)
            rk = np.linalg.matrix_rank(O, 1e-9)
            line = (f"  {name}  v0 = {v0:4.2f}: rank O = {rk} of 3, "
                f"detectable (PBH) = {pbh_detectable(A, C)}")

            if rk < 3:
                Vt = np.linalg.svd(O)[2]
                line +=  "   hidden direction(s) (SVD): " + ", ".join(str(v.round(3)) for v in Vt[rk:])
            print(line)



    fig, ax = plt.subplots(1,2, figsize = (10,4))
    speeds = np.linspace(0,0.2,101)
    ax[0].plot(speeds, [np.abs(np.linalg.det(ct.ctrb(*tracking_error_ss(V)))) for V in speeds], color="navy")
    ax[0].axvline(0.15, ls="--", c="navy", label="navigation speed")
    ax[0].set_xlabel("v0 (m/s)"); ax[0].set_ylabel("|det [B, AB]|")
    ax[0].set_title("Controllability of [e_y, e_theta]"); ax[0].legend(); ax[0].grid(alpha=.3)

    for name, Hf, style in (("range only", H_range, "--"), ("range + bearing", H_rb, "-")):
        C = Hf(operating_point)
        ax[1].plot(speeds, [np.linalg.svd(ct.obsv(unicycle_lin(V)[0], C), compute_uv=False)[-1]
                        for V in speeds], label=name, color="navy", ls=style)
    ax[1].set_xlabel("v0 (m/s)"); ax[1].set_ylabel("smallest singular value of O")
    ax[1].set_title("Observability vs speed"); ax[1].legend(); ax[1].grid(alpha=.3)

    fig.tight_layout()
    fig.savefig("modelling_results.png", dpi=130)


if __name__ == "__main__":
    main()





WHEEL_R_CAL = 0.03248        # m, calibrated wheel radius 
RANGE_SCALE = 0.9868         # true range / solvePnP range 
BEARING_BIAS = np.deg2rad(0.062)   # constant solvePnP bearing bias, subtracted 


QD = np.diag([0.000125 ** 2, 0.000077 ** 2, np.deg2rad(0.0843) ** 2])
RN = np.diag([0.0136 ** 2, np.deg2rad(0.041) ** 2])

P_start = np.diag([0.1 ** 2, 0.1 ** 2, np.deg2rad(7) ** 2])  



def ekf_correction(xh_pred, P_pred, y, h, H_jac, Rn, residual=None):
    y_hat = h(xh_pred)                       # nonlinear h gives the expected measurement
    H = H_jac(xh_pred)
    S = H @ P_pred @ H.T + Rn                
    K = P_pred @ H.T @ np.linalg.inv(S)      # Kalman gain
    nu = (y - y_hat) if residual is None else residual(y, y_hat)
    xh = xh_pred + K @ nu
    P = (np.eye(len(xh_pred)) - K @ H) @ P_pred
    return xh, P, nu, S

def ekf_step(xh, P, u, z, Qd=QD, Rn=RN, calibrated=True):
    u = np.asarray(u, float)
    if calibrated:
        u = u * WHEEL_R_CAL / WHEEL_R        
    xh, P = ekf_prediction(xh, P, u, f_uni, F_uni, Qd)
    xh[2] = wrap(xh[2])
    if z is not None:
        if calibrated:                       
            y = np.array([z[0] * RANGE_SCALE, z[1] - BEARING_BIAS])
        else:
            y = np.array([z[0], z[1]])
        xh, P, _, _ = ekf_correction(xh, P, y, h_rb, H_rb, Rn, range_bearing_residual)
        xh[2] = wrap(xh[2])
    return xh, P


def ekf_prediction(xh, P, u, f, F_jac, Qd):
    xh_pred = f(xh, u)                       # nonlinear f propagates the state
    F = F_jac(xh, u)                         
    P_pred = F @ P @ F.T + Qd
    return xh_pred, P_pred

def range_bearing_residual(y, y_hat):
    return np.array([y[0] - y_hat[0], wrap(y[1] - y_hat[1])])





SIM_START = np.array([-1.5, -0.5, np.deg2rad(30)])
PRE_DOCK = np.array([1.54, 0.0])
DOCK_X = 2.0
CAM_MAX_RANGE = 3.0             
N_STEPS = 700


def simulate(seed=1, max_range=CAM_MAX_RANGE, Qd_filter=None, Rn_filter=None, use_ekf=True):
    r = np.random.default_rng(seed)
    Qd_filter = QD if Qd_filter is None else Qd_filter
    Rn_filter = RN if Rn_filter is None else Rn_filter

    true_pose = SIM_START.copy()
    x_dr = true_pose.copy()
    xh = true_pose.copy()
    P = P_start.copy()
    TRUE, DR, XH, PP, SEEN = [true_pose.copy()], [x_dr.copy()], [xh.copy()], [P.copy()], [False]

    for k in range(N_STEPS):
        # --- controller on the estimate
        goal = PRE_DOCK if xh[0] < PRE_DOCK[0] - 0.1 else np.array([DOCK_X, 0.0])
        err = wrap(np.arctan2(goal[1] - xh[1], goal[0] - xh[0]) - xh[2])
        dist = np.hypot(*(goal - xh[:2]))
        u = (min(0.15, 0.5 * dist) * np.cos(err) ** 2, float(np.clip(2.0 * err, -1.2, 1.2)))
        if goal[0] == DOCK_X and dist < 0.03:
            u = (0.0, 0.0)                                  # docked: stop
        # --- true motion, with process noise actually applied
        true_pose = f_uni(true_pose, u) + r.multivariate_normal(np.zeros(3), QD)
        true_pose[2] = wrap(true_pose[2])
        # --- dead reckoning: same u, no corrections
        x_dr = f_uni(x_dr, u)
        # --- camera: marker within range
        z = None
        if np.hypot(*(MARKER - true_pose[:2])) <= max_range:
            z = h_rb(true_pose) + r.multivariate_normal(np.zeros(2), RN)
        if use_ekf:
            xh, P = ekf_step(xh, P, u, z, Qd_filter, Rn_filter, calibrated=False)
        else:
            xh, P = ekf_step(xh, P, u, None, Qd_filter, Rn_filter, calibrated=False)
        TRUE.append(true_pose.copy()); DR.append(x_dr.copy()); XH.append(xh.copy())
        PP.append(P.copy()); SEEN.append(z is not None)
    return {"true": np.array(TRUE), "dr": np.array(DR), "xh": np.array(XH),
            "P": np.array(PP), "seen": np.array(SEEN)}


def score(out):
    e = np.linalg.norm(out["true"][:, :2] - out["xh"][:, :2], axis=1)
    env = 2 * np.sqrt(out["P"][:, 0, 0] + out["P"][:, 1, 1])
    return np.sqrt(np.mean(e ** 2)), e[-1], np.mean(e < env)


def mean_over_seeds(num_runs=20, **kw):
    s = np.array([score(simulate(seed=sd, **kw)) for sd in range(num_runs)])
    return s.mean(axis=0)


def main(num_runs=20):

    print("=== Dead reckoning vs EKF ===")
    print(f"{'':16s} {'RMSE [m]':>9s} {'final [m]':>10s} {'2-sigma coverage':>17s}")
    dr = [simulate(seed=sd) for sd in range(num_runs)]
    e_dr = [np.linalg.norm(o["true"][:, :2] - o["dr"][:, :2], axis=1) for o in dr]
    print(f"{'dead reckoning':16s} {np.mean([np.sqrt(np.mean(e**2)) for e in e_dr]):9.3f} "
          f"{np.mean([e[-1] for e in e_dr]):10.3f} {'-':>17s}")
    print(f"{'EKF':16s} " + "{:9.3f} {:10.3f} {:17.2f}".format(*mean_over_seeds(num_runs)))

    print("\nQd scale")
    print(f"{'Qd scale':>9s} {'RMSE [m]':>9s} {'final [m]':>10s} {'2-sigma coverage':>17s}")
    for s in [0.01, 0.1, 1.0, 10.0, 100.0]:
        print(f"{s:9.2f} " + "{:9.3f} {:10.3f} {:17.2f}".format(*mean_over_seeds(num_runs, Qd_filter=s * QD)))

    print("\n Rn scale: the trust dial ")
    print(f"{'Rn scale':>9s} {'RMSE [m]':>9s} {'final [m]':>10s} {'2-sigma coverage':>17s}")
    for s in [0.1, 1.0, 10.0]:
        print(f"{s:9.2f} " + "{:9.3f} {:10.3f} {:17.2f}".format(*mean_over_seeds(num_runs, Rn_filter=s * RN)))

    print("\n Localisation degrades: shorter camera range")
    print(f"{'max range':>9s} {'RMSE [m]':>9s} {'final [m]':>10s} {'2-sigma coverage':>17s}")
    for mr in [3.0, 2.0, 1.0]:
        print(f"{mr:8.1f}m " + "{:9.3f} {:10.3f} {:17.2f}".format(*mean_over_seeds(num_runs, max_range=mr)))

    
    res = simulate(seed=1)
    t_k = np.arange(len(res["xh"])) * T
    err = np.linalg.norm(res["true"][:, :2] - res["xh"][:, :2], axis=1)
    fig, ax = plt.subplots(3, 1, figsize=(10, 6.5), sharex=True)
    ax[0].plot(t_k, err, "r", lw=1.6, label="actual position error")
    ax[0].plot(t_k, 2 * np.sqrt(res["P"][:, 0, 0] + res["P"][:, 1, 1]), "k--", lw=1.5,
               label=r"$2\sigma$ envelope")
    ax[0].set_ylabel("[m]"); ax[0].legend(fontsize=8)
    ax[0].set_title("EKF error stays inside its own uncertainty")
    ax[1].plot(t_k, np.rad2deg(np.sqrt(res["P"][:, 2, 2])), "g", lw=1.8)
    ax[1].set_ylabel(r"$\sigma_\theta$ [deg]")
    ax[2].step(t_k, res["seen"].astype(int), "b", lw=1.4, where="post")
    ax[2].set_ylabel("marker seen"); ax[2].set_xlabel("t [s]")
    plt.tight_layout(); plt.savefig("estimation_error_2sigma.png", dpi=130)

    plt.figure(figsize=(7, 5))
    plt.plot(res["true"][:, 0], res["true"][:, 1], "k", label="true")
    plt.plot(res["dr"][:, 0], res["dr"][:, 1], "r:", label="dead reckoning")
    plt.plot(res["xh"][:, 0], res["xh"][:, 1], "b--", label="EKF")
    plt.plot(*MARKER, "gs", ms=10, label="marker")
    plt.axis("equal"); plt.grid(alpha=.3); plt.legend(); plt.xlabel("x [m]"); plt.ylabel("y [m]")
    plt.title("True path, dead reckoning and EKF")
    plt.tight_layout(); plt.savefig("estimation_paths.png", dpi=130)


if __name__ == "__main__":
    main()



CAM_W = 640
CAM_H = 480
CAM_FOVY = 60
MARKER_X = 2.44
MARKER_SIZE = 0.2
ENC_NOISE = 0.002
MAX_CAMERA_RANGE = 3.0


# experiment settings I chose
STRAIGHT_START = (-2.0, -1.5, 0.0)
STRAIGHT_SPEED = 0.13
STRAIGHT_TIME = 20.0
DRIVE_START = (0.5, 1.4, -0.3)
DRIVE_SEED = 7
DRIVE_TIME = 8
TEST_DRIVES = [(0.18, 0.0), (0.10, 0.0), (0.06, 0.0), (0.12, -0.3), (0.15, -0.3),
               (0.10, 0.6), (0.10, -0.6), (0.0, 0.8), (0.0, -0.8), (0.12, 0.15)]
QD_SCALES = [1, 3, 10, 30, 100]
COVERAGE_TARGET = 0.95
N_CAMERA_POSES = 600
CAMERA_SEED = 1
CAMERA_X_RANGE = (-0.6, 2.05)
CAMERA_Y_RANGE = (-1.5, 1.5)
CAMERA_HEADING_SPREAD = 0.35
DISTANCE_BANDS = [(0.4, 1.0), (1.0, 2.0), (2.0, 3.0)]
RESULTS_FILE = "calibration_results.txt"


def _marker_png():
    tag = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), 0, 400)
    img = cv2.flip(cv2.copyMakeBorder(tag, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=255), 1)
    return cv2.imencode(".png", img)[1].tobytes()


_wheel = lambda s,y: (f'<body name="w{s}" pos="0 {y} 0"><joint name="w{s}" type="hinge" axis="0 1 0" damping=".001"/>'
                       f'<geom type="sphere" size="{WHEEL_R}" rgba="0 0 0 0" friction="1.5 .005 .0001" mass=".05"/></body>')


def build_scene():
    xml = f"""
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
    <geom type="box" pos="{MARKER_X} 0 .13" size=".005 .15 .15" material="tag" contype="0" conaffinity="0"/>
    <body name="robot" pos="0 0 .034">
      <freejoint/>
      <geom type="box" size=".07 .07 .03" pos="-.03 0 .025" rgba=".15 .15 .15 1" mass=".8" contype="0" conaffinity="0"/>
      <geom type="sphere" size=".012" pos="-.06 0 -.021" friction="0 0 0" priority="1" condim="1" mass=".01" rgba="0 0 0 0"/>
      {_wheel("l", WHEEL_SEP / 2)}{_wheel("r", -WHEEL_SEP / 2)}
      <camera name="cam" pos=".07 0 .095" xyaxes="0 -1 0 0 0 1" fovy="{CAM_FOVY}"/>
    </body>
  </worldbody>
  <actuator>
    <velocity joint="wl" kv=".5" ctrlrange="-6.6 6.6" forcerange="-.05 .05"/>
    <velocity joint="wr" kv=".5" ctrlrange="-6.6 6.6" forcerange="-.05 .05"/>
  </actuator>
  <sensor><jointpos name="el" joint="wl"/><jointpos name="er" joint="wr"/></sensor>
</mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml, assets={"marker.png": _marker_png()})
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, CAM_H, CAM_W)
    return model, data, renderer


sim_renderer = renderer
model, data, renderer = build_scene()

f_px = (CAM_H / 2) / np.tan(np.radians(CAM_FOVY) / 2)        # pinhole focal length (px)
K_CAM = np.array([[f_px, 0, CAM_W / 2], [0, f_px, CAM_H / 2], [0, 0, 1]])
detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
corners_3d = MARKER_SIZE / 2 * np.array([[-1, 1, 0], [1, 1, 0], [1, -1, 0], [-1, -1, 0]], np.float32)

#drive a straight line, compare distance from the encoders with

def calibrate_wheel_radius():
    set_robot_pose(*STRAIGHT_START)
    pose_start,e0 = get_true_pose(), encoders()

    for _ in range(int(STRAIGHT_TIME/T)):
        drive_step(STRAIGHT_SPEED,0.0)

    p1, e1 = get_true_pose(), encoders()
    true_distance = np.hypot(*(p1[:2]-pose_start[:2]))

    d_reported = WHEEL_R * (e1-e0).mean() 
    return WHEEL_R * true_distance / d_reported, true_distance, d_reported # the true distance, and scale the radius: r_hat = r * d_true / d_reported

def set_robot_pose(x,y,th,settle=350):
    mujoco.mj_resetData(model, data)
    data.qpos[:3] = [x, y, 0.034]
    data.qpos[3:7] = [np.cos(th/2), 0, 0, np.sin(th/2)]
    mujoco.mj_forward(model, data)
    for _ in range(settle):
        mujoco.mj_step(model, data)




def encoders():
    return np.array([data.sensor("el").data[0], data.sensor("er").data[0]])


def drive_step(v,w):
    data.ctrl[:] = [(v-w*WHEEL_SEP/2)/WHEEL_R, (v+w*WHEEL_SEP/2)/WHEEL_R]

    for _ in range(int(T / model.opt.timestep)):
        mujoco.mj_step(model, data)


def read_camera():
    renderer.update_scene(data, camera = 'cam')
    gray = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return None

    ok, _, t = cv2.solvePnP(corners_3d, corners[0].reshape(4, 2), K_CAM, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)

    if not ok:
        return None
    r,b = np.hypot(t[0,0], t[2,0]), np.arctan2(-t[0,0], t[2,0])

    if r >= MAX_CAMERA_RANGE:
        return None

    return np.array([r, b])


def get_true_pose():
    R = data.body("robot").xmat.reshape(3,3)
    return np.array([*data.body("robot").xpos[:2], np.arctan2(R[1,0], R[0,0])])



# covers ~95 % of the dead-reckoning error.
def estimate_process_noise(logs):
    errs = []

    for U, X in logs:
        for k, u in enumerate(U):
            e = X[k+1]-f_uni(X[k], u)
            e[2] = wrap(e[2])
            errs.append(e)

    Qd0 = np.diag(np.array(errs).var(axis=0))

    coverage_table = []
    for s in QD_SCALES:
        inside, total = 0,0
        for U, X in logs:
            xh, P = X[0].copy(), np.zeros((3,3))
            for k, u in enumerate(U):
                P = F_uni(xh, u) @ P @ F_uni(xh, u).T + s * Qd0
                xh = f_uni(xh, u)
                e = np.hypot(*(X[k+1][:2]-xh[:2]))
                inside += e < 2 * np.sqrt(P[0,0] + P[1,1])
                total += 1
        coverage_table.append((s, inside / total))
    return Qd0, coverage_table

#the mean bearing error is a bias to subtract, the spread gives Rn.
def calibrate_camera(n=N_CAMERA_POSES, seed = CAMERA_SEED):
    rng = np.random.default_rng(seed)
    Z, H = [], []
    tries = 0

    while len(Z) < n:
        tries += 1
        x,y = rng.uniform(*CAMERA_X_RANGE), rng.uniform(*CAMERA_Y_RANGE)
        th = np.arctan2(MARKER[1] - y, MARKER[0] - x) + rng.uniform(-CAMERA_HEADING_SPREAD, CAMERA_HEADING_SPREAD)
        set_robot_pose(x, y, th, settle=0)
        z = read_camera()

        if z is None:
            continue

        Z.append(z)
        H.append(h_rb(np.array([x,y,th])))

    Z,H = np.array(Z), np.array(H)
    scale = H[:,0].sum() / Z[:,0].sum()

    e_r = scale * Z[:,0] - H[:,0]
    e_b = wrap(Z[:,1] - H[:,1])

    bands = []
    for lo, hi in DISTANCE_BANDS:
        in_band = (H[:,0] >= lo) & (H[:,0] < hi)
        if in_band.sum() > 1:
            bands.append((lo, hi, int(in_band.sum()), e_r[in_band].std(), e_b[in_band].std()))

    fig, ax = plt.subplots(1,2,figsize=(8,4))
    ax[0].scatter(H[:, 0], (Z[:, 0] - H[:, 0]) * 100, s=6, label="raw")
    ax[0].scatter(H[:, 0], e_r * 100, s=6, label=f"after range scale {scale:.4f}")
    ax[0].set_ylabel("range error (cm)"); ax[0].legend(fontsize=8)
    ax[1].scatter(H[:, 0], np.degrees(e_b), s=6)
    ax[1].set_ylabel("bearing error (deg)")

    for a in ax:
        a.set_xlabel("true range (m)"); a.grid()
    fig.suptitle("solvePnP range and bearing error at 600 known poses")
    fig.tight_layout(); fig.savefig("calibration_camera.png", dpi=130)
    return scale, e_r.std(), e_b.std(), e_b.mean(), tries, bands


# plot of the Qd sweep: coverage vs scale, with the 95 % target
def plot_qd_coverage(coverage_table, s_pick):
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot([s for s,c in coverage_table], [c for s,c in coverage_table], "o-")
    ax.axhline(COVERAGE_TARGET, ls="--", c="r", label=f"target {COVERAGE_TARGET}")
    ax.axvline(s_pick, ls=":", c="g", label=f"chosen scale {s_pick}")
    ax.set_xscale("log"); ax.set_xlabel("Qd scale"); ax.set_ylabel("2-sigma coverage")
    ax.set_title("Dead-reckoning error inside 2-sigma vs Qd scale"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig("calibration_qd.png", dpi=130)

def drive_logs(radius_est, seed = DRIVE_SEED, seconds = DRIVE_TIME):
    rng = np.random.default_rng(seed)
    logs = []
    for v, w in TEST_DRIVES:
        set_robot_pose(*DRIVE_START)
        enc_prev = encoders() + rng.normal(0, ENC_NOISE, 2)
        U, X = [], [get_true_pose()]

        for _ in range(int(seconds/T)):
            drive_step(v, w)
            enc = encoders() + rng.normal(0, ENC_NOISE, 2)
            d_l, d_r = (enc - enc_prev) * radius_est
            enc_prev = enc
            U.append(((d_l + d_r) / 2 / T, (d_r - d_l) / WHEEL_SEP / T))
            X.append(get_true_pose())
        logs.append((U, X))
    return logs





def print_results(radius_est, true_distance, d_reported, Qd0, coverage_table, s_pick, sd,
                  scale, e_r_std, e_b_std, e_b_mean, tries, bands):
    text = f"""
=== Results ===
Part 1: reported {d_reported:.4f} m, true {true_distance:.4f} m
Part 2: one-step model error sd: x {np.sqrt(Qd0[0, 0]) * 1000:.3f} mm, y {np.sqrt(Qd0[1, 1]) * 1000:.3f} mm, heading {np.degrees(np.sqrt(Qd0[2, 2])):.4f} deg
        Qd scale vs 2-sigma coverage of dead-reckoning error (target {COVERAGE_TARGET}):
"""
    for s, c in coverage_table:
        text += f"          scale {s:5.0f}: coverage {c:.2f}" + ("   <- chosen" if s == s_pick else "") + "\n"
    text += f"""Part 3: marker detected at {N_CAMERA_POSES} of {tries} poses tried ({tries - N_CAMERA_POSES} rejected)
        range scale {scale:.4f}; after it, range sd {e_r_std * 100:.2f} cm, bearing sd {np.degrees(e_b_std):.3f} deg (mean {np.degrees(e_b_mean):+.3f} deg)
"""
    for lo, hi, count, sr, sb in bands:
        text += f"          {lo:.1f}-{hi:.1f} m ({count:3d} poses): range sd {sr * 100:.2f} cm, bearing sd {np.degrees(sb):.3f} deg\n"
    text += f"""
Copy into estimation.py:
WHEEL_R_CAL = {radius_est:.5f}
RANGE_SCALE = {scale:.4f}
QD = np.diag([{sd[0]:.6f} ** 2, {sd[1]:.6f} ** 2, np.deg2rad({np.degrees(sd[2]):.4f}) ** 2])
BEARING_BIAS = np.deg2rad({np.degrees(e_b_mean):.3f})
RN = np.diag([{e_r_std:.4f} ** 2, np.deg2rad({np.degrees(e_b_std):.3f}) ** 2])

Saved calibration_qd.png, calibration_camera.png and {RESULTS_FILE}"""
    print(text)
    with open(RESULTS_FILE, "w") as f:
        f.write(text)


def main():
    print("Part 1: wheel radius")
    radius_est, true_distance, d_reported = calibrate_wheel_radius()
    print("Part 2: process noise")
    Qd0, coverage_table = estimate_process_noise(drive_logs(radius_est))
    print("Part 3: camera calibration")
    scale, e_r_std, e_b_std, e_b_mean, tries, bands = calibrate_camera()

    good = [s for s,c in coverage_table if c > COVERAGE_TARGET]
    s_pick = good[0] if len(good) > 0 else coverage_table[-1][0]
    sd = np.sqrt(np.diag(Qd0) * s_pick)
    plot_qd_coverage(coverage_table, s_pick)
    print_results(radius_est, true_distance, d_reported, Qd0, coverage_table, s_pick, sd,
                  scale, e_r_std, e_b_std, e_b_mean, tries, bands)


if __name__ == "__main__":
    main()

#importing the required libraries
import numpy as np
import cv2
import mujoco
import matplotlib.pyplot as plt

from modelling import WHEEL_R, WHEEL_SEP, CAM_OFFSET, MARKER, T, encoders_to_u, f_uni, F_uni, h_rb, H_rb, h_range, wrap

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
    dist_true = np.hypot(*(p1[:2]-pose_start[:2]))

    d_reported = WHEEL_R * (e1-e0).mean() 
    return WHEEL_R * dist_true / d_reported, dist_true, d_reported # the true distance, and scale the radius: r_hat = r * d_true / d_reported

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





def print_results(radius_est, dist_true, d_reported, Qd0, coverage_table, s_pick, sd,
                  scale, e_r_std, e_b_std, e_b_mean, tries, bands):
    text = f"""
=== Results ===
Part 1 (Week 2 B.3): reported {d_reported:.4f} m, true {dist_true:.4f} m
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
    radius_est, dist_true, d_reported = calibrate_wheel_radius()
    print("Part 2: process noise")
    Qd0, coverage_table = estimate_process_noise(drive_logs(radius_est))
    print("Part 3: camera calibration")
    scale, e_r_std, e_b_std, e_b_mean, tries, bands = calibrate_camera()

    good = [s for s,c in coverage_table if c > COVERAGE_TARGET]
    s_pick = good[0] if len(good) > 0 else coverage_table[-1][0]
    sd = np.sqrt(np.diag(Qd0) * s_pick)
    plot_qd_coverage(coverage_table, s_pick)
    print_results(radius_est, dist_true, d_reported, Qd0, coverage_table, s_pick, sd,
                  scale, e_r_std, e_b_std, e_b_mean, tries, bands)


if __name__ == "__main__":
    main()
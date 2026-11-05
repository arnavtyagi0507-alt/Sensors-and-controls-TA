#importing the required libraries
import numpy as np
import sympy as sp
import control as ct

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
    import matplotlib.pyplot as plt
    np.set_printoptions(precision=3, suppress=True)

    print('jacobian checks')
    xt, ut = np.array([-0.3, 1.5, 0.5]), np.array([0.13, -0.2])
    print("Largest difference between analytic and numerical F: =",
          np.abs(F_uni(xt, ut) - num_jac(lambda s: f_uni(s, ut), xt)).max())
    print("encoders_to_u(1 rad, 1 rad) =", encoders_to_u(1.0, 1.0), "(straight, 0.66 m/s)")

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
    print("Full 3-state model at 0.15 m/s - controllability rank:", np.linalg.matrix_rank(ct.ctrb(A_full, B_full)),
      " stabilisable (PBH):", pbh_stabilisable(A_full, B_full))
    print("Rank drops at zero speed: a stopped robot cannot correct a sideways error.")

    operating_point = np.array([0.20,0.04,0.0])
    print(f"Observability, linearised at pose {operating_point} with heading 0:")
    print("Odometry only: no measurement, C is empty -> nothing observable.")

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


    print("Range + bearing: observable only while the robot moves. Standing still,")
    print("one direction cannot be told from the measurement. Range only is not observable.")

    fig, ax = plt.subplots(1,2, figsize = (10,4))
    speeds = np.linspace(0,0.2,101)
    ax[0].plot(speeds, [np.abs(np.linalg.det(ct.ctrb(*tracking_error_ss(V)))) for V in speeds])
    ax[0].axvline(0.15, ls="--", c="g", label="navigation speed")
    ax[0].set_xlabel("v0 (m/s)"); ax[0].set_ylabel("|det [B, AB]|")
    ax[0].set_title("Controllability of [e_y, e_theta]"); ax[0].legend(); ax[0].grid(alpha=.3)

    for name, Hf in (("range only", H_range), ("range + bearing", H_rb)):
        C = Hf(operating_point)
        ax[1].plot(speeds, [np.linalg.svd(ct.obsv(unicycle_lin(V)[0], C), compute_uv=False)[-1]
                        for V in speeds], label=name)
    ax[1].set_xlabel("v0 (m/s)"); ax[1].set_ylabel("smallest singular value of O")
    ax[1].set_title("Observability vs speed"); ax[1].legend(); ax[1].grid(alpha=.3)

    fig.tight_layout()
    fig.savefig("modelling_results.png", dpi=130)
    print("Figure written to modelling_results.png")


if __name__ == "__main__":
    main()
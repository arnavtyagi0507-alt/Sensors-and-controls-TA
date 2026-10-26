"""Run the integrated Track A simulation and repeated robustness trials."""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from src.SensingNoiseCalibration import NoiseConfig
from src.robustness import Disturbance, compare_conditions, make_trial_seeds
from trackA_demo import MARKER_XY, OBSTACLES, T, run


def main(argv=None):
    parser = argparse.ArgumentParser(description="Track A sensing, estimation and control demo")
    parser.add_argument("--seed", type=int, default=41014, help="base seed for reproducible trials")
    parser.add_argument("--headless", action="store_true", help="run without viewer or plots")
    parser.add_argument("--robustness", action="store_true",
                        help="compare repeated baseline and disturbed conditions")
    parser.add_argument("--trials", type=int, default=10,
                        help="trials per condition in robustness mode")
    args = parser.parse_args(argv)

    if args.robustness:
        seeds = make_trial_seeds(args.trials, args.seed)
        conditions = {
            "baseline": Disturbance(),
            "disturbed": Disturbance(
                encoder_bias_rad_per_sample=0.001,
                lidar_bias_m=0.03,
                camera_dropout_probability=0.15,
                lidar_dropout_probability=0.03,
                wheel_scale_left=0.98,
                wheel_scale_right=1.02,
                start_x_offset_m=0.08,
                start_y_offset_m=-0.08,
                start_heading_offset_rad=np.radians(5),
            ),
        }
        condition_results = {}
        for name, disturbance in conditions.items():
            condition_results[name] = []
            for trial_seed in seeds:
                result = run(
                    live=False,
                    seed=trial_seed,
                    disturbance=disturbance,
                    noise_config=NoiseConfig(),
                )
                condition_results[name].append(result)
                print(f"{name} seed={trial_seed}: "
                      f"success={result['success']} "
                      f"position RMSE={result['position_rmse_m']:.3f} m "
                      f"colliding steps={result['collisions']}")
        for row in compare_conditions(condition_results):
            print(row)
        return

    result = run(
        live=not args.headless,
        seed=args.seed,
        disturbance=Disturbance(),
        noise_config=NoiseConfig(),
    )
    est = result["estimated_poses"]
    truth = result["true_poses"]
    error = np.linalg.norm(est[:, :2] - truth[:, :2], axis=1)
    print(f"seed {args.seed}  success={result['success']}  "
          f"final position error {error[-1] * 100:.1f} cm, "
          f"mean {error.mean() * 100:.1f} cm; "
          f"dock offset {truth[-1, 1] * 100:+.1f} cm; "
          f"colliding steps {result['collisions']}")
    if args.headless:
        return

    _, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(truth[:, 0], truth[:, 1], "k-", label="true path")
    axes[0].plot(est[:, 0], est[:, 1], "b--", label="EKF estimate")
    for obstacle in OBSTACLES:
        axes[0].add_patch(plt.Circle(obstacle, 0.18, color="r", alpha=.5))
    axes[0].plot(*MARKER_XY, "gs", ms=10, label="marker")
    axes[0].set_xlim(-2.5, 2.5)
    axes[0].set_ylim(-2.5, 2.5)
    axes[0].set_aspect("equal")
    axes[0].legend()
    axes[0].grid()
    axes[1].plot(np.arange(len(error)) * T, error * 100)
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("EKF position error (cm)")
    axes[1].grid()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

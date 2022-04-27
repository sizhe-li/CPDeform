import os
import argparse
import pykeops

Envs = [
    'multistage_airplane',
    'multistage_chair',
    'multistage_bottle',
    'multistage_move',
    'multistage_rope',
    'multistage_writer',
    'multistage_star',
]

FILE_PATH = os.path.dirname(os.path.abspath(__file__))

DIFF_PHYS_ALGOS = ["cpdeform", "fixed"]
RL_ALGOS = []  # upcoming, stay tuned!
ALGOS = DIFF_PHYS_ALGOS + RL_ALGOS


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=ALGOS)
    parser.add_argument("--env_name", type=str, choices=Envs)
    parser.add_argument("--seed", default=1234, type=str)
    parser.add_argument("--device", default='0', type=str)
    parser.add_argument("--root_dir", type=str,
                        default='/disk1/long_horizon_experiments/diff_phys/')

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    # read script
    cmd_path = FILE_PATH

    cmd = f"CUDA_VISIBLE_DEVICES={args.device} "
    if args.algo in DIFF_PHYS_ALGOS:
        folder = "cpdeform" if args.algo == "cpdeform" else "baseline"
        cmd_path = os.path.join(cmd_path, "plab-m/remote_training", folder, f"train_{args.env_name}.txt")

        with open(cmd_path) as f:
            handcrafted_cmd = f.readline().strip('\n')
        cmd += f"python3 -m CPDeform.solve_env {handcrafted_cmd} " \
               f"--algo {args.algo} --seed {args.seed} --root_dir {args.root_dir}"

    else:
        raise NotImplementedError

    print("[Launching Job...]:", cmd)
    os.system(cmd)


if __name__ == '__main__':
    main()

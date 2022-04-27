import os
import torch
import shutil
import random
import datetime
import numpy as np
import argparse

from plb.envs import PlasticineEnv
from .training_initializers import ENV_TO_HEURISTIC, Reshape, Transport, MAX_ENV_STEPS

PATH = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(PATH, '../plb/', 'envs/env_configs/')
base_path = CONFIG_DIR + '{}.yml'


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


DIFF_PHYS_ALGOS = ["cpdeform", "fixed"]


def get_args():
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument("--algo", required=True, choices=DIFF_PHYS_ALGOS)
    parser.add_argument("--env_name", type=str, choices=list(ENV_TO_HEURISTIC.keys()))
    parser.add_argument("--root_dir", type=str,
                        default='/disk1/long_horizon_experiments/diff_phys/')
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--version", type=int, default=1)
    # heuristic solver
    parser.add_argument("--n_trials", type=int, default=2000)
    parser.add_argument("--n_iters", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--use_early_stopper", action='store_true')
    parser.add_argument("--use_padding", action='store_true',
                        help='action padding for stability')
    parser.add_argument("--stage_max_n_attempts", type=int, default=5)
    # th solver
    parser.add_argument("--use_prim_dist_loss", action='store_true')
    parser.add_argument("--use_contact_loss", action='store_true')
    # ti solver
    parser.add_argument("--soft_contact_loss", action='store_true')
    # primitive sampler
    parser.add_argument("--prim_sampler", type=str, default=None)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--min_dist", type=float, default=0.015)
    parser.add_argument("--n_prims_per_gripper", type=int, default=1)
    parser.add_argument("--neighborhood_size", type=float, default=0.1)
    parser.add_argument("--prim_dist_thresh", type=float, default=0.01)
    parser.add_argument("--topn_potentials", type=int, default=0)
    parser.add_argument("--max_dist_traveled", type=float, default=0.5)
    parser.add_argument("--p", type=int, default=1)
    parser.add_argument("--blur", type=float, default=0.001)
    # misc
    parser.add_argument("--init_phase", type=str, default='start')
    parser.add_argument("--goal_phase", type=str, default='final')
    # whether to use open3d to render geometries
    parser.add_argument("--use_open3d", action="store_true")

    parser.add_argument("--env_dir", type=str, default="")
    parser.add_argument("--target_path", type=str, default="")

    args = parser.parse_args()
    return args


def get_experiment_id(args):
    this_dict = vars(args)
    exp_id = []
    for k in this_dict.keys():
        v = str(getattr(args, k))
        exp_id.append(f'{k}_{v}')
    exp_id = '-'.join(exp_id)

    return exp_id


def main():
    args = get_args()
    root_dir = os.path.join(args.root_dir, '{}/v{:1d}/')

    set_random_seed(args.seed)
    args.exp_id = get_experiment_id(args)
    #######################
    ## Setup directories ##
    #######################

    env_name, version = args.env_name, args.version
    exp_id = args.exp_id

    env_path = base_path.format(env_name)
    if not args.env_dir:
        args.env_dir = root_dir.format(env_name, version)
    else:
        print(f"using env_dir: {args.env_dir}")
    env_dir = args.env_dir
    assert os.path.exists(env_dir)

    exp_dir = os.path.join(env_dir, 'exp_dir')
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    exp_num = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    exp_name = f"{args.algo}_{exp_num}"
    exp_dir = args.exp_dir = os.path.join(exp_dir, exp_name)
    if os.path.exists(exp_dir):
        shutil.rmtree(exp_dir)
    os.makedirs(exp_dir)

    file = open(os.path.join(exp_dir, "exp_id.txt"), "w")
    file.write(exp_id)
    file.close()

    ##################
    ## Load Configs ##
    ##################

    cfg = PlasticineEnv.load_varaints(env_path, version=version)
    if args.target_path:
        assert os.path.exists(args.target_path)
        cfg.defrost()
        cfg.ENV.loss.target_path = args.target_path
        print(f"using target path: {args.target_path}")

    ###############
    ## Start Job ##
    ###############

    job_type = ENV_TO_HEURISTIC[args.env_name]
    args.max_env_steps = MAX_ENV_STEPS.get(args.env_name, np.inf)

    if isinstance(job_type, Reshape):
        if job_type == Reshape.COLLIDE_SINGLE_PRIMITIVE:
            from .training_initializers.reshape.collide_single_primitive import solve
            solve(args, cfg)
        elif job_type == Reshape.CLAMP_DOUBLE_PRIMITIVE:
            from .training_initializers.reshape.clamp_double_primitive import solve
            solve(args, cfg)

    elif isinstance(job_type, Transport):
        if job_type == Transport.CLAMP_DOUBLE_PRIMITIVE:
            from .training_initializers.transport.clamp_double_primitive import solve
            solve(args, cfg)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    main()

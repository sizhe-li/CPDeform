import os
import pickle
import numpy as np

from plb.engine.taichi_env import TaichiEnv

from CPDeform.utils.base_logger import StageLogger
from CPDeform.solver.iter_deform import IterDeformSolver


def solve(args, cfg):
    env_dir, exp_dir = args.env_dir, args.exp_dir

    # hacky solution
    cfg.defrost()
    cfg.ENV.loss.soft_contact = args.soft_contact_loss
    cfg.freeze()
    lower_bound = (0.0, 0.0, 0.0)
    upper_bound = (1.0, 1.0, 1.0)

    ##############
    ## load env ##
    ##############

    taichi_env = TaichiEnv(cfg, loss=True)
    taichi_env.initialize()
    init_state = taichi_env.get_state()

    pose_rots, pose_dirs = [], []
    for pose in cfg.PRIMITIVES[0]['poses']:
        rot = eval(pose['rot'])
        dir_ = eval(pose['dir'])
        pose_rots.append(rot)
        pose_dirs.append(dir_)

    ############
    ## Logger ##
    ############

    h, w = cfg.RENDERER.image_res
    camera_setting = os.path.join(env_dir, 'camera_setting.pkl')
    camera_setting = pickle.load(open(camera_setting, 'rb'))
    cam_positions = camera_setting['positions']
    cam_rotations = camera_setting['rotations']
    logger = StageLogger(exp_dir,
                         use_open3d=args.use_open3d,
                         cam_positions=cam_positions,
                         cam_rotations=cam_rotations,
                         w=w, h=h)

    ####################################
    ## Visualize init and goal states ##
    ####################################

    goal_obs_path = os.path.join(env_dir, 'target_particles.npy')
    print(goal_obs_path)
    goal_obs = np.load(goal_obs_path)
    taichi_env.simulator.reset(goal_obs)
    logger.log_state_as_image(taichi_env,
                              filename='goal_state.png',
                              primitive=0)

    taichi_env.set_state(**init_state)
    taichi_env.set_gmass_target_from_particles(goal_obs)

    if (init_phase := args.init_phase) != 'start':
        init_obs_path = os.path.join(env_dir, f'{init_phase}.npy')
        init_obs = np.load(init_obs_path)
        taichi_env.simulator.reset(init_obs)

    logger.log_state_as_image(taichi_env,
                              filename='init_state.png',
                              primitive=0)
    ################
    ## Run Solver ##
    ################

    prim_sampler = None
    if args.algo == "cpdeform":
        prim_sampler = 'ClampReshape'
    elif args.algo == "fixed":
        prim_sampler = 'FixedStartBaseline'

    # use appointed primitive sampler
    if args.prim_sampler is not None:
        prim_sampler = args.prim_sampler

    solver = IterDeformSolver(taichi_env, goal_obs,
                              logger, cfg=None,
                              max_env_steps=args.max_env_steps,
                              solver='th',
                              prim_sampler=prim_sampler,
                              n_iters=args.n_iters,
                              lr=args.lr,
                              horizon=args.horizon,
                              use_early_stopper=args.use_early_stopper,
                              use_padding=args.use_padding,
                              stage_max_n_attempts=args.stage_max_n_attempts,
                              **{
                                  'prim_sampler_cfg.topk': args.topk,
                                  'prim_sampler_cfg.pose_rots': pose_rots,
                                  'prim_sampler_cfg.pose_dirs': pose_dirs,
                                  'prim_sampler_cfg.min_dist': args.min_dist,
                                  'prim_sampler_cfg.lower_bound': lower_bound,
                                  'prim_sampler_cfg.upper_bound': upper_bound,
                                  'prim_sampler_cfg.n_prims_per_gripper': args.n_prims_per_gripper,
                                  'prim_sampler_cfg.topn_potentials': args.topn_potentials,
                                  'prim_sampler_cfg.neighborhood_size': args.neighborhood_size,
                                  'prim_sampler_cfg.max_dist_traveled': args.max_dist_traveled,
                                  'loss.prim_dist_thresh': args.prim_dist_thresh,
                                  'loss.use_prim_dist_loss': args.use_prim_dist_loss,
                                  'loss.use_contact_loss': args.use_contact_loss,
                                  'loss.p': args.p,
                                  'loss.blur': args.blur
                              }
                              )

    solver.iter_deform(args.n_trials)

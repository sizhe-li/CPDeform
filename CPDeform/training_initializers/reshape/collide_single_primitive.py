import os
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

    taichi_env = TaichiEnv(cfg, loss=True)
    taichi_env.initialize()
    init_state = taichi_env.get_state()

    pose_rots = [tuple(taichi_env.primitives[0].rotation.to_numpy()[0])]

    ############
    ## Logger ##
    ############

    h, w = cfg.RENDERER.image_res
    cam_positions = [cfg.RENDERER.camera_pos]
    cam_rotations = [cfg.RENDERER.camera_rot]
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
    logger.log_state_as_image(taichi_env,
                              filename='init_state.png',
                              primitive=0)
    ################
    ## Run Solver ##
    ################

    prim_sampler = None
    if args.algo == "cpdeform":
        prim_sampler = 'CollideReshape'
    elif args.algo == "fixed":
        prim_sampler = 'FixedStartBaseline'

    # use appointed primitive sampler
    if args.prim_sampler is not None:
        prim_sampler = args.prim_sampler

    solver = IterDeformSolver(taichi_env, goal_obs,
                              logger, cfg=None,
                              max_env_steps=args.max_env_steps,
                              solver='ti',
                              prim_sampler=prim_sampler,
                              n_iters=args.n_iters,
                              lr=args.lr,
                              horizon=args.horizon,
                              use_early_stopper=args.use_early_stopper,
                              use_padding=args.use_padding,
                              **{
                                 'prim_sampler_cfg.topk': args.topk,
                                 'prim_sampler_cfg.pose_rots': pose_rots,
                                 'prim_sampler_cfg.min_dist': args.min_dist,
                                 'prim_sampler_cfg.neighborhood_size': args.neighborhood_size,
                                 'prim_sampler_cfg.topn_potentials': args.topn_potentials,
                                 'loss.prim_dist_thresh': args.prim_dist_thresh,
                                 'loss.p': args.p,
                                 'loss.blur': args.blur
                             }
                              )

    solver.iter_deform(args.n_trials)

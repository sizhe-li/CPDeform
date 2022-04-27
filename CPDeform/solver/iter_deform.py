import torch
import numpy as np
from yacs.config import CfgNode as CN

from CPDeform.loss import OptimalTransportLoss
from CPDeform.solver.grad_solver.th_solver import TorchSolver
from CPDeform.solver.grad_solver.ti_solver import TaichiSolver
from CPDeform.utils.base_logger import StageLogger
from CPDeform.manipulator_sampler import *
from plb.config.utils import make_cls_config
from plb.engine.taichi_env import TaichiEnv


class IterDeformSolver:
    def __init__(self,
                 taichi_env: TaichiEnv,
                 goal_obs: np.ndarray,
                 stage_logger: StageLogger,
                 cfg=None,
                 device='cuda:0',
                 **kwargs
                 ):
        self.cfg = cfg = make_cls_config(self, cfg, **kwargs)
        print('long-horizon-solver config \n', cfg)

        self.env_steps_cnt = 0
        self.max_env_steps = cfg.max_env_steps
        self.env_steps_maxed_out = False

        self.device = device
        self.env = taichi_env
        self.stage_logger = stage_logger

        self.goal_obs = goal_obs = torch.from_numpy(goal_obs).to(self.device)
        self.loss = OptimalTransportLoss(taichi_env, goal_obs, cfg.loss)

        # initialize solver
        if self.cfg.solver == 'ti':
            self.solver = TaichiSolver(taichi_env, self.loss)
        elif self.cfg.solver == 'th':
            self.solver = TorchSolver(taichi_env, self.loss)
        else:
            raise NotImplementedError("incorrect solver type!")

        SamplerClass = eval(self.cfg.prim_sampler)
        assert issubclass(SamplerClass, PrimitiveSampler)
        self.prim_sampler = SamplerClass(taichi_env, self.loss, cfg=cfg.prim_sampler_cfg)

        self.best_overall_loss = np.inf

    def reset_cnt(self):
        self.stage_logger.reset()
        self.best_overall_loss = self.loss.start_loss if self.cfg.solver == 'th' else np.inf

    def iter_deform(self, n_stages, horizon=None):
        # use default horizon if not specified
        if horizon is None: horizon = self.cfg.horizon

        state = self.env.get_state()
        self.env.set_state(**state)

        self.reset_cnt()
        # at stage = 0, we record init loss and init iou to our log
        self.stage_logger.step(self.loss.start_loss, self.env.loss._init_iou)

        best_actions_overall = []
        best_plans_overall = []
        for n in range(n_stages):

            # increment stage cnt and visualize the new state
            self.stage_logger.increment_stage_cnt()
            self.stage_logger.log_state_as_image(self.env,
                                                 f"stage_{self.stage_logger.stage_cnt:05d}-init_state.png",
                                                 primitive=0)

            stage_best_loss = np.inf
            stage_best_plan = None
            stage_best_actions = None
            while True:
                print(f'stage: {self.stage_logger.stage_cnt:05d} | '
                      f'attempt: {self.stage_logger.attempt_cnt:05d} | '
                      f'best_loss: {self.best_overall_loss:.6f}')

                stage_outputs = self.solve_single_stage(horizon)

                self.stage_logger.step(stage_outputs['best_loss'], stage_outputs['best_incremental_iou'])

                if stage_outputs['best_loss'] < stage_best_loss:
                    stage_best_loss = stage_outputs['best_loss']
                    stage_best_plan = stage_outputs['best_plan']
                    stage_best_actions = stage_outputs['best_action']

                if (stage_best_loss < self.best_overall_loss) or \
                        (self.stage_logger.attempt_cnt > self.cfg.stage_max_n_attempts) or \
                        self.env_steps_maxed_out:
                    break

            self.best_overall_loss = stage_best_loss
            # reset explored inds
            self.prim_sampler.reset()

            # 1. unpack and set up plan
            self.prim_sampler.setup_single_plan(stage_best_plan)
            best_plans_overall.append(stage_best_plan)

            # 2. get actions
            # padding to stabilize
            if self.cfg.use_padding:
                padding = np.zeros((5, self.env.primitives.action_dim))
                stage_best_actions = np.concatenate([stage_best_actions, padding], axis=0)

            # 3. take actions
            images = []
            for act_idx, act in enumerate(stage_best_actions):
                self.env.step(act)
                images.append(self.stage_logger.render_state_multiview(self.env))
            best_actions_overall.append(stage_best_actions)

            # 4. log best attempt as videos
            self.stage_logger.log_images_as_video(images)
            self.stage_logger.log_best_trajectory(best_actions_overall, best_plans_overall)

            if self.env_steps_maxed_out:
                break

        self.env.set_state(**state)

        return {'best_actions': best_actions_overall,
                'best_plans': best_plans_overall,
                }

    def solve_single_stage(self, horizon=None):
        if horizon is None:
            horizon = self.cfg.horizon

        plans, potential = self.prim_sampler.sample_stage_plans(visualize=True)
        # log potential landscape
        self.stage_logger.log_geometry_as_image(potential, f"stage_{self.stage_logger.stage_cnt:05d}-potential.png")

        best_loss = np.inf
        best_plan = None
        best_action = None
        best_incremental_iou = None

        for plan_cnt, curr_plan in enumerate(plans):
            init_actions = self.init_actions(self.env, self.cfg, horizon)
            outputs = self.solve_plan(plan_cnt + 1, init_actions, curr_plan, best_loss)

            curr_loss = outputs['best_loss']
            if curr_loss < best_loss:
                best_loss = curr_loss
                best_plan = curr_plan
                best_action = outputs['best_action']
                best_incremental_iou = outputs['best_incremental_iou']

            self.env_steps_cnt += outputs['num_env_steps']
            if env_steps_maxed_out := (self.env_steps_cnt >= self.max_env_steps):
                self.env_steps_maxed_out = env_steps_maxed_out
                break

        return {
            'best_loss': best_loss,
            'best_plan': best_plan,
            'best_action': best_action,
            'best_incremental_iou': best_incremental_iou,
        }

    def solve_plan(self, plan_cnt, init_actions, plan, best_loss):
        header = f'stage:{self.stage_logger.stage_cnt:05d} | ' \
                 f'attempt:{self.stage_logger.attempt_cnt + 1:05d} | ' \
                 f'plan:{plan_cnt:01d}'

        state = self.env.get_state()

        self.prim_sampler.setup_single_plan(plan)

        # visualize current plan
        img = self.stage_logger.render_state_multiview(self.env, primitive=1)
        img = img[:, :, ::-1]

        solver_kwargs = {'lr': self.cfg.lr, 'n_iters': self.cfg.n_iters, 'header': header}
        if self.cfg.use_early_stopper:
            from CPDeform.utils.training_utils import EarlyStopper
            early_stopper = EarlyStopper(patience=int(self.cfg.n_iters // 4))
            early_stopper.reset(best_loss)
            solver_kwargs['early_stopper'] = early_stopper

        step_logger = self.stage_logger.get_step_logger(plan_cnt, self.env_steps_cnt)
        outputs = self.solver.solve(init_actions, logger=step_logger, **solver_kwargs)

        self.stage_logger.save_img(img, f"stage_{self.stage_logger.stage_cnt:05d}-"
                                        f"attempt_{self.stage_logger.attempt_cnt:05d}-"
                                        f"plan_{plan_cnt}_{outputs['best_loss']:.6f}.png")

        self.env.set_state(**state)

        return outputs

    @staticmethod
    def init_actions(env, cfg, horizon):
        action_dim = env.primitives.action_dim
        if cfg.init_sampler == 'uniform':
            return np.random.uniform(-cfg.init_range, cfg.init_range, size=(horizon, action_dim))
        elif cfg.init_sampler == 'normal':
            return np.random.randn(horizon, action_dim) * 0.01
        else:
            raise NotImplementedError

    @classmethod
    def default_config(cls):
        cfg = CN()

        # iter_deform
        cfg.max_env_steps = np.inf
        cfg.stage_max_n_attempts = 5

        # diff. phys. optimizer
        cfg.solver = 'th'
        cfg.n_iters = 200
        cfg.horizon = 10
        cfg.lr = 0.01
        cfg.init_range = 0.0001
        cfg.init_sampler = 'normal'
        cfg.use_early_stopper = False

        # stabilize action by padding
        cfg.use_padding = False

        # primitive sampler
        cfg.prim_sampler = ''
        cfg.prim_sampler_cfg = PrimitiveSampler.default_config()

        # loss
        cfg.loss = OptimalTransportLoss.default_config()

        return cfg

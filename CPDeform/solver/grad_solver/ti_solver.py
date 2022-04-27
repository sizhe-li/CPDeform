import torch
import tqdm
import taichi as ti
import numpy as np
from yacs.config import CfgNode as CN

from plb.optimizer.optim import Adam
from plb.engine.taichi_env import TaichiEnv
from plb.config.utils import make_cls_config
from CPDeform.utils.base_logger import StepLogger
from CPDeform.loss import OptimalTransportLoss


class TaichiSolver:
    def __init__(self,
                 env: TaichiEnv,
                 loss_fn: OptimalTransportLoss,
                 cfg=None,
                 **kwargs):

        self.cfg = make_cls_config(self, cfg, **kwargs)
        self.env = env
        self.loss_fn = loss_fn
        self.loss_fn.set_start_loss(self.env.loss._start_loss)

    def solve(self,
              init_actions, logger: StepLogger=None,
              lr=0.01, n_iters=200,
              header=None,
              early_stopper=None,
              verbose=True):

        env = self.env
        optim = Adam(init_actions, lr=lr)
        # set softness ..
        env_state = env.get_state()

        def forward(sim_state, action):

            env.set_state(sim_state, self.cfg.softness, False)
            with ti.Tape(loss=env.loss.loss):
                for i in range(len(action)):
                    env.step(action[i])
                    loss_info = env.compute_loss()
                    if logger is not None:
                        logger.step()

            loss = env.loss.loss[None]

            x = self.env.simulator.get_x(self.env.simulator.cur)
            x = torch.from_numpy(x).to(self.loss_fn.goal_obs.device)
            ot_loss = self.loss_fn.calculate_ot_loss(self.loss_fn.last_index, x)
            loss_info['particle_loss'] = ot_loss['particle_loss'].item()
            if logger is not None:
                logger.record(loss_info)

            return loss, env.primitives.get_grad(len(action)), loss_info

        best_action, best_loss, best_incremental_iou, best_particle_loss = None, np.inf, None, None
        actions = init_actions

        ran = tqdm.trange if verbose else range
        it = ran(n_iters)

        loss = None
        for iter_id in it:
            loss, grad, loss_info = forward(env_state['state'], actions)

            if loss < best_loss:
                best_loss = loss
                best_action = actions.copy()
                best_incremental_iou = loss_info["incremental_iou"]
                best_particle_loss = loss_info["particle_loss"]

            actions = optim.step(grad)

            if verbose:
                iter_header = iter_id if header is None else header
                it.set_description(f"{iter_header} | loss: {loss:.06f}", refresh=True)

            if early_stopper is not None:
                stopping, improved = early_stopper(best_loss, iter_id)
                if stopping:
                    break

        env.set_state(**env_state)
        return {
            'best_loss': best_loss,
            'best_action': best_action,
            'best_incremental_iou': best_incremental_iou,
            'best_particle_loss': best_particle_loss,
            'last_loss': loss,
            'last_action': actions,
            'num_env_steps': (iter_id + 1) * len(actions)
        }

    @classmethod
    def default_config(cls):
        cfg = CN()
        cfg.softness = 666.

        return cfg

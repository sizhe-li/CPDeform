import torch
import pykeops

from geomloss import SamplesLoss
from yacs.config import CfgNode as CN
from plb.config.utils import make_cls_config
from plb.engine.taichi_env import TaichiEnv


class OptimalTransportLoss:
    def __init__(self, env: TaichiEnv, goal_obs, cfg=None, **kwargs):
        self.cfg = cfg = make_cls_config(self, cfg, **kwargs)

        self.env = env
        self._start_loss = None

        self.last_index = None
        self.goal_obs = goal_obs

        self.ot_solver = SamplesLoss(loss='sinkhorn', p=cfg.p, blur=cfg.blur)

    @property
    def start_loss(self):
        assert self._start_loss is not None
        return self._start_loss

    def set_start_loss(self, start_loss):
        self._start_loss = start_loss

    def set_target(self, obs):
        self.goal_obs = obs

    def set_horizon(self, horizon):
        self.last_index = horizon - 1

    def _extract_loss(self, idx, x, c):
        loss = dict()

        if self.cfg.use_prim_dist_loss:
            loss.update(self.calculate_primitive_distance_loss(idx, c))
        if self.cfg.use_contact_loss:
            loss.update(self.calculate_contact_loss(idx, x))

        loss.update(self.calculate_ot_loss(idx, x))
        return loss

    def compute_loss_info(self, idx, loss_dict):
        loss_info = dict()

        s = (idx + 1) * self.env.simulator.substeps
        loss_info['incremental_iou'] = 0

        if idx == self.last_index:
            loss_info['loss'] = loss_info['particle_loss'] = loss_dict['particle_loss'].item()

        return loss_info

    def __call__(self, idx, x, c):
        loss = self._extract_loss(idx, x, c)
        return loss

    def calculate_primitive_distance_loss(self, idx, c):
        loss = dict()

        pos1 = c[0, :3]
        pos2 = c[1, :3]

        dist_loss = torch.pow(torch.norm(pos1 - pos2, p=2), 2)
        if dist_loss.item() <= self.cfg.prim_dist_thresh:
            loss['dist_loss'] = dist_loss
            # loss['dist_loss'] = dist_loss * 100

        return loss

    def calculate_contact_loss(self, idx, x):
        loss = dict()

        if idx <= self.last_index // 2:
            contact_loss = torch.relu(x[:, self.env.simulator.dim * 2:].min(0)[0]).sum()
            loss['contact_loss'] = contact_loss

        return loss

    def calculate_ot_loss(self, idx, x):
        loss = dict()
        if idx == self.last_index:
            curr_obs = x[:, :3].contiguous()
            particle_loss = self.ot_solver(curr_obs, self.goal_obs)
            loss['particle_loss'] = particle_loss
        return loss

    def calculate_potentials(self, x, y):
        self.ot_solver.potentials = True
        F, G = self.ot_solver(x, y)
        self.ot_solver.potentials = False
        return F, G

    def calculate_transport_gradient(self, x, y):
        x = x.clone().requires_grad_()
        L = self.ot_solver(x, y)
        [g] = torch.autograd.grad(L, [x])
        return g.detach()

    @staticmethod
    def get_heuristic_loss(loss_dict):
        return loss_dict['particle_loss'].detach().item()

    @classmethod
    def default_config(cls):
        cfg = CN()
        cfg.prim_dist_thresh = 0.000
        cfg.p = 1
        cfg.blur = 0.001

        cfg.use_prim_dist_loss = False
        cfg.use_contact_loss = False

        return cfg

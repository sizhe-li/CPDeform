import itertools

import numpy as np
import torch
from torch_geometric.nn import knn
from yacs.config import CfgNode as CN

from CPDeform.loss import OptimalTransportLoss
from plb.config.utils import make_cls_config
from plb.engine.taichi_env import TaichiEnv


def length(x):
    return np.sqrt(x.dot(x) + 1e-14)


def find_knn(x, F, target_idx, k=5, topm=None):
    """
        Returns the indices of the k nearest neighbors of the x[target_idx] in a list.

        If topm is set, we sort the k neighbors by F and grab the top m of them.
    """

    y = x[target_idx:target_idx + 1]  # (1, 3)

    nn_inds = knn(x, y, k)[1].cpu().numpy().tolist()

    if topm is not None:
        nn_F = F[nn_inds]
        sorted_nn_F_inds = nn_F.cpu().numpy().argsort()[::-1]
        sorted_nn_F_inds = sorted_nn_F_inds.tolist()[:topm]
        nn_inds = [nn_inds[idx] for idx in sorted_nn_F_inds]

    return nn_inds + [target_idx]


def get_gripper_to_prim_table(prim_list, n_prims_per_gripper):
    gripper_ids_to_prim_inds = dict()
    for i in range(len(prim_list) // n_prims_per_gripper):
        gripper_ids_to_prim_inds[i] = prim_list[i * n_prims_per_gripper: (i + 1) * n_prims_per_gripper]
    return gripper_ids_to_prim_inds


class PrimitiveSampler:
    def __init__(self,
                 taichi_env: TaichiEnv,
                 loss: OptimalTransportLoss,
                 cfg=None,
                 **kwargs):

        self.cfg = cfg = make_cls_config(self, cfg, **kwargs)
        self.env = taichi_env
        self.sim = taichi_env.simulator
        self.loss = loss

        ##################
        # object-related #
        ##################

        # keep track of particles that have been used
        self.explored_inds = []
        # which object are we placing manipulator for
        self.curr_obj_idx = 0
        # object id of every particle
        self.particle_obj_id = self.sim.particle_obj_id.to_numpy()
        self.obj_particle_bool_map = None
        self.update_obj_particle_bool_map(obj_idx=0)

        #######################
        # manipulator-related #
        #######################

        self.curr_manipulator_idx = 0
        # this is for selecting our manipulator, since some geometries are part of the environment
        self.actionable_prim_inds = self.sim.primitives.actionable_prim_list
        self.manipulator2primitive = get_gripper_to_prim_table(self.actionable_prim_inds, cfg.n_prims_per_gripper)
        self.n_manipulators = len(self.manipulator2primitive.keys())
        assert len(set(self.particle_obj_id)) >= self.n_manipulators

        self.lower_bound = np.array(self.cfg.lower_bound)
        self.upper_bound = np.array(self.cfg.upper_bound)

    def reset(self):
        self.explored_inds = []

    def add_explored_idx(self, idx):
        self.explored_inds.append(idx)

    def update_obj_particle_bool_map(self, obj_idx):
        self.obj_particle_bool_map = (self.particle_obj_id == obj_idx)

    def get_curr_prim_inds(self):
        return self.get_prim_inds(self.curr_manipulator_idx)

    def get_prim_inds(self, gripper_idx):
        return self.manipulator2primitive[gripper_idx]

    def sort_by_potentials(self, F, topk, obj_inds):

        sorted_inds = F.argsort(descending=True).tolist()
        sorted_inds = [x for x in sorted_inds if obj_inds[x] not in self.explored_inds]
        if topk is not None and topk > 0:
            sorted_inds = sorted_inds[:topk]
        sorted_inds = np.array(sorted_inds)

        return sorted_inds

    def compute_sdf_dists(self, prim_idx, loc):
        self.sim.primitives.primitives[prim_idx].set_position(loc)
        self.sim.compute_primitive_sdf(0)
        sdf_dists = self.sim.sdf_dists[prim_idx].to_numpy()
        return sdf_dists[self.obj_particle_bool_map]

    def gen_neighborhood(self, poi, prim_idx, pose_idx=None):
        """
            grid search around point of interest (poi)
        """

        side_length = [self.cfg.neighborhood_size] * 3
        if pose_idx is not None:
            side_length[pose_idx] += 0.2

        n_spaces = 20
        x = np.linspace(poi[0] - side_length[0], poi[0] + side_length[0], n_spaces)
        y = np.linspace(poi[1] - side_length[1], poi[1] + side_length[1], n_spaces)
        z = np.linspace(poi[2] - side_length[2], poi[2] + side_length[2], n_spaces)
        xyz_grid = np.meshgrid(x, y, z, indexing="ij")
        xyz_grid = np.stack(xyz_grid, -1)

        output = []
        for i in range(xyz_grid.shape[0]):
            for j in range(xyz_grid.shape[1]):
                for k in range(xyz_grid.shape[2]):
                    loc = xyz_grid[i, j, k]

                    if self.is_manipulator_out_of_bound(loc):
                        continue

                    sdf_dists = self.compute_sdf_dists(prim_idx, loc)

                    if sdf_dists.min() > self.cfg.min_dist:
                        output.append((loc, sdf_dists))

        return output

    @staticmethod
    def calc_placement_score(sdf_dists, F):
        return ((1 / sdf_dists) * F).mean().item()

    def calc_potentials(self, x, y):
        with torch.no_grad():
            F, G = self.loss.calculate_potentials(x, y)
        return F, G

    def is_manipulator_out_of_bound(self, point):
        return np.any(point < self.lower_bound) or np.any(point > self.upper_bound)

    def sort_object_ids(self, object_ids, F):
        values = np.zeros((len(object_ids),))
        for i, id in enumerate(object_ids):
            values[i] = F[self.particle_obj_id == id].mean()
        inds = values.argsort()[::-1]
        object_ids = [object_ids[i] for i in inds]
        return object_ids

    def find_best_placement(self, neighborhood, F):
        # best_point = None
        best_score = -np.inf

        best_point = []
        for (point, sdf_dists) in neighborhood:
            # push to gpu
            sdf_dists = torch.from_numpy(sdf_dists).to(F.device)
            score = self.calc_placement_score(sdf_dists, F)

            if score > best_score:
                best_score = score
                best_point = [point]
            elif score == best_score:
                best_point.append(point)
        best_point = np.stack(best_point, axis=0)
        best_point = best_point.mean(0)

        return best_point, best_score

    def sample_stage_plans(self, visualize=False, debug=False):
        y = self.loss.goal_obs
        x = torch.from_numpy(self.sim.get_x(0)).to(y.device)

        g = self.loss.calculate_transport_gradient(x, y)
        F, G = self.calc_potentials(x, y)

        plans = self._sample_stage_plans(x, g, F, G)

        output = (plans,)
        if visualize:
            from CPDeform.utils.visualize import draw_potentials_on_pts, draw_transport_gradient
            pcd = draw_potentials_on_pts(x, F)
            obj_inds = np.arange(0, self.particle_obj_id.shape[0])
            inds = self.sort_by_potentials(F, topk=50, obj_inds=obj_inds)
            arws = draw_transport_gradient(x, -1 * g, inds)
            geometry = tuple([pcd, *arws])
            output += (geometry,)

        if debug:
            output += (x, g, F, G,)

        return output

    def _sample_stage_plans(self, x, g, F, G):

        object_ids = list(set(self.particle_obj_id))

        if len(object_ids) > 1:
            # if multiple objects are present, we should focus on object with higher transport priority
            object_ids = self.sort_object_ids(object_ids, F)

        plans = []
        for manipulator_idx in range(self.n_manipulators):
            self.curr_manipulator_idx = manipulator_idx
            self.curr_obj_idx = object_ids.pop(0)
            self.update_obj_particle_bool_map(self.curr_obj_idx)

            obj_x, obj_g = x[self.obj_particle_bool_map], g[self.obj_particle_bool_map]
            obj_F, obj_G = F[self.obj_particle_bool_map], G[self.obj_particle_bool_map]
            obj_particle_inds = np.arange(0, self.particle_obj_id.shape[0])[self.obj_particle_bool_map]

            manipulator_plans = self._sample_object_level_plans(obj_x, obj_g, obj_F, obj_G, obj_particle_inds)
            plans.append(manipulator_plans)
        plans = list(itertools.product(*plans))

        return plans

    def _sample_object_level_plans(self, x, g, F, G, particle_inds):

        sorted_inds = self.sort_by_potentials(F, self.cfg.topk, particle_inds)
        best_idx = sorted_inds[0]

        nn_inds = find_knn(x, F, best_idx, 500, 200)
        poi = x[torch.LongTensor(nn_inds)].mean(0).cpu().numpy()

        self.add_explored_idx(particle_inds[best_idx])

        if self.cfg.topn_potentials > 0:
            below_threshold = F.argsort(descending=True)[self.cfg.topn_potentials:]
            F = F.clone()
            F[below_threshold] = 0.0

        print("point of interest", poi)
        plans, scores = self.sample_single_stage_plans(poi, x, g, F)

        # sort plans in descending score for faster training with early stopper
        sorted_inds = np.array(scores).argsort()[::-1]
        plans = [plans[i] for i in sorted_inds]

        # recursive call if the plan is empty
        if len(plans) == 0:
            plans = self._sample_object_level_plans(x, g, F, G, particle_inds)

        return plans

    def sample_single_stage_plans(self, poi, x, g, F):
        raise NotImplementedError

    def setup_single_plan(self, plan):
        raise NotImplementedError

    @classmethod
    def default_config(cls):
        cfg = CN()

        # primitive placement
        cfg.topk = 10
        cfg.min_dist = 0.015
        cfg.neighborhood_size = 0.1
        cfg.n_prims_per_gripper = 1
        cfg.topn_potentials = 0
        cfg.max_dist_traveled = 1.0
        cfg.pose_rots = []
        cfg.pose_dirs = []

        # out of bound check
        cfg.lower_bound = (0., 0., 0.)
        cfg.upper_bound = (1., 1., 1.)

        return cfg


########################################
# Fixed Start Baseline (PlasticineLab) #
########################################

class FixedStartBaseline(PrimitiveSampler):
    def __init__(self, taichi_env: TaichiEnv, loss: OptimalTransportLoss, **kwargs):
        super(FixedStartBaseline, self).__init__(taichi_env, loss, **kwargs)

    def sample_single_stage_plans(self, poi, x, g, F):
        return [None], [1]

    def setup_single_plan(self, grippers_plan):
        pass

    def place_primitiveB(self, direction, prim_idxB, pointA, x, g, F):
        pass

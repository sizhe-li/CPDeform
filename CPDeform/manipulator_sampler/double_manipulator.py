import numpy as np
import torch
from torch_geometric.nn import knn

from CPDeform.loss import OptimalTransportLoss
from CPDeform.manipulator_sampler.base import PrimitiveSampler, length
from plb.engine.taichi_env import TaichiEnv


class Clamp(PrimitiveSampler):
    def __init__(self, taichi_env: TaichiEnv, loss: OptimalTransportLoss, **kwargs):
        super(Clamp, self).__init__(taichi_env, loss, **kwargs)

    def sample_single_stage_plans(self, poi, x, g, F):

        prim_inds = self.get_curr_prim_inds()
        assert len(prim_inds) == 2  # must be clamping

        plans = []
        scores = []
        for pose_idx, (rot, direction) in enumerate(zip(self.cfg.pose_rots, self.cfg.pose_dirs)):
            print(f"planning for pose {pose_idx}...")

            rot, direction = np.array(rot), np.array(direction)

            for prim_idx in prim_inds:
                self.env.primitives.primitives[prim_idx].set_rotation(rot)

            best_pointA, best_scoreA = self.place_primitiveA(direction, prim_inds[0], poi, x, g, F)
            if best_pointA is None:
                print("best_pointA is None!")
                continue

            best_pointB, best_scoreB = self.place_primitiveB(direction, prim_inds[1], best_pointA, x, g, F)
            if best_pointB is None:
                print("best_pointB is None!")
                continue

            plan = (best_pointA, best_pointB, rot)
            score = best_scoreA + best_scoreB

            plans.append(plan)
            scores.append(score)

        return plans, scores

    def place_primitiveA(self, direction, prim_idxA, poi, x, g, F):
        raise NotImplementedError


    def place_primitiveB(self, direction, prim_idxB, pointA, x, g, F):
        raise NotImplementedError

    def setup_single_plan(self, grippers_plan):
        for gripper_idx, gripper_plan in enumerate(grippers_plan):
            pointA, pointB, rot = gripper_plan
            assert pointA.shape == pointB.shape == (3,)
            assert rot.shape == (4,)

            prim_inds = self.get_prim_inds(gripper_idx)
            for i, point in enumerate((pointA, pointB)):
                self.sim.primitives.primitives[prim_inds[i]].set_position(point)
                self.sim.primitives.primitives[prim_inds[i]].set_rotation(rot)



class ClampReshape(Clamp):
    def __init__(self, taichi_env: TaichiEnv, loss: OptimalTransportLoss, **kwargs):
        super(ClampReshape, self).__init__(taichi_env, loss, **kwargs)

    def place_primitiveA(self, direction, prim_idxA, poi, x, g, F):
        assert direction.sum() != 0

        step_size = self.cfg.min_dist / 4
        best_pointA, best_scoreA = None, -np.inf

        signs = (-1., 1.)
        for sign in signs:
            pointA = poi.copy()
            accepted = False
            sdf_distsA = None
            dist_traveled = 0.0
            while not accepted:
                displacement = direction * step_size * sign
                pointA += displacement

                if self.is_manipulator_out_of_bound(pointA):
                    break

                sdf_distsA = self.compute_sdf_dists(prim_idxA, pointA)
                min_sdf_distB = sdf_distsA.min()
                accepted = min_sdf_distB > self.cfg.min_dist
                dist_traveled += length(displacement)
                if dist_traveled >= self.cfg.max_dist_traveled:
                    break

            if accepted:
                sdf_distsA = torch.from_numpy(sdf_distsA).to(F.device)
                scoreA = self.calc_placement_score(sdf_distsA, F)
                if scoreA > best_scoreA:
                    best_scoreA = scoreA
                    best_pointA = pointA

        return best_pointA, best_scoreA

    def place_primitiveB(self, direction, prim_idxB, pointA, x, g, F):

        assert direction.sum() != 0

        step_size = self.cfg.min_dist / 4
        best_pointB, best_scoreB = None, -np.inf

        signs = (-1., 1.)
        for sign in signs:
            pointB = pointA.copy()
            accepted = False
            sdf_distsB = None
            dist_traveled = 0.0
            passed_object = False
            while not accepted:
                displacement = direction * step_size * sign
                pointB += displacement

                if self.is_manipulator_out_of_bound(pointB):
                    break

                sdf_distsB = self.compute_sdf_dists(prim_idxB, pointB)
                min_sdf_distB = sdf_distsB.min()
                passed_object = passed_object or (min_sdf_distB < 0)
                accepted = (min_sdf_distB > self.cfg.min_dist) and passed_object
                dist_traveled += length(displacement)
                if dist_traveled >= self.cfg.max_dist_traveled:
                    # considered infeasible, quitting
                    break

            if accepted:
                sdf_distsB = torch.from_numpy(sdf_distsB).to(F.device)
                scoreB = self.calc_placement_score(sdf_distsB, F)
                if scoreB > best_scoreB:
                    best_scoreB = scoreB
                    best_pointB = pointB

        return best_pointB, best_scoreB


class ClampTransport(Clamp):
    def __init__(self, taichi_env: TaichiEnv, loss: OptimalTransportLoss, **kwargs):
        super(ClampTransport, self).__init__(taichi_env, loss, **kwargs)

    def place_primitiveA(self, direction, prim_idxA, poi, x, g, F):
        neighborhoodA = self.gen_neighborhood(poi, prim_idxA)
        if len(neighborhoodA) == 0:
            return None, 0.0

        best_pointA, best_scoreA = self.find_best_placement(neighborhoodA, F)

        return best_pointA, best_scoreA

    def place_primitiveB(self, direction, prim_idxB, pointA, x, g, F):
        with torch.no_grad():
            pointA_th = torch.from_numpy(pointA).to(x.device).unsqueeze(0)  # 1 x 3
            inds = knn(x, pointA_th, k=50)[1]
            x = x[F[inds].argsort(descending=True)][:10].mean(0)
            direction = x.cpu().numpy() - pointA
            direction[1] = 0.0  # make sure the clamp is vertically stable
            direction = direction / np.linalg.norm(direction)

        step_size = self.cfg.neighborhood_size * 0.05
        pointB = pointA.copy()

        accepted = False
        sdf_distsB = None
        dist_traveled = 0.0
        passed_object = False
        while not accepted:
            displacement = direction * step_size
            pointB += displacement

            if self.is_manipulator_out_of_bound(pointB):
                break

            sdf_distsB = self.compute_sdf_dists(prim_idxB, pointB)
            min_sdf_distB = sdf_distsB.min()
            passed_object = passed_object or (min_sdf_distB < 0)
            accepted = (min_sdf_distB > self.cfg.min_dist) and passed_object
            dist_traveled += length(displacement)
            if dist_traveled >= self.cfg.max_dist_traveled:
                # considered infeasible, quitting
                break

        best_pointB = None
        best_scoreB = -np.inf
        if accepted:
            best_pointB = pointB
            sdf_distsB = torch.from_numpy(sdf_distsB).to(F.device)
            best_scoreB = self.calc_placement_score(sdf_distsB, F)
        return best_pointB, best_scoreB

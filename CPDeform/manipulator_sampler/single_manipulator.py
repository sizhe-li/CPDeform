from CPDeform.loss import OptimalTransportLoss
from CPDeform.manipulator_sampler.base import PrimitiveSampler
from plb.engine.taichi_env import TaichiEnv


class CollideReshape(PrimitiveSampler):
    def __init__(self, taichi_env: TaichiEnv, loss: OptimalTransportLoss, **kwargs):
        super(CollideReshape, self).__init__(taichi_env, loss, **kwargs)

    def sample_single_stage_plans(self, poi, x, g, F):
        """
            F is still on gpu
        """

        # we only have one manipulator
        prim_inds = self.get_curr_prim_inds()
        assert len(prim_inds) == 1
        prim_idx = prim_inds[0]

        neighborhood = self.gen_neighborhood(poi, prim_idx)
        best_point, best_score = self.find_best_placement(neighborhood, F)

        plans, scores = [best_point], [best_score]
        return plans, scores

    def setup_single_plan(self, grippers_plan):
        assert len(grippers_plan) == 1
        plan = grippers_plan[0]
        self.sim.primitives.primitives[0].set_position(plan)

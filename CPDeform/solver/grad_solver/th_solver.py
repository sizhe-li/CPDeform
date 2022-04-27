import tqdm
import torch
import numpy as np

from CPDeform.utils.base_logger import StepLogger
from plb.engine.grad_model import GradModel
from CPDeform.utils import training_utils as ut
from CPDeform.utils import MetricLogger
from CPDeform.loss import OptimalTransportLoss

ut.set_default_tensor_type(torch.DoubleTensor)


class TorchSolver:
    def __init__(self, env, loss_fn: OptimalTransportLoss, ouput_grid=(), **kwargs):
        self.env = env
        self.loss_fn = loss_fn
        self.func = GradModel(env, output_grid=ouput_grid, **kwargs)
        self.set_start_loss()

    def set_start_loss(self):
        with torch.no_grad():
            init_state = self.env.get_state()
            observations = self.func.reset(init_state['state'])
            loss = self.loss_fn.calculate_ot_loss(self.loss_fn.last_index, observations[0])
            self.loss_fn.set_start_loss(loss['particle_loss'].item())
            self.env.set_state(**init_state)

    def solve(self, initial_actions, logger: StepLogger,
              lr=0.01, n_iters=200,
              header=None,
              early_stopper=None,
              verbose=True):
        self.loss_fn.set_horizon(len(initial_actions))

        # loss is a function of the observer ..
        best_action, best_loss, best_incremental_iou = None, np.inf, None
        if initial_actions is not None:
            self._buffer = []
            self.action = action = torch.nn.Parameter(ut.np2th(np.array(initial_actions)), requires_grad=True)
            self.optim = optim = torch.optim.Adam([action], lr=lr)
            self.initial_state = initial_state = self.env.get_state()
            iter_id = 0
        else:
            initial_state = self.initial_state
            optim = self.optim
            action = self.action
            for act in self._buffer:
                if act['loss'] < best_loss:
                    best_loss = act['loss']
                    best_action = act['action']
                    best_incremental_iou = act['incremental_iou']
            iter_id = len(self._buffer)

        ran = tqdm.trange if verbose else range
        it = ran(iter_id, iter_id + n_iters)

        for iter_id in it:

            optim.zero_grad()

            metric_logger = MetricLogger(delimiter='  ')
            loss, loss_dict = 0, None
            observations = self.func.reset(initial_state['state'])
            for idx, act in enumerate(action):
                observations = self.func.forward(idx, act, *observations)
                loss_dict = self.loss_fn(idx, *observations)
                if logger is not None: logger.step()

                for k, v in loss_dict.items():
                    loss += v
                    metric_logger.update(**{k: v.item()})

            loss.backward()
            optim.step()
            action.data.clip_(-1, 1)

            loss_info = self.loss_fn.compute_loss_info(idx, loss_dict)
            if logger is not None: logger.record(loss_info)

            with torch.no_grad():
                loss = loss_info['loss']
                incremental_iou = loss_info['incremental_iou']
                last_act = action.data.detach().cpu().numpy()

                if loss < best_loss:
                    best_loss = loss
                    best_action = last_act
                    best_incremental_iou = incremental_iou

            self._buffer.append({'action': last_act, 'loss': loss, 'incremental_iou': incremental_iou})
            if verbose:
                iter_header = iter_id if header is None else header
                it.set_description(f"{iter_header} | heuristic_loss: {loss:.06f} | {str(metric_logger)}", refresh=True)

            if early_stopper is not None:
                stopping, improved = early_stopper(best_loss, iter_id)
                if stopping:
                    break

        self.env.set_state(**initial_state)

        return {
            'best_loss': best_loss,
            'best_action': best_action,
            'best_incremental_iou': best_incremental_iou,
            'last_loss': loss,
            'last_action': last_act,
            'num_env_steps': (iter_id + 1) * len(action)
        }

    def eval(self, action, render_fn):
        self.env.simulator.cur = 0
        initial_state = self.initial_state
        self.env.set_state(**initial_state)
        outs = []
        import tqdm
        for i in tqdm.tqdm(action, total=len(action)):
            self.env.step(i)
            outs.append(render_fn())

        self.env.set_state(**initial_state)
        return outs

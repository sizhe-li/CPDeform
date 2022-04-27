import os
import cv2
import time
import shutil
import pickle
import numpy as np

from yacs.config import CfgNode as CN
from plb.config.utils import make_cls_config
from CPDeform.utils.visualize import animate, draw_o3d_geometry


class StageLogger:
    def __init__(self, exp_dir=None, cfg=None, **kwargs):
        self.cfg = cfg = make_cls_config(self, cfg, **kwargs)

        self.use_open3d = cfg.use_open3d
        self._stage_cnt = 0
        self._attempt_cnt = 0
        self.start_time = time.time()

        self.keys = ['stage_cnt', 'attempt_cnt', 'loss', 'incremental_iou', 'seconds_elapsed']

        self.exp_dir = exp_dir
        if exp_dir is not None:
            self.log_dir = log_dir = os.path.join(exp_dir, f'stage_attempt_log')
            self.vis_dir = vis_dir = os.path.join(exp_dir, f'stage_attempt_vis')

            for directory in (log_dir, vis_dir):
                if os.path.exists(directory):
                    shutil.rmtree(directory)
                os.makedirs(directory)

            with open(self.filepath(), 'w') as f:
                f.write(','.join(self.keys) + '\n')

        self.cam_positions = cfg.cam_positions
        self.cam_rotations = cfg.cam_rotations

    def filepath(self):
        return os.path.join(self.exp_dir, 'stage_overall_log.txt')

    @property
    def stage_cnt(self):
        return self._stage_cnt

    @property
    def attempt_cnt(self):
        return self._attempt_cnt

    def write(self, values):
        with open(self.filepath(), 'a') as f:
            f.write(','.join(str(values[i]) for i in self.keys) + '\n')

    def reset(self):
        self._stage_cnt = 0
        self._attempt_cnt = 0

    def increment_stage_cnt(self):
        self._stage_cnt += 1
        self._attempt_cnt = 0

    def get_step_logger(self, plan_cnt, init_step=0):
        # we need to add one to the attempt_cnt since it is currently executing
        name = [f"stage_{self.stage_cnt:03d}", f"attempt_{self.attempt_cnt + 1:03d}", f"plan_{plan_cnt:03d}"]
        name = "-".join(name)
        step_logger = StepLogger(self.log_dir, name, self.start_time, init_step)
        return step_logger

    def step(self, loss, incremental_iou):
        self._attempt_cnt += 1

        values = dict()
        values['stage_cnt'] = self._stage_cnt
        values['attempt_cnt'] = self._attempt_cnt
        values['loss'] = loss
        values['incremental_iou'] = incremental_iou
        values['seconds_elapsed'] = time.time() - self.start_time

        self.write(values=values)

    def check_if_loggable(self):
        assert self.exp_dir is not None, 'exp_dir is not defined!'

    def log_best_trajectory(self, best_actions, best_plans):
        assert self.exp_dir is not None
        traj = dict()

        traj['best_actions'] = np.concatenate(best_actions, 0)
        traj['best_plans'] = best_plans

        filename = os.path.join(self.exp_dir, f'best_traj.pkl')
        pickle.dump(traj, open(filename, "wb"))

    def log_images_as_video(self, images, name=None):
        self.check_if_loggable()
        if name is None:
            filename = os.path.join(self.vis_dir, f'step_{self._stage_cnt:06d}.webm')
        else:
            filename = os.path.join(self.vis_dir, f'{name}.webm')

        animate(images, filename, fps=10, _return=False)

    def save_img(self, img, filename):
        filename = os.path.join(self.vis_dir, filename)
        cv2.imwrite(filename, img)

    def log_state_as_image(self, taichi_env, filename=None, primitive=1):
        self.check_if_loggable()
        if filename is None:
            filename = os.path.join(self.vis_dir, f'state_step_{self._stage_cnt:06d}.png')
        else:
            filename = os.path.join(self.vis_dir, filename)

        img = self.render_state_multiview(taichi_env, primitive)
        img = img[:, :, ::-1]
        cv2.imwrite(filename, img)

    def log_geometry_as_image(self, geometry, filename=None):
        self.check_if_loggable()
        if not self.use_open3d:
            return

        if filename is None:
            filename = os.path.join(self.vis_dir, f'geometry_step_{self._stage_cnt:06d}.png')
        else:
            filename = os.path.join(self.vis_dir, filename)

        img = self.render_geometry_multiview(geometry)
        img = img[:, :, ::-1]
        cv2.imwrite(filename, img)

    def render_state_multiview(self, taichi_env, primitive=1):
        images = []
        for (camera_pos, camera_rot) in zip(self.cam_positions, self.cam_rotations):
            taichi_env.renderer.update_camera_matrix(camera_pos, camera_rot)
            img = taichi_env.render(mode='array', render_mode='rgb', primitive=primitive)
            images.append(img)
        img = np.concatenate(images, axis=1)
        return img

    def render_geometry_multiview(self, geometry):
        images = []
        for (camera_pos, camera_rot) in zip(self.cam_positions, self.cam_rotations):
            images.append(draw_o3d_geometry(geometry, camera_pos, camera_rot))
        img = np.concatenate(images, axis=1)
        return img

    @classmethod
    def default_config(cls):
        cfg = CN()

        cfg.use_open3d = True
        cfg.cam_positions = [(0.5, 2.5, 2.), (0.5, 1.2, 4.)]
        cfg.cam_rotations = [(1.0, 0.), (0.2, 0.)]
        cfg.w = 512
        cfg.h = 512
        return cfg


class StepLogger:
    def __init__(self, exp_dir, name, start_time=None, init_step=0):
        self.name = name
        self.exp_dir = exp_dir
        self.keys = ['step_cnt', 'incremental_iou', 'loss', 'seconds_elapsed']
        if start_time is None:
            start_time = time.time()
        self.start_time = start_time
        self.init_step = init_step

        with open(self.filepath(), 'w') as f:
            f.write(','.join(self.keys) + '\n')

        self.step_cnt = 0
        self.start = None

    def filepath(self):
        return os.path.join(self.exp_dir, self.name + '.txt')

    def write(self, values):
        with open(self.filepath(), 'a') as f:
            f.write(','.join(str(values[i]) for i in self.keys) + '\n')

    def step(self):
        self.step_cnt += 1

    def record(self, info):
        values = dict()
        values['step_cnt'] = self.init_step + self.step_cnt
        values['incremental_iou'] = info['incremental_iou']
        values['loss'] = info['particle_loss']
        values['seconds_elapsed'] = time.time() - self.start_time

        self.write(values=values)

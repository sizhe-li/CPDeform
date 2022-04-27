import os
import numpy as np
import open3d as o3d

COLORS = [
    (127 << 16) + 127,
    (127 << 8),
    127,
    127 << 16,
]

PATH = os.path.dirname(os.path.abspath(__file__))




class Shapes:
    # make shapes from the configuration
    def __init__(self, cfg):
        self.object_ids = []
        self.object_particles = []
        self.object_colors = []

        self.dim = 3

        state = np.random.get_state()
        np.random.seed(0)  # fix seed 0
        for i in cfg:
            kwargs = dict()
            for key, val in i.items():
                if isinstance(val, str) and key not in ['shape', 'pcd_path', 'mesh_path']:
                    val = eval(val)
                kwargs[key] = val
            print(kwargs)
            if i['shape'] == 'box':
                self.add_box(**kwargs)
            elif i['shape'] == 'sphere':
                self.add_sphere(**kwargs)
            elif i['shape'] == 'torus':
                self.add_torus(**kwargs)
            elif i['shape'] == 'ply':
                self.add_ply_object(**kwargs)
            elif i['shape'] == 'mesh':
                self.add_mesh_object(**kwargs)
            else:
                raise NotImplementedError(f"Shape {i['shape']} is not supported!")
        np.random.set_state(state)

    def get_n_particles(self, volume):
        return max(int(volume / 0.2 ** 3) * 10000, 1)

    def add_object(self, particles, color=None, init_rot=None, object_idx=None, **extras):
        # 1. add object id
        if object_idx is None:
            object_idx = len(self.object_particles)
        self.object_ids.append(np.array([object_idx] * len(particles)))

        # 2. add object particles
        if init_rot is not None:
            import transforms3d
            q = transforms3d.quaternions.quat2mat(init_rot)
            origin = particles.mean(axis=0)
            particles = (particles[:, :self.dim] - origin) @ q.T + origin
        self.object_particles.append(particles[:, :self.dim])
        # 3. add object color
        if color is None or isinstance(color, int):
            tmp = COLORS[len(self.object_particles) - 1] if color is None else color
            color = np.zeros(len(particles), np.int32)
            color[:] = tmp
        self.object_colors.append(color)

    def add_box(self, init_pos, width, n_particles=10000, color=None, init_rot=None, object_idx=None, **extras):
        # pass
        if isinstance(width, float):
            width = np.array([width] * self.dim)
        else:
            width = np.array(width)
        if n_particles is None:
            n_particles = self.get_n_particles(np.prod(width))
        p = (np.random.random((n_particles, self.dim)) * 2 - 1) * (0.5 * width) + np.array(init_pos)
        self.add_object(p, color, init_rot=init_rot, object_idx=object_idx)

    def add_sphere(self, init_pos, radius, n_particles=10000, color=None, init_rot=None, object_idx=None, **extras):
        if n_particles is None:
            if self.dim == 3:
                volume = (radius ** 3) * 4 * np.pi / 3
            else:
                volume = (radius ** 2) * np.pi
            n_particles = self.get_n_particles(volume)

        p = np.random.normal(size=(n_particles, self.dim))
        p /= np.linalg.norm(p, axis=-1, keepdims=True)
        u = np.random.random(size=(n_particles, 1)) ** (1. / self.dim)
        p = p * u * radius + np.array(init_pos)[:self.dim]
        self.add_object(p, color, init_rot=init_rot, object_idx=object_idx)

    def add_torus(self, init_pos, tx, ty, n_particles=10000, color=None, init_rot=None, object_idx=None, **extras):

        def length(x):
            return np.sqrt(np.einsum('ij,ij->i', x, x) + 1e-14)

        if n_particles is None:
            raise NotImplementedError

        p = np.ones((n_particles, 3)) * 5

        remain_cnt = n_particles  # how many left to sample
        while remain_cnt > 0:
            x = np.random.random((remain_cnt,)) * (2 * ty + 2 * tx) - (ty + tx)
            y = np.random.random((remain_cnt,)) * (4 * ty) - (2 * ty)
            z = np.random.random((remain_cnt,)) * (2 * ty + 2 * tx) - (ty + tx)

            vec1 = np.stack([x, z], axis=-1)
            len1 = length(vec1) - tx
            vec2 = np.stack([len1, y], axis=-1)
            len2 = length(vec2) - ty

            accept_map = len2 <= 0
            accept_cnt = sum(accept_map)
            start = n_particles - remain_cnt
            p[start:start + accept_cnt] = np.stack([x, y, z], axis=-1)[accept_map]

            remain_cnt -= accept_cnt

        assert np.all(p != 5)
        p = p + np.array(init_pos)[:self.dim]
        self.add_object(p, color, init_rot=init_rot, object_idx=object_idx)

    def add_ply_object(self, pcd_path, color=None, init_rot=None, object_idx=None, **extras):
        pcd_path = os.path.join(PATH, '../../', pcd_path)
        pcd = o3d.io.read_point_cloud(pcd_path)
        bbox_max = np.array(pcd.points).max(0)
        bbox_min = np.array(pcd.points).min(0)
        c = bbox_max + bbox_min
        tmp_c = np.array([0.5, 0.5, 0.5])
        trans = tmp_c - c
        pcd = pcd.translate(trans)
        pcd.scale(0.2, center=tmp_c)
        # translate to bottom surface
        _, y_max, _ = np.array(pcd.points).max(0)
        _, y_min, _ = np.array(pcd.points).min(0)
        trans = np.array([0.0, -y_min, 0.0])
        pcd.translate(trans)

        p = np.array(pcd.points)
        self.add_object(p, color, init_rot=init_rot, object_idx=object_idx)

    def get(self):
        assert len(self.object_particles) > 0, "please add at least one shape into the scene"
        return (np.concatenate(self.object_particles), np.concatenate(self.object_colors),
                np.concatenate(self.object_ids))

    def remove_object(self, index):
        self.object_particles.pop(index)
        self.object_colors.pop(index)
        self.object_ids.pop(index)

    def clear_objects(self):
        for i in range(len(self.object_particles)):
            self.remove_object(i)

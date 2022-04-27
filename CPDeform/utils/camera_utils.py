import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch


def get_int(image_res=(512, 512)):
    fov = 0.23
    int = np.array([
        -np.array([2 * fov / image_res[1], 0, -fov - 1e-5, ]),
        -np.array([0, 2 * fov / image_res[1], -fov - 1e-5, ]),
        [0, 0, 1]
    ])
    return np.linalg.inv(int)


def get_ext(camera_rot, trans):
    T = np.zeros((4, 4))
    T[:3, :3] = np.array([
        [np.cos(camera_rot[1]), 0.0000000, np.sin(camera_rot[1])],
        [0.0000000, 1.0000000, 0.0000000],
        [-np.sin(camera_rot[1]), 0.0000000, np.cos(camera_rot[1])],
    ]) @ np.array([
        [1.0000000, 0.0000000, 0.0000000],
        [0.0000000, np.cos(camera_rot[0]), np.sin(camera_rot[0])],
        [0.0000000, -np.sin(camera_rot[0]), np.cos(camera_rot[0])],
    ])
    T[:3, 3] = np.array(trans)
    T[3, 3] = 1

    T = np.linalg.inv(T)
    return np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]) @ T


def visualize_pcd(pcd):
    pcd_np = pcd.numpy()[0]
    # filter zeros
    bool_map = (pcd_np[:, 2] > 0)
    pcd_np = pcd_np[bool_map]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_np)
    # Flip it, otherwise the pointcloud will be upside down
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    o3d.visualization.draw_geometries([pcd])


def plt_voxels(voxs, elev=5., azim=1.):
    voxs = voxs.cpu().numpy()[0, 0]
    voxs = voxs > 0

    # set the colors of each object
    colors = np.empty(voxs.shape, dtype=object)
    colors[voxs] = 'red'

    # and plot everything
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(voxs, edgecolor='k')
    ax.view_init(elev=elev, azim=azim)

    plt.show()


def animate(imgs, filename='animation.webm', _return=True, fps=1):
    print(f'animating {filename}')
    from moviepy.editor import ImageSequenceClip
    imgs = ImageSequenceClip(imgs, fps=fps)
    imgs.write_videofile(filename, fps=fps)
    if _return:
        from IPython.display import Video
        return Video(filename, embed=True)


def apply_4x4(RT, xyz):
    B, N, _ = list(xyz.shape)
    ones = torch.ones_like(xyz[:, :, 0:1])
    xyz1 = torch.cat([xyz, ones], 2)
    xyz1_t = torch.transpose(xyz1, 1, 2)
    # this is B x 4 x N
    xyz2_t = torch.matmul(RT, xyz1_t)
    xyz2 = torch.transpose(xyz2_t, 1, 2)
    xyz2 = xyz2[:, :, :3]
    return xyz2

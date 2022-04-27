import torch
import numpy as np
import open3d as o3d

from CPDeform.utils.camera_utils import get_int, get_ext


def get_cross_prod_mat(pVec_Arr):
    # pVec_Arr shape (3)
    qCross_prod_mat = np.array([
        [0, -pVec_Arr[2], pVec_Arr[1]],
        [pVec_Arr[2], 0, -pVec_Arr[0]],
        [-pVec_Arr[1], pVec_Arr[0], 0],
    ])
    return qCross_prod_mat


def caculate_align_mat(pVec_Arr):
    scale = np.linalg.norm(pVec_Arr)
    pVec_Arr = pVec_Arr / scale
    # must ensure pVec_Arr is also a unit vec.
    z_unit_Arr = np.array([0, 0, 1])
    z_mat = get_cross_prod_mat(z_unit_Arr)

    z_c_vec = np.matmul(z_mat, pVec_Arr)
    z_c_vec_mat = get_cross_prod_mat(z_c_vec)

    if np.dot(z_unit_Arr, pVec_Arr) == -1:
        qTrans_Mat = -np.eye(3, 3)
    elif np.dot(z_unit_Arr, pVec_Arr) == 1:
        qTrans_Mat = np.eye(3, 3)
    else:
        qTrans_Mat = np.eye(3, 3) + z_c_vec_mat + np.matmul(z_c_vec_mat,
                                                            z_c_vec_mat) / (1 + np.dot(z_unit_Arr, pVec_Arr))

    qTrans_Mat *= scale
    return qTrans_Mat


def draw_vector_open3d(begin, end, scale=1.0):
    """
        begin = [a, b, c]
        end = [a, b, c]
    """
    vec_Arr = np.array(end) - np.array(begin)
    vec_len = np.linalg.norm(vec_Arr)

    mesh_arrow = o3d.geometry.TriangleMesh.create_arrow(
        cone_height=0.2 * vec_len * scale,
        cone_radius=0.06 * vec_len * scale,
        cylinder_height=0.8 * vec_len * scale,
        cylinder_radius=0.04 * vec_len * scale
    )

    mesh_arrow.paint_uniform_color([1, 0, 1])
    mesh_arrow.compute_vertex_normals()

    # mesh_sphere_begin = o3d.geometry.TriangleMesh.create_sphere(radius=0.1, resolution=20)
    # mesh_sphere_begin.translate(begin)
    # mesh_sphere_begin.paint_uniform_color([0, 1, 1])
    # mesh_sphere_begin.compute_vertex_normals()
    #
    # mesh_sphere_end = o3d.geometry.TriangleMesh.create_sphere(radius=0.1, resolution=20)
    # mesh_sphere_end.translate(end)
    # mesh_sphere_end.paint_uniform_color([0, 1, 1])
    # mesh_sphere_end.compute_vertex_normals()

    rot_mat = caculate_align_mat(vec_Arr)
    mesh_arrow.rotate(rot_mat, center=(0, 0, 0))
    mesh_arrow.translate(np.array(begin))  # 0.5*(np.array(end) - np.array(begin))

    return mesh_arrow


def animate(imgs, filename='animation.webm', _return=True, fps=1):
    print(f'animating {filename}')
    from moviepy.editor import ImageSequenceClip
    imgs = ImageSequenceClip(imgs, fps=fps)
    imgs.write_videofile(filename, fps=fps)
    if _return:
        from IPython.display import Video
        return Video(filename, embed=True)


def get_open3d_cam_params(camera_pos, camera_rot, w, h):
    intrinsic = get_int((h, w))
    extrinsic = get_ext(camera_rot, camera_pos)
    fx = fy = intrinsic[0, 0]
    cx = cy = intrinsic[0, 2]
    intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)
    extrinsic = get_ext(camera_rot, camera_pos)

    o3d_cam = o3d.camera.PinholeCameraParameters()
    o3d_cam.intrinsic = intrinsic
    o3d_cam.extrinsic = extrinsic
    return o3d_cam


def draw_o3d_geometry(geometry, camera_pos, camera_rot, w=512, h=512):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=w, height=h, visible=False)
    if isinstance(geometry, tuple):
        for geom in geometry:
            vis.add_geometry(geom)
    else:
        vis.add_geometry(geometry)
    ctr = vis.get_view_control()
    o3d_cam = get_open3d_cam_params(camera_pos, camera_rot, w, h)
    ctr.convert_from_pinhole_camera_parameters(o3d_cam, allow_arbitrary=True)
    vis.update_renderer()
    image = vis.capture_screen_float_buffer(do_render=True)
    vis.destroy_window()
    image = np.uint8(np.asarray(image) * 255)

    return image


def draw_potentials_on_pts(points, potentials):
    if torch.is_tensor(points):
        points = points.cpu().numpy()
    if torch.is_tensor(potentials):
        potentials = potentials.cpu().numpy()

    pcd = points_to_pcd(points)
    pcd.colors = o3d.utility.Vector3dVector(rgb_np(potentials.min(), potentials.max(), potentials))
    return pcd


def draw_transport_gradient(x, g, inds, dist=0.2):
    inds_list = inds

    arrows = []
    for idx in inds_list:
        gradient = (g[idx] / torch.linalg.norm(g[idx]))
        pt_start = x[idx]
        pt_end = pt_start + dist * gradient
        arrows.append(draw_vector_open3d(pt_start.cpu().numpy(), pt_end.cpu().numpy()))

    return tuple(arrows)


def points_to_pcd(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def rgb_np(minimum, maximum, value):
    minimum, maximum = float(minimum), float(maximum)
    ratio = 2 * (value - minimum) / (maximum - minimum)
    b = (255 * (1 - ratio)).clip(min=0)
    r = (255 * (ratio - 1)).clip(min=0)
    g = 255 - b - r

    return np.stack([r, g, b], axis=-1) / 255

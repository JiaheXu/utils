import cv2
import numpy as np
import open3d as o3d
import numpy as np
from numpy.linalg import inv
from scipy.spatial.transform import Rotation
import copy
from .math_utils import *
def project_point( point_3d, color, image, extrinsic, intrinsic, radius = 5):
                
    point4d = np.append(point_3d, 1)
    new_point4d = np.matmul(extrinsic, point4d)
    point3d = new_point4d[:-1]
    zc = point3d[2]
    new_point3d = np.matmul(intrinsic, point3d)
    new_point3d = new_point3d/new_point3d[2]
    u = int(round(new_point3d[0]))
    v = int(round(new_point3d[1]))
    if(v<0 or v>= image.shape[0] or u<0 or u>= image.shape[1]):
        return image
    
    image[max(0, v-radius): min(v+radius, image.shape[0]), max(0, u-radius): min(u+radius, image.shape[1]) ] = color
    # print("updated")
    return image

def cropping(rgb, xyz, bound_box, label = None, return_image = True):

    x = xyz[:,:,0]
    y = xyz[:,:,1]
    z = xyz[:,:,2]

    # print("bound_box: ", bound_box)
    valid_idx = np.where( (x>=bound_box[0][0]) & (x <=bound_box[0][1]) & (y>=bound_box[1][0]) & (y<=bound_box[1][1]) & (z>=bound_box[2][0]) & (z<=bound_box[2][1]) )

    if(return_image):

        cropped_rgb = np.zeros(rgb.shape)
        cropped_xyz = np.zeros(xyz.shape) 
        cropped_rgb[valid_idx] = rgb[valid_idx]
        cropped_xyz[valid_idx] = xyz[valid_idx]

        return cropped_rgb, cropped_xyz

    valid_xyz = xyz[valid_idx]
    valid_rgb = rgb[valid_idx]
    valid_label = None
    if(label is not None):
        valid_label = label[valid_idx]
            
    valid_pcd = o3d.geometry.PointCloud()
    valid_pcd.points = o3d.utility.Vector3dVector( valid_xyz)
    if(np.max(valid_rgb) > 1.0):
        valid_pcd.colors = o3d.utility.Vector3dVector( valid_rgb/255.0 )
    else:
        valid_pcd.colors = o3d.utility.Vector3dVector( valid_rgb )
    return valid_xyz, valid_rgb, valid_label, valid_pcd

def xyz_rgb_validation(rgb, xyz):
    # verify xyz and depth value
    valid_pcd = o3d.geometry.PointCloud()
    xyz = xyz.reshape(-1,3)
    rgb = (rgb/255.0).reshape(-1,3)
    valid_pcd.points = o3d.utility.Vector3dVector( xyz )
    valid_pcd.colors = o3d.utility.Vector3dVector( rgb )
    # visualize_pcd(valid_pcd)

def visualize_pcd_transform(pcd, left = None, right = None):
    coor_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    coor_frame.scale(0.1, center=(0., 0., 0.))
    vis.add_geometry(coor_frame)
    vis.get_render_option().background_color = np.asarray([255, 255, 255])

    view_ctl = vis.get_view_control()

    vis.add_geometry(pcd)

    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    mesh.scale(0.1, center=(0., 0., 0.) )
    
    # left_mesh.scale(0.1, center=(left[0][3], left[1][3], left[2][3]))
    # right_mesh.scale(0.1, center=(right[0][3], right[1][3], right[2][3]))
    
    if left is not None:
        for trans in left:
            left_mesh = copy.deepcopy(mesh).transform(trans)
            vis.add_geometry(left_mesh)

    if right is not None:
        for trans in right:
            right_mesh = copy.deepcopy(mesh).transform(trans)
            vis.add_geometry(right_mesh)
    
    # view_ctl.set_up([-0.4, 0.0, 1.0])
    # view_ctl.set_front([-4.02516493e-01, 3.62146675e-01, 8.40731978e-01])
    # view_ctl.set_lookat([0.0 ,0.0 ,0.0])
    
    view_ctl.set_up((1, 0, 0))  # set the positive direction of the x-axis as the up direction
    # view_ctl.set_up((0, -1, 0))  # set the negative direction of the y-axis as the up direction
    view_ctl.set_front((-0.3, 0.0, 0.2))  # set the positive direction of the x-axis toward you
    view_ctl.set_lookat((0.0, 0.0, 0.3))  # set the original point as the center point of the window
    vis.run()
    vis.destroy_window()
    
def visualize_pcd(pcd, traj_lists = None, curr_pose = None, drawlines = False):

    coor_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    coor_frame.scale(0.05, center=(0., 0., 0.))
    vis.add_geometry(coor_frame)
    vis.get_render_option().background_color = np.asarray([255, 255, 255])

    view_ctl = vis.get_view_control()

    vis.add_geometry(pcd)

    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    mesh.scale(0.1, center=(0., 0., 0.) )
    # if(use_arrow):
    #     mesh = o3d.geometry.TriangleMesh.create_arrow( cylinder_radius=0.01, cone_radius=0.01, cylinder_height=0.005, cone_height=0.01, resolution=20, cylinder_split=4, cone_split=1 )
    # print("curr_pose: ", curr_pose)
    if(traj_lists is not None):

        for traj_idx ,traj in enumerate( traj_lists, 0 ):
            points = [ [0,0,0] ]
            lines = []
            colors = []
            if(curr_pose is not None):
                points.append( curr_pose[traj_idx][0:3,3] )
                lines.append( [ len(points) - 1 , len(points) - 2])
                colors.append( [1,0,0] )
            for node_idx, point in enumerate( traj , 0 ):
                new_mesh = copy.deepcopy(mesh).transform(point)
                vis.add_geometry(new_mesh)
                if drawlines:
                    points.append(point[0:3,3])
                    lines.append( [ len(points) - 1 , len(points) - 2])
                    colors.append( [1,0,0] )


            
            if drawlines:
                # lines.append( [ 0, len(points)])
                # colors.append( [1,0,0] )
                # print("points: ", len(points))
                # print("lines: ", lines)
                line_set = o3d.geometry.LineSet(
                    points=o3d.utility.Vector3dVector(points),
                    lines=o3d.utility.Vector2iVector(lines),
                )
                line_set.colors = o3d.utility.Vector3dVector(colors)
                vis.add_geometry(line_set)
                # o3d.visualization.draw_geometries([line_set])

    if(curr_pose is not None):
        for pose in curr_pose:
            curr_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
            curr_mesh.scale(0.2, center=(0., 0., 0.) )
            curr_mesh = curr_mesh.transform(pose)
            vis.add_geometry(curr_mesh)

    view_ctl.set_up((1, 0, 0))  # set the positive direction of the x-axis as the up direction
    view_ctl.set_front((-0.3, 0.0, 0.2))  # set the positive direction of the x-axis toward you
    view_ctl.set_lookat((0.0, 0.0, 0.3))  # set the original point as the center point of the window
    vis.run()
    vis.destroy_window()
def visualize_pcd_delta_transform(pcd, start_ts = [], delta_transforms=[]):

    coor_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    coor_frame.scale(0.2, center=(0., 0., 0.) )
    vis.add_geometry(coor_frame)
    vis.get_render_option().background_color = np.asarray([255, 255, 255])

    view_ctl = vis.get_view_control()

    vis.add_geometry(pcd)

    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    mesh.scale(0.1, center=(0., 0., 0.))

    new_mesh = copy.deepcopy(mesh).transform( get_transform(start_t[0:3], start_t[3:7]) )
    vis.add_geometry(new_mesh)
    for start_t, delta_transform in zip( start_ts, delta_transforms):
        init_transform = get_transform( start_t )
        for delta_t in delta_transform:
            last_trans = get_transform( delta_t ) @ init_transform
            new_mesh = copy.deepcopy(mesh).transform(last_trans)
            vis.add_geometry(new_mesh)

    view_ctl.set_up((1, 0, 0))  # set the positive direction of the x-axis as the up direction
    view_ctl.set_front((-0.3, 0.0, 0.2))  # set the positive direction of the x-axis toward you
    view_ctl.set_lookat((0.0, 0.0, 0.3))  # set the original point as the center point of the window
    vis.run()
    vis.destroy_window()

def visualize_pcds(pcds):

    coor_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    coor_frame.scale(0.2, center=(0., 0., 0.) )
    vis.add_geometry(coor_frame)
    vis.get_render_option().background_color = np.asarray([255, 255, 255])

    view_ctl = vis.get_view_control()

    

    # mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    # mesh.scale(0.1, center=(0., 0., 0.))

    # new_mesh = copy.deepcopy(mesh).transform( get_transform(start_t[0:3], start_t[3:7]) )
    # vis.add_geometry(new_mesh)
    for pcd in pcds:
        vis.add_geometry(pcd)

    view_ctl.set_up((1, 0, 0))  # set the positive direction of the x-axis as the up direction
    view_ctl.set_front((-0.3, 0.0, 0.2))  # set the positive direction of the x-axis toward you
    view_ctl.set_lookat((0.0, 0.0, 0.3))  # set the original point as the center point of the window
    vis.run()
    vis.destroy_window()

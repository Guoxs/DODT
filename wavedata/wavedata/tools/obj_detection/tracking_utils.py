import os

import numpy as np

from wavedata.tools.core import calib_utils
from wavedata.tools.obj_detection import obj_utils


class TrackingLabel(obj_utils.ObjectLabel):
    """tracking Label Class
    1    frame_id     frame id
    1    object_id    id for object in frame
    1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                      'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                      'Misc' or 'DontCare'

    1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                      truncated refers to the object leaving image boundaries

    1    occluded     Integer (0,1,2,3) indicating occlusion state:
                      0 = fully visible, 1 = partly occluded
                      2 = largely occluded, 3 = unknown

    1    alpha        Observation angle of object, ranging [-pi..pi]

    4    bbox         2D bounding box of object in the image (0-based index):
                      contains left, top, right, bottom pixel coordinates

    3    dimensions   3D object dimensions: height, width, length (in meters)

    3    location     3D object location x,y,z in camera coordinates (in meters)

    1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]

    1    score        Only for results: Float, indicating confidence in
                      detection, needed for p/r curves, higher is better.
    """

    def __init__(self):
        super(TrackingLabel, self).__init__()
        self.frame_id = ""
        self.object_id = ""


def read_labels(label_dir, name, results=False):
    """Reads in label data file from Kitti tracking Dataset.

    Returns:
    obj_list -- List of instances of class TrackingLabel.

    Keyword arguments:
    label_dir -- directory of the label files
    img_idx -- index of the image, including the video info and frame info
    """

    # Define the object list
    obj_list = []

    # Extract the list
    assert len(name) == 6, print('Sample name incorrect!')
    video_id = int(name[:2])
    frame_id = int(name[2:])
    if os.stat(label_dir + "/%04d.txt" % video_id).st_size == 0:
        return

    if results:
        p = np.loadtxt(label_dir + "/%04d.txt" % video_id, delimiter=' ',
                       dtype=str,
                       usecols=np.arange(start=0, step=1, stop=18))
    else:
        p = np.loadtxt(label_dir + "/%04d.txt" % video_id, delimiter=' ',
                       dtype=str,
                       usecols=np.arange(start=0, step=1, stop=17))

    frame_idx = (p[:,0] == str(frame_id))
    p = p[frame_idx]

    label_num = p.shape[0]

    for idx in np.arange(label_num):
        obj = TrackingLabel()
        # Fill in the object list
        obj.frame_id = int(p[idx, 0])
        obj.object_id = int(p[idx, 1])
        obj.type = p[idx, 2]
        obj.truncation = float(p[idx, 3])
        obj.occlusion = float(p[idx, 4])
        obj.alpha = float(p[idx, 5])
        obj.x1 = float(p[idx, 6])
        obj.y1 = float(p[idx, 7])
        obj.x2 = float(p[idx, 8])
        obj.y2 = float(p[idx, 9])
        obj.h = float(p[idx, 10])
        obj.w = float(p[idx, 11])
        obj.l = float(p[idx, 12])
        obj.t = (float(p[idx, 13]), float(p[idx, 14]), float(p[idx, 15]))
        obj.ry = float(p[idx, 16])
        if results:
            obj.score = float(p[idx, 17])
        else:
            obj.score = 0.0

        obj_list.append(obj)

    return obj_list


def get_raw_lidar_point_cloud(name, velo_dir):
    assert len(name) == 6, print('Sample name incorrect!')
    video_id = int(name[:2])
    frame_id = int(name[2:])
    velo_dir = velo_dir + '/' + str(video_id).zfill(4)
    x, y, z, i = calib_utils.read_lidar(velo_dir=velo_dir, img_idx=frame_id)
    pts = np.vstack((x,y,z,i))
    return pts

def get_lidar_in_camera_view(pts, name, calib_dir, im_size=None, min_intensity=None):
    assert len(name) == 6, print('Sample name incorrect!')
    video_id = int(name[:2])
    # Read calibration info
    frame_calib = calib_utils.read_tracking_calibration(calib_dir, video_id)
    i = pts[3]
    pts = pts[:3]
    pts = calib_utils.lidar_to_cam_frame(pts.T, frame_calib)
    # The given image is assumed to be a 2D image
    if not im_size:
        point_cloud = pts.T
        return point_cloud

    else:
        # Only keep points in front of camera (positive z)
        pts = pts[pts[:, 2] > 0]
        point_cloud = pts.T

        # Project to image frame
        point_in_im = calib_utils.project_to_image(point_cloud, p=frame_calib.p2).T

        # Filter based on the given image size
        image_filter = (point_in_im[:, 0] > 0) & \
                       (point_in_im[:, 0] < im_size[0]) & \
                       (point_in_im[:, 1] > 0) & \
                       (point_in_im[:, 1] < im_size[1])

    if not min_intensity:
        return pts[image_filter].T

    else:
        intensity_filter = i > min_intensity
        point_filter = np.logical_and(image_filter, intensity_filter)
        return pts[point_filter].T


def get_lidar_point_cloud(name, calib_dir, velo_dir,
                          im_size=None, min_intensity=None):
    """ Calculates the lidar point cloud, and optionally returns only the
    points that are projected to the image.

    :param name: frame name
    :param calib_dir: directory with calibration files
    :param velo_dir: directory with velodyne files
    :param im_size: (optional) 2 x 1 list containing the size of the image
                      to filter the point cloud [w, h]
    :param min_intensity: (optional) minimum intensity required to keep a point

    :return: (3, N) point_cloud in the form [[x,...][y,...][z,...]]
    """
    assert len(name) == 6, print('Sample name incorrect!')
    video_id = int(name[:2])
    frame_id = int(name[2:])
    # Read calibration info
    frame_calib = calib_utils.read_tracking_calibration(calib_dir, video_id)
    velo_dir = velo_dir + '/' + str(video_id).zfill(4)
    x, y, z, i = calib_utils.read_lidar(velo_dir=velo_dir, img_idx=frame_id)

    # Calculate the point cloud
    pts = np.vstack((x, y, z)).T
    pts = calib_utils.lidar_to_cam_frame(pts, frame_calib)

    # The given image is assumed to be a 2D image
    if not im_size:
        point_cloud = pts.T
        return point_cloud

    else:
        # Only keep points in front of camera (positive z)
        pts = pts[pts[:, 2] > 0]
        point_cloud = pts.T

        # Project to image frame
        point_in_im = calib_utils.project_to_image(point_cloud, p=frame_calib.p2).T

        # Filter based on the given image size
        image_filter = (point_in_im[:, 0] > 0) & \
                       (point_in_im[:, 0] < im_size[0]) & \
                       (point_in_im[:, 1] > 0) & \
                       (point_in_im[:, 1] < im_size[1])

    if not min_intensity:
        return pts[image_filter].T

    else:
        intensity_filter = i > min_intensity
        point_filter = np.logical_and(image_filter, intensity_filter)
        return pts[point_filter].T


def get_road_plane(name, planes_dir):
    """Reads the road plane from file

    :param int img_idx : Index of image
    :param str planes_dir : directory containing plane text files

    :return plane : List containing plane equation coefficients
    """
    assert len(name) == 6, print('Sample name incorrect!')
    video_id = int(name[:2])
    frame_id = int(name[2:])
    plane_file = planes_dir + '/%04d/%06d.txt' % (video_id,frame_id)

    with open(plane_file, 'r') as input_file:
        lines = input_file.readlines()
        input_file.close()

    # Plane coefficients stored in 4th row
    lines = lines[3].split()

    # Convert str to float
    lines = [float(i) for i in lines]

    plane = np.asarray(lines)

    #######################################
    # fixed the plane for tracking datasets
    #######################################
    # TODO calculate the plane files for the tracking datasets
    plane = np.asarray([0,-1,0,1.65])

    # Ensure normal is always facing up.
    # In Kitti's frame of reference, +y is down
    if plane[1] > 0:
        plane = -plane

    # Normalize the plane coefficients
    norm = np.linalg.norm(plane[0:3])
    plane = plane / norm

    return plane


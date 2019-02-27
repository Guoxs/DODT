import numpy as np
import os

from wavedata.tools.obj_detection import tracking_utils
from wavedata.tools.core.voxel_grid_2d import VoxelGrid2D

from avod.datasets.kitti.kitti_utils import KittiUtils


class KittiTrackingUtils(KittiUtils):

    def __init__(self, dataset):
        super(KittiTrackingUtils, self).__init__(dataset)


    def get_point_cloud(self, source, name, image_shape=None):
        """ Gets the points from the point cloud for a particular image,
            keeping only the points within the area extents, and takes a slice
            between self._ground_filter_offset and self._offset_distance above
            the ground plane

        Args:
            source: point cloud source, e.g. 'lidar'
            name: An String , e.g. '000123' or '000500'
            image_shape: image dimensions (h, w), only required when
                source is 'lidar' or 'depth'

        Returns:
            The set of points in the shape (N, 3)
        """

        if source == 'lidar':
            # wavedata wants im_size in (w, h) order
            im_size = [image_shape[1], image_shape[0]]

            point_cloud = tracking_utils.get_lidar_point_cloud(
                name, self.dataset.calib_dir, self.dataset.velo_dir,
                im_size=im_size)

        else:
            raise ValueError("Invalid source {}".format(source))

        return point_cloud

    def get_ground_plane(self, sample_name):
        """Reads the ground plane for the sample

        Args:
            sample_name: name of the sample, e.g. '000123'

        Returns:
            ground_plane: ground plane coefficients
        """
        ground_plane = tracking_utils.get_road_plane(int(sample_name),
                                                self.dataset.planes_dir)
        return ground_plane

    def create_sliced_voxel_grid_2d(self, sample_name, source,
                                    image_shape=None):
        """Generates a filtered 2D voxel grid from point cloud data

        Args:
            sample_name: image name to generate stereo pointcloud from
            source: point cloud source, e.g. 'lidar'
            image_shape: image dimensions [h, w], only required when
                source is 'lidar' or 'depth'

        Returns:
            voxel_grid_2d: 3d voxel grid from the given image
        """
        ground_plane = tracking_utils.get_road_plane(sample_name,
                                                self.dataset.planes_dir)

        point_cloud = self.get_point_cloud(source, sample_name,
                                           image_shape=image_shape)

        filtered_points = self._apply_slice_filter(point_cloud, ground_plane)

        # Create Voxel Grid
        voxel_grid_2d = VoxelGrid2D()
        voxel_grid_2d.voxelize_2d(filtered_points, self.voxel_size,
                                  extents=self.area_extents,
                                  ground_plane=ground_plane,
                                  create_leaf_layout=True)


class Oxts(object):
    '''
    GPS/IMU information, written for each synchronized frame, each text file contains 30 values
    '''

    def __init__(self, oxts_lines):
        data = oxts_lines.split()
        self.latitude   = float(data[0])       # latitude of the oxts-unit (deg)
        self.longitude  = float(data[1])       # longitude of the oxts-unit (deg)
        self.altitude   = float(data[2])       # altitude of the oxts-unit (m)
        self.roll       = float(data[3])       # roll angle (rad),  0 = level, positive = left side up (-pi..pi)
        self.pitch      = float(data[4])       # pitch angle (rad), 0 = level, positive = front down (-pi/2..pi/2)
        self.yaw        = float(data[5])       # heading (rad),     0 = east,  positive = counter clockwise (-pi..pi)


    def rotx(self, t):
        ''' 3D Rotation about the x-axis. '''
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[1, 0, 0],
                         [0, c, -s],
                         [0, s, c]])

    def roty(self, t):
        ''' Rotation about the y-axis. '''
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c, 0, s],
                         [0, 1, 0],
                         [-s, 0, c]])

    def rotz(self, t):
        ''' Rotation about the z-axis. '''
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c, -s, 0],
                         [s, c, 0],
                         [0, 0, 1]])

    def distance(self, object):
        '''
        calculate the diatance of two point using (latitude, longitude)

        L = 2R * arcsin(sqrt(sin^2((lat1-lat2)/2) + cos(lon1) * cos(lon2) * sin^2((lon1-lon2)/2)))
        '''
        def rad(deg):
            '''Convert degree to rad'''
            return deg * np.pi / 180.00

        lat1, lon1 = rad(self.latitude), rad(self.longitude)
        lat2, lon2 = rad(object.latitude), rad(object.longitude)
        R = 6378137.0   # radius of earth (m)
        a = lat1 - lat2
        b = lon1 - lon2
        dis = 2 * R * np.arcsin(
                            np.sqrt(np.power(np.sin(a/2), 2) +
                            np.cos(lat1) * np.cos(lat2) * np.power(np.sin(b/2),2))
                    )
        return dis

    def displacement(self, object):
        d = self.distance(object)
        delta_yaw = self.yaw - object.yaw
        delta_pitch = self.pitch - object.pitch
        delta_z = d * np.cos(delta_yaw)
        delta_x = d * np.sin(delta_yaw)
        delta_y = d * np.sin(delta_pitch)
        return np.array([delta_x, delta_y, delta_z])

    def get_rotate_matrix(self, object, axis='y'):
        if axis == 'x':
            delta_pitch = self.pitch - object.pitch
            return self.rotx(delta_pitch)
        if axis == 'z':
            delta_roll = self.roll - object.roll
            return self.rotz(delta_roll)
        elif axis == 'y':
            delta_yaw = self.yaw - object.yaw
            return self.roty(delta_yaw)

    def get_delta(self, object, theta='yaw'):
        if theta == 'yaw':
            return self.yaw - object.yaw
        if theta == 'roll':
            return self.roll - object.roll
        if theta == 'pitch':
            return self.pitch - object.pitch
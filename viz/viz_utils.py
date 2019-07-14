''' Visualization code for point clouds and 3D bounding boxes with mayavi.

Modified by Charles R. Qi
Date: September 2017

Ref: https://github.com/hengck23/didi-udacity-2017/blob/master/baseline-04/kitti_data/draw.py
'''
import cv2
import numpy as np
import mayavi.mlab as mlab


def draw_projected_box3d(image, qs, color=(255, 255, 255), thickness=2):
    ''' Draw 3d bounding box in image
        qs: (8,3) array of vertices for the 3d box in following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
    '''
    qs = qs.astype(np.int32)
    for k in range(0, 4):
        # Ref: http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
        i, j = k, (k + 1) % 4
        # use LINE_AA for opencv3
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness, cv2.LINE_AA)

        i, j = k + 4, (k + 1) % 4 + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness, cv2.LINE_AA)

        i, j = k, k + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness, cv2.LINE_AA)
    return image


def draw_lidar_simple(pc, fig=None, color=None):
    ''' Draw lidar points. simplest set up. '''
    if fig is None:
        fig = mlab.figure(figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1600, 1000))
    if color is None: color = pc[:, 2]
    # draw points
    mlab.points3d(pc[:, 0], pc[:, 2], pc[:, 1], color, color=None, mode='point', colormap='gnuplot', scale_factor=1,
                  figure=fig)
    # draw origin
    mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='sphere', scale_factor=0.2)
    # draw axis
    axes = np.array([
        [2., 0., 0., 0.],
        [0., 2., 0., 0.],
        [0., 0., 2., 0.],
    ], dtype=np.float64)
    mlab.plot3d([0, axes[0, 0]], [0, axes[0, 1]], [0, axes[0, 2]], color=(1, 0, 0), tube_radius=None, figure=fig)
    mlab.plot3d([0, axes[1, 0]], [0, axes[1, 1]], [0, axes[1, 2]], color=(0, 1, 0), tube_radius=None, figure=fig)
    mlab.plot3d([0, axes[2, 0]], [0, axes[2, 1]], [0, axes[2, 2]], color=(0, 0, 1), tube_radius=None, figure=fig)
    # mlab.view(azimuth=180, elevation=70, focalpoint=[12.0909996, -1.04700089, -2.03249991], distance=62.0, figure=fig)
    # mlab.show()
    return fig


def draw_lidar(pc, color=None, fig=None, bgcolor=(0, 0, 0), pts_scale=1, pts_mode='point', pts_color=None,
               draw_fov=True, draw_rigion=True):
    ''' Draw lidar points
    Args:
        pc: numpy array (n,3) of XYZ
        color: numpy array (n) of intensity or whatever
        fig: mayavi figure handler, if None create new one otherwise will use it
    Returns:
        fig: created or used fig
    '''
    if fig is None: fig = mlab.figure(figure=None, bgcolor=bgcolor, fgcolor=None, engine=None, size=(1600, 1000))
    if color is None: color = pc[:, 2]
    mlab.points3d(pc[:, 0], pc[:, 2], pc[:, 1], color, color=pts_color, mode=pts_mode, colormap='gnuplot',
                  scale_factor=pts_scale, figure=fig)

    # draw origin
    mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='sphere', scale_factor=0.2)

    # draw axis
    axes = np.array([
        [2., 0., 0., 0.],
        [0., 2., 0., 0.],
        [0., 0., 2., 0.],
    ], dtype=np.float64)
    mlab.plot3d([0, axes[0, 0]], [0, axes[0, 1]], [0, axes[0, 2]], color=(1, 0, 0), tube_radius=None, figure=fig)
    mlab.plot3d([0, axes[1, 0]], [0, axes[1, 1]], [0, axes[1, 2]], color=(0, 1, 0), tube_radius=None, figure=fig)
    mlab.plot3d([0, axes[2, 0]], [0, axes[2, 1]], [0, axes[2, 2]], color=(0, 0, 1), tube_radius=None, figure=fig)

    # draw fov (todo: update to real sensor spec.)
    if draw_fov:
        fov = np.array([  # 45 degree
            [40., 40., 0., 0.],
            [40., -40., 0., 0.],
        ], dtype=np.float64)

        mlab.plot3d([0, fov[0, 0]], [0, fov[0, 1]], [0, fov[0, 2]], color=(1, 1, 1), tube_radius=None, line_width=1,
                    figure=fig)
        mlab.plot3d([0, fov[1, 0]], [0, fov[1, 1]], [0, fov[1, 2]], color=(1, 1, 1), tube_radius=None, line_width=1,
                    figure=fig)

    # draw square region
    if draw_rigion:
        TOP_Y_MIN = -20
        TOP_Y_MAX = 20
        TOP_X_MIN = 0
        TOP_X_MAX = 40
        TOP_Z_MIN = -2.0
        TOP_Z_MAX = 0.4

        x1 = TOP_X_MIN
        x2 = TOP_X_MAX
        y1 = TOP_Y_MIN
        y2 = TOP_Y_MAX
        mlab.plot3d([x1, x1], [y1, y2], [0, 0], color=(0.5, 0.5, 0.5), tube_radius=0.1, line_width=1, figure=fig)
        mlab.plot3d([x2, x2], [y1, y2], [0, 0], color=(0.5, 0.5, 0.5), tube_radius=0.1, line_width=1, figure=fig)
        mlab.plot3d([x1, x2], [y1, y1], [0, 0], color=(0.5, 0.5, 0.5), tube_radius=0.1, line_width=1, figure=fig)
        mlab.plot3d([x1, x2], [y2, y2], [0, 0], color=(0.5, 0.5, 0.5), tube_radius=0.1, line_width=1, figure=fig)

    # mlab.orientation_axes()
    mlab.view(azimuth=180, elevation=70, focalpoint=[12.0909996, -1.04700089, -2.03249991], distance=62.0, figure=fig)
    return fig


def draw_gt_boxes3d(gt_boxes3d, fig, box_id=0, color=(1, 1, 1), line_width=1, draw_text=True, text_scale=(1, 1, 1),
                    color_list=None, text_color=None, text_type='name'):
    ''' Draw 3D bounding boxes
    Args:
        gt_boxes3d: numpy array (n,8,3) for XYZs of the box corners
        fig: mayavi figure handler
        color: RGB value tuple in range (0,1), box line color
        line_width: box line width
        draw_text: boolean, if true, write box indices beside boxes
        text_scale: three number tuple
        color_list: a list of RGB tuple, if not None, overwrite color.
        text_color: RGB value tuple in range (0,1), score text color
    Returns:
        fig: updated fig
    '''
    num = len(gt_boxes3d)
    for n in range(num):
        b = gt_boxes3d[n]

        name = box_id
        if str(type(name)) != "<class 'int'>":
            name = box_id[n]

        if color_list is not None:
            color = color_list[n]
        txt_color = color if text_color is None else text_color
        if draw_text:
            if text_type == 'name':
                mlab.text3d(b[4, 0], b[4, 2], b[4, 1], '%d' % name, scale=text_scale, color=txt_color, figure=fig)
            if text_type == 'scores':
                mlab.text3d(b[4, 0], b[4, 2], b[4, 1], '%.4f' % name, scale=text_scale, color=txt_color, figure=fig)
        for k in range(0, 4):
            # http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
            i, j = k, (k + 1) % 4
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 2], b[j, 2]], [b[i, 1], b[j, 1]],
                        color=color, tube_radius=None, line_width=line_width, figure=fig)

            i, j = k + 4, (k + 1) % 4 + 4
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 2], b[j, 2]], [b[i, 1], b[j, 1]],
                        color=color, tube_radius=None, line_width=line_width, figure=fig)

            i, j = k, k + 4
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 2], b[j, 2]], [b[i, 1], b[j, 1]],
                        color=color, tube_radius=None, line_width=line_width, figure=fig)
    # mlab.show(1)
    # mlab.view(azimuth=180, elevation=70, focalpoint=[ 12.0909996 , -1.04700089, -2.03249991], distance=62.0, figure=fig)
    return fig


def draw_bounding_box(box3d, fig, color=(1, 1, 1), line_width=1):
    x_min,x_max, y_min,y_max, z_min,z_max = box3d
    vertices = [[x_min,z_max,y_max],[x_min,z_min,y_max],[x_max,z_min,y_max],[x_max,z_max,y_max],
                [x_min,z_max,y_min],[x_min,z_min,y_min],[x_max,z_min,y_min],[x_max,z_max,y_min]]
    vertices = np.asarray(vertices)
    for k in range(0, 4):
        i, j = k, (k + 1) % 4
        mlab.plot3d([vertices[i, 0], vertices[j, 0]], [vertices[i, 1], vertices[j, 1]], [vertices[i, 2], vertices[j, 2]], color=color, tube_radius=None,
                    line_width=line_width, figure=fig)

        i, j = k + 4, (k + 1) % 4 + 4
        mlab.plot3d([vertices[i, 0], vertices[j, 0]], [vertices[i, 1], vertices[j, 1]], [vertices[i, 2], vertices[j, 2]], color=color, tube_radius=None,
                    line_width=line_width, figure=fig)

        i, j = k, k + 4
        mlab.plot3d([vertices[i, 0], vertices[j, 0]], [vertices[i, 1], vertices[j, 1]], [vertices[i, 2], vertices[j, 2]], color=color, tube_radius=None,
                    line_width=line_width, figure=fig)

    mlab.show(1)
    mlab.view(azimuth=180, elevation=70, focalpoint=[ 12.0909996 , -1.04700089, -2.03249991], distance=62.0, figure=fig)
    return fig

if __name__ == '__main__':
    pc = np.loadtxt('test_pc.txt')
    fig = draw_lidar_simple(pc)
    mlab.show()
    input()

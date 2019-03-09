import numpy as np
from viz.viz_utils import draw_lidar_simple, draw_gt_boxes3d
from wavedata.tools.obj_detection.obj_utils import compute_box_corners_3d

def draw_lidar_and_boxes(lidar, gt_boxes, calib, fig):
    lidar_draw = lidar.T
    fig = draw_lidar_simple(lidar_draw, fig=fig)
    length = len(gt_boxes)
    corners_3ds = np.zeros((length, 8, 3), dtype=np.float32)
    box_id = []
    for i in range(length):
        gt_box = gt_boxes[i]
        boxes3d = compute_box_corners_3d(gt_box).T
        corners_3ds[i] = calib.project_rect_to_velo(boxes3d)
        box_id.append(gt_box.object_id)

    fig = draw_gt_boxes3d(corners_3ds, fig, box_id=box_id)
    return fig

def draw_lidar_and_boxes_in_camera_view(lidar, gt_boxes, calib, fig):
    lidar_draw = lidar.T
    fig = draw_lidar_simple(lidar_draw, fig=fig)
    length = len(gt_boxes)
    corners_3ds = np.zeros((length, 8, 3), dtype=np.float32)
    box_id = []
    for i in range(length):
        gt_box = gt_boxes[i]
        boxes3d = compute_box_corners_3d(gt_box).T
        corners_3ds[i] = boxes3d
        box_id.append(gt_box.object_id)

    fig = draw_gt_boxes3d(corners_3ds, fig, box_id=box_id)
    return fig


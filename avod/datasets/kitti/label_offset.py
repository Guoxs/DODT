import numpy as np
from copy import deepcopy

O = [0, 0]
A = [40, 40]
B = [40, 70]
C = [-40, 70]
D = [-40, 40]
E = [-40, 0]
F = [40, 0]


class dotPair():
    def __init__(self, sPoint, ePoint):
        self.sPoint = sPoint
        self.ePoint = ePoint


class line():
    def __init__(self, sPoint, theta):
        self.sPoint = sPoint
        self.theta = theta
        self.ang = [np.cos(self.theta), np.sin(self.theta)]
        self.c = -self.ang[1] * self.sPoint[0] + self.ang[0] * self.sPoint[1]
        self.abc = [self.ang[1], -self.ang[0], self.c]


def ifIntersect(line, seg):
    sFlag = np.sum(np.array(line.abc) * np.array([seg.sPoint[0], seg.sPoint[1], 1]))
    eFlag = np.sum(np.array(line.abc) * np.array([seg.ePoint[0], seg.ePoint[1], 1]))
    if sFlag * eFlag > 0:
        return False
    else:
        return True


def getCross(line, seg):
    abLine = [line.abc[0], line.abc[1]]
    abDot = [seg.ePoint[1] - seg.sPoint[1], seg.sPoint[0] - seg.ePoint[0]]
    c = np.array([-line.abc[2], seg.ePoint[1] * seg.sPoint[0] - seg.sPoint[1] * seg.ePoint[0]])
    # print(abLine, abDot, c)
    cross = np.linalg.solve(np.array([abLine, abDot]), c)
    # print(cross)
    if (cross[0] - line.sPoint[0]) * line.ang[0] >= 0 and \
            (cross[1] - line.sPoint[1]) * line.ang[1] >= 0:
        if (cross[0] >= -40 and cross[0] <= 40) and \
                (cross[1] >= 0 and cross[1] <= 70):
            return cross
    else:
        return None


def getOffsets(label):
    center = [label[0], label[2]]
    theta = label[-2]
    obj_id = label[-1]
    # [delta_x, delta_z, delta_ry=0, obj_id]
    offsets = [0, 0, 0, obj_id]

    intersect_point = [0, [0, 0]]
    poly = [dotPair(B, C), dotPair(C, E),
            dotPair(E, F), dotPair(F, B)]
    l1 = line(center, theta)
    sflag = False
    for i in range(len(poly)):
        if ifIntersect(l1, poly[i]):
            cross = getCross(l1, poly[i])
            if cross is not None:
                sflag = True
                intersect_point[0] = i
                intersect_point[1] = cross

    # no intersect with border
    if not sflag:
        return offsets

    x0, y0 = center[0], center[1]
    x1, y1 = intersect_point[1][0], intersect_point[1][1]
    w, l = label[4], label[3]
    d = 0
    if intersect_point[0] in [0, 2]:
        if y1 - y0 == 0:
            d = l / 2
        else:
            # d = w|x1-x0|/(2|y1-y0|)
            d = w * abs(x1 - x0) / (2 * abs(y1 - y0)) + l / 2
    elif intersect_point[0] in [1, 3]:
        if x1 - x0 == 0:
            d = l / 2
        else:
            # d = w|y1-y0|/(2|x1-x0|)
            d = w * abs(y1 - y0) / (2 * abs(x1 - x0)) + l / 2
    if d > l:
        d = l

    offsets[0] = (x1 - x0) + d * np.cos(theta)
    offsets[1] = (y1 - y0) + d * np.sin(theta)
    return np.asarray(offsets)


def getOffsets2(label):
    theta = label[-2]
    obj_id = label[-1]
    # [delta_x, delta_z, delta_ry=0, obj_id]
    offsets = [0, 0, 0, obj_id]
    # move one bounding box in ry direction
    offsets[0] = label[3] * np.cos(theta)
    offsets[1] = label[3] * np.sin(theta)
    return np.asarray(offsets)


def cal_label_birth(labels):
    offsets = []
    for i in range(len(labels)):
        label = deepcopy(labels[i])
        label[-2] = -label[-2]
        offset = getOffsets2(label)
        offsets.append(offset)
    return offsets


def cal_label_death(labels):
    offsets = []
    for i in range(len(labels)):
        label = deepcopy(labels[i])
        offset = getOffsets2(label)
        offsets.append(offset)
    return offsets


def cal_label_offsets(label_1, label_2):
    '''calculate (delta_x, delta_z, delta_ry) between two adjacent key frames.
        label_1: pre label  [[x,y,z,l,w,h,ry, obj_id],[...],...]
        label_2: next label [[x,y,z,l,w,h,ry, obj_id],[...],...]
    '''
    offsets = []
    if len(label_1) > 0:
        if len(label_2) > 0:
            # idx
            pre_label_idx = [i for i in range(len(label_1))]
            next_label_idx = [i for i in range(len(label_2))]

            for i in range(len(label_1)):
                pre_off = label_1[i]
                obj_id = pre_off[-1]
                for j in range(len(label_2)):
                    next_off = label_2[j]
                    # object match
                    if int(obj_id) == int(next_off[-1]):
                        offset = next_off - pre_off
                        offset[-1] = obj_id
                        # only need (delta_x, delta_z, delta_ry, obj_id)
                        offset = offset[[0,2,6,7]]
                        offsets.append(offset)
                        # remove match idx
                        pre_label_idx.remove(i)
                        next_label_idx.remove(j)
                        continue
            if len(pre_label_idx) > 0:
                # object death
                death_offsets = cal_label_death(label_1[pre_label_idx])
                offsets += death_offsets
            if len(next_label_idx) > 0:
                # object birth
                birth_offsets = cal_label_birth(label_2[next_label_idx])
                offsets += birth_offsets
        else:
            # objects death, assume objects death just before next frame
            death_offsets = cal_label_death(label_1)
            offsets += death_offsets
    else:
        if len(label_2) > 0:
            # objects birth, assume objects birth just after pre frame
            birth_offsets = cal_label_birth(label_2)
            offsets += birth_offsets
    return np.asarray(offsets)
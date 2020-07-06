import numpy as np
import shapely
from shapely.geometry import LineString, Point, Polygon
import matplotlib.pyplot as plt
import math
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
        # 射线
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
    # 两条线求交，联立直线方程，返回（x，y）
   # print('solve: ',end = '')
    #print(cross)


    # print('hhhhhhhh',end = ' ')
    # print((cross[0] - seg.sPoint[0]) * line.abc[0],end = ' ')
    # print((cross[1] - seg.sPoint[1]) * line.abc[1])

    #if (cross[0] - line.sPoint[0]) * line.ang[0] >= 0 and \
     #       (cross[1] - line.sPoint[1]) * line.ang[1] >= 0:

    if (cross[0] - seg.sPoint[0]) * line.abc[0] >= 0 and \
            (cross[1] - seg.sPoint[1]) * line.abc[1] >= 0:
        if (cross[0] >= -40 and cross[0] <= 40) and \
                (cross[1] >= 0 and cross[1] <= 70):
            return cross
        else:
            return None
    else:
        return None


def getOffsets2(label):
    center = [label[0], label[2]]
    theta = label[-2]
    obj_id = label[-1]
    # [delta_x, delta_z, delta_ry=0, obj_id]
    offsets = [0, 0, 0, obj_id]

    intersect_point = [0, [0, 0]]
    poly = [dotPair(O, A), dotPair(A, B), dotPair(B, C),
            dotPair(C, D), dotPair(D, O)]
    l1 = line(center, theta)
    sflag = False
    for i in range(len(poly)):
        if ifIntersect(l1, poly[i]):
            cross = getCross(l1, poly[i])

            if cross is not None:
                sflag = True
                intersect_point[0] = i
                intersect_point[1] = cross
    if not sflag:
        return offsets
    x0, y0 = center[0], center[1]
    x1, y1 = intersect_point[1][0], intersect_point[1][1]
    w, l = label[4], label[3]
    if intersect_point[0] == 2:
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
    elif intersect_point[0] in [0, 4]:
        D2 = (x1 - x0) ** 2 + (y1 - y0) ** 2
        if intersect_point[0] == 0:
            h = abs(x0 - y0) / 2
        else:
            h = abs(x0 + y0) / 2
        if h == 0:
            d = l / 2
        else:
            d = w * np.sqrt(D2 - h ** 2) / (2 * h) + l / 2

    if d > l:
        d = l

    offsets[0] = (x1 - x0) + d * np.cos(theta)
    offsets[1] = (y1 - y0) + d * np.sin(theta)
    return offsets


def getOffsets3(label):     # [中心点x，中心点z，中心点y, 长, 宽, 高, 旋转角,分类id]
    center = [label[0], label[2]]
    theta = label[-2]
    obj_id = label[-1]
    # [delta_x, delta_z, delta_ry=0, obj_id]
    offsets = [0, 0, 0, obj_id]
    intersect_point = [0, [0, 0]]
    poly = [dotPair(B, C), dotPair(C, E),
            dotPair(E, F), dotPair(F, B)]    # 边框
    l1 = line(center, theta)

    sflag = False
    #count = 0
    for i in range(len(poly)):
        # 每条边框判断是否相交
        if ifIntersect(l1, poly[i]):
            cross = getCross(l1, poly[i])
            if cross is not None:
                #count += 1
                #print('cross:',end = ' ')
                #print(cross, end = ' ')
                #if(count == 2):
                 #   print("aaaaaaaaaaaaaaaaa",end = ' ')
                  #  print(l1.abc, end = ' ')
                   # print("角度角度角度角度:     ",end = ' ')
                    #print(theta)
                sflag = True
                intersect_point[0] = i
                intersect_point[1] = cross
                #print(i,end = ' ')
                #break # new
    #print('oh')
    if not sflag:
        return offsets
    # print(intersect_point)
    # 找到了交点
    x0, y0 = center[0], center[1]
    x1, y1 = intersect_point[1][0], intersect_point[1][1]
    # [中心点x，中心点z，中心点y, 长, 宽, 高, 旋转角,分类id]
    w, l = label[4], label[3]
    if intersect_point[0] in [0, 2]:
        if y1 - y0 == 0 or x1<= -40 + abs(label[4]/(2*np.sin(theta))) or x1>= 40-abs(label[4]/(2*np.sin(theta))):
            d = l / 2         # 特判水平
        else:
            # d = w|x1-x0|/(2|y1-y0|)
            d = w * abs(x1 - x0) / (2 * abs(y1 - y0)) + l / 2
    elif intersect_point[0] in [1, 3]:
        if x1 - x0 == 0 or y1<= 0 + abs(label[4]/(2*np.cos(theta))) or y1>= 70-abs(label[4]/(2*np.cos(theta))):     # 特判垂直
            d = l / 2
        else:
            # d = w|y1-y0|/(2|x1-x0|)
            d = w * abs(y1 - y0) / (2 * abs(x1 - x0)) + l / 2
    if d > l: # d大于l/2；小于l
        d = l
    # print(d)
    offsets[0] = -(x0 - x1) + np.where(x0-x1 >= 0,-abs(d*np.cos(theta)),abs(d*np.cos(theta)))
    offsets[1] = -(y0 - y1) + np.where(y0-y1 >= 0,-abs(d*np.sin(theta)),abs(d*np.sin(theta)))
    return offsets

TOTAL_SEG = 100
dlabel = [40, 1, 50, 8, 4, 4, 0, 5]
ltheta = np.pi / 6
point1 = [dlabel[0], dlabel[2]]
point2 = [dlabel[0] + 10 * np.cos(ltheta),
          dlabel[2] + 10 * np.sin(ltheta)]
dpoly = Polygon([B,C,E,F,B])
dline = LineString([point1, point2])


plt.figure(figsize=(10,10))
plt.plot(*dpoly.exterior.xy) # 画边框
plt.plot(*dline.xy)          # 画线
plt.xlim([-50, 50])
plt.ylim([-10, 90])
'''
ltheta = np.pi / 2
dlabel[-2] = ltheta
offsets = getOffsets3(dlabel)
#if ltheta > 0 and ltheta < np.pi/3:
 #   print(math.degrees(ltheta),end = ' ')
  #  print(offsets, end = ' ')
   # print(math.sqrt(offsets[0]**2 + offsets[1]**2))
point3 = [offsets[0]+point1[0], offsets[1]+point1[1]]
#如果offset是[0,0,0,id]，只是点加粗
eline = LineString([point1, point3])
plt.plot(*eline.xy)     # 画线
plt.show()
'''

# 线重合要特判，否则运行方向会不定。

for i in range(-TOTAL_SEG, TOTAL_SEG+1):
    ltheta = np.pi * i / TOTAL_SEG
    dlabel[-2] = ltheta
    offsets = getOffsets3(dlabel)
    #if ltheta > 0 and ltheta < np.pi/3:
     #   print(math.degrees(ltheta),end = ' ')
      #  print(offsets, end = ' ')
       # print(math.sqrt(offsets[0]**2 + offsets[1]**2))
    point3 = [offsets[0]+point1[0], offsets[1]+point1[1]]
    #如果offset是[0,0,0,id]，只是点加粗
    eline = LineString([point1, point3])
    plt.plot(*eline.xy)     # 画线
plt.show()

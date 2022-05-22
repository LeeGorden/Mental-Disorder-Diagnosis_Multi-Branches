# -*- coding: utf-8 -*-
"""
Author: LiGorden
Email: likehao1006@gmail.com
URL: https://www.linkedin.com/in/kehao-li-06a9a2235/
ResearchGate: https://www.researchgate.net/profile/Gorden-Li

Find the smallest rectangular covering all data points(Convex Hull Problem)
"""
from math import sqrt


# ------------------------------------------------------------------------------------------------------------
# Calculate the point of convex_hull
class Point(object):
    """
    Function: Create a class to store data point info
    这个类的作用在于保存数据点的信息, 便于后面一系列计算以及将其放入极坐标系
    """
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def distance_to(self, other):

        dx = self.x - other.x
        dy = self.y - other.y
        return sqrt(dx ** 2 + dy ** 2)

    # 计算两点相对于X轴的（cos）
    def angle_cos(self, other):
        # 计算self->other向量与X轴的夹角
        # cos = dx/dis(self,other)
        cos = (other.x - self.x)/self.distance_to(other)
        return cos

    def angle_sin(self, other):
        sin = (other.y - self.y) / self.distance_to(other)
        return sin

    def __str__(self):
        return '(%s, %s)' % (str(self.x), str(self.y))


# 找到极坐标系原点
def get_bottom_point(points):
    """
    variable: point - point实例的list
    return: 极坐标远点的point实例以及除了极坐标远点以外的point实例list-points
    """
    bot_point = points[0]
    temp = 0
    for i in range(1, len(points)):
        # 找到最左下角的点作为极坐标系原点
        if bot_point.y > points[i].y or (bot_point.y == points[i].y and bot_point.x > points[i].x):
            bot_point = points[i]
            temp = i
    # 删除作为原点的那个点
    del(points[temp])
    return bot_point, points


# 极坐标排序, 按cos,从大到小
def sort_polar_angle_cos(points, bot_point):
    """
    return: 按照极坐标与point连线夹角从小到大(对应cosine值从大到小)输出point。本质上是逆时针的雷达扫描式输出points
    """
    dic = dict()
    # 引进变量pre_point记录极坐标系中前一个点, 便于后面同angle点的排序(离极坐标近的点排在前, 这样排时后面Graham的需要)
    pre_point = bot_point
    for point in points:
        if pre_point != bot_point and bot_point.angle_cos(pre_point) == bot_point.angle_cos(point):
            print('there is repetition')
            if bot_point.distance_to(pre_point) > bot_point.distance_to(point):
                dic[bot_point.angle_cos(pre_point)] = point
                dic[bot_point.angle_cos(point) + 0.000000001] = pre_point
        else:
            dic[bot_point.angle_cos(point)] = point
        pre_point = point

    # for key,value in dic.items():
    #     print("{}:{}".format(key,value))

    # for key ,value in [(k,dic[k]) for k in sorted(dic.keys(),reverse=True)]:
    #     print("{}::::::{}".format(key,value))
    # 按dict的key(angle)倒叙的顺序, 输出dict的value(point实例)到一个list
    return [dic[k] for k in sorted(dic.keys(), reverse=True)]


# 叉积
def cross_product(p1, p2, p3):
    """
    return: 计算三个点p1, p2, p3组成的两个向量p1p2, p1p3的叉积。
            判断三个点依此连成两条线段走向是否为逆时针，用这两条线段向量的叉积判断：叉积>0，逆时针；反之顺时针或者共线。
    """
    x1 = p2.x-p1.x
    y1 = p2.y-p1.y
    x2 = p3.x-p1.x
    y2 = p3.y-p1.y
    return x1*y2 - x2*y1


# Graham扫描法计算凸包
def graham_scan(points, bot_point):
    """
    return: 凸包的点集list
    """
    # 这里需要将远点纳入points, 因为最后几个点的夹角需要原点进入参与评判
    points.append(bot_point)
    # 凸包列表，先加前三个, 因为从第四个点开始才能考虑要不要把之前的点移出凸包点集
    con_list = list()
    con_list.append(bot_point)
    con_list.append(points[0])
    con_list.append(points[1])

    # 寻找其他凸包上的点
    for i in range(2, len(points)):
        cro = cross_product(con_list[-2], con_list[-1], points[i])
        # 当新进入的pi与pi-2构成的向量pi-2->pi和pi-2->pi-1构成逆时针走向时, 将该点放入凸包点
        if cro > 0:
            con_list.append(points[i])
        # 当新进入的pi与pi-2构成的向量pi-2->pi和pi-2->pi-1构成顺时针走向时, 先将之前的点移除凸包点集, 再将该点放入凸包点
        elif cro <= 0:
            while True:
                con_list.pop()
                cro = cro = cross_product(con_list[-2], con_list[-1], points[i])
                if cro > 0:
                    break
            con_list.append(points[i])
    # 极坐标远点重复了两次, 所以这里要让最后一个point(原点)出队
    con_list.pop()
    # # 打印所有凸包点坐标
    # for each in con_list:
    #     print(each)
    return con_list


# ------------------------------------------------------------------------------------------------------------
# Calculate Minimum Rectangular
def projection_length(p1, p2, p3):
    """
    variable: p1, p2为凸多边形边上的两个连续点, p3为p1, p2直线外的一个点
    return: p1->p3在向量p1->p2方向上投影长度(这里有正负)
    """
    vecter_1_x = p2.x - p1.x
    vecter_1_y = p2.y - p1.y
    vecter_2_x = p3.x - p1.x
    vecter_2_y = p3.y - p1.y
    return (vecter_1_x * vecter_2_x + vecter_1_y * vecter_2_y) / p1.distance_to(p2)


def distance_to_edge(p1, p2, p3):
    """
    return: p3到向量p1->p2所在直线距离(这里只有正), 利用叉乘公式|a x b| = |a|*|b|*sinθ, a和b为向量。 |b|*sinθ = |a x b|/|a|
    """
    return cross_product(p1, p2, p3) / p1.distance_to(p2)


def find_smallest_rec(points):
    """
    Function: 旋转卡壳算法, 逆时针旋转
    Variable: points-凸包的点, 是points的实例
    Return: 最小矩形四个顶点以及面积
    """
    for i in range(len(points)):
        # 初始化外接矩形面积以及顶点, 顶点顺序依次是edge同一直线右侧顶点, 同一直线左侧顶点, 顺时针方向
        rec_area = 99999999999
        rec_vertex = [[0, 0] for _ in range(4)]
        rec_center = [0, 0]

        cur_point = points[i]
        # 当遍历到最后一个点时, 要让他与原点形成边
        j = (i + 1) % len(points)

        # 找到当前点右边的点
        right_point = points[j]

        # 遍历边以外的点
        k = (i + 2) % len(points)
        another_point = points[k]
        projection_dist = list()
        vertical_dist = list()
        while True:
            projection_dist.append(projection_length(cur_point, right_point, another_point))
            vertical_dist.append(distance_to_edge(cur_point, right_point, another_point))
            if (k + 1) % len(points) == i:
                break
            k = (k + 1) % len(points)
            another_point = points[k]

        # 创建location_adjustment用以基于p1, p2, p3计算最小外包矩阵左下角与右下角顶点。
        # 左上角与右上角不需要adjustment因为是根据左下角与右下角的出的
        location_adjustment = [[0, 0] for _ in range(2)]
        # 判断p1, p2, p3点的位置关系以获得最小外接矩阵的长宽
        if cur_point.distance_to(right_point) <= max(projection_dist) and min(projection_dist) <= 0:
            # p1, p2以外的最右边的点的投影在点p2右边, 最左边的点投影在p1左边
            rec_length = max(projection_dist) - min(projection_dist)
            location_adjustment[0][0] = cur_point.angle_cos(right_point) * max(projection_dist)
            location_adjustment[0][1] = cur_point.angle_sin(right_point) * max(projection_dist)
            location_adjustment[1][0] = cur_point.angle_cos(right_point) * min(projection_dist)
            location_adjustment[1][1] = cur_point.angle_sin(right_point) * min(projection_dist)
        elif cur_point.distance_to(right_point) <= max(projection_dist) and min(projection_dist) > 0:
            # p1, p2以外的最右边的点的投影在点p2右边, 最左边的点投影在p1右边
            rec_length = max(projection_dist)
            location_adjustment[0][0] = cur_point.angle_cos(right_point) * max(projection_dist)
            location_adjustment[0][1] = cur_point.angle_sin(right_point) * max(projection_dist)
            location_adjustment[1][0] = 0
            location_adjustment[1][1] = 0
        elif cur_point.distance_to(right_point) > max(projection_dist) and min(projection_dist) <= 0:
            # p1, p2以外的最右边的点的投影在点p2左边, 最左边的点投影在p1左边
            rec_length = cur_point.distance_to(right_point) - min(projection_dist)
            location_adjustment[0][0] = cur_point.angle_cos(right_point) * cur_point.distance_to(right_point)
            location_adjustment[0][1] = cur_point.angle_sin(right_point) * cur_point.distance_to(right_point)
            location_adjustment[1][0] = cur_point.angle_cos(right_point) * min(projection_dist)
            location_adjustment[1][1] = cur_point.angle_sin(right_point) * min(projection_dist)
        elif cur_point.distance_to(right_point) > max(projection_dist) and min(projection_dist) > 0:
            # p1, p2以外的最右边的点的投影在点p2左边, 最左边的点投影在p1右边
            rec_length = cur_point.distance_to(right_point)
            location_adjustment[0][0] = cur_point.angle_cos(right_point) * cur_point.distance_to(right_point)
            location_adjustment[0][1] = cur_point.angle_sin(right_point) * cur_point.distance_to(right_point)
            location_adjustment[1][0] = 0
            location_adjustment[1][1] = 0

        rec_width = max(vertical_dist)
        cur_rec_area = rec_length * rec_width

        if cur_rec_area <= rec_area:
            rec_area = cur_rec_area
            # 计算顶点坐标
            # 找到p1->p2直线上右边的点, 这是最小外包矩阵的右下角
            rec_vertex[0][0] = cur_point.x + location_adjustment[0][0]
            rec_vertex[0][1] = cur_point.y + location_adjustment[0][1]

            # 找到p1->p2直线上左边的点, 这是最小外包矩阵的左下角
            rec_vertex[1][0] = cur_point.x + location_adjustment[1][0]
            rec_vertex[1][1] = cur_point.y + location_adjustment[1][1]

            # 找到最小外包矩阵的左上角
            rec_vertex[2][0] = rec_vertex[1][0] - cur_point.angle_sin(right_point) * rec_width
            rec_vertex[2][1] = rec_vertex[1][1] + cur_point.angle_cos(right_point) * rec_width

            # 找到最小外包矩阵的右上角
            rec_vertex[3][0] = rec_vertex[0][0] - cur_point.angle_sin(right_point) * rec_width
            rec_vertex[3][1] = rec_vertex[0][1] + cur_point.angle_cos(right_point) * rec_width

    rec_center[0] = (rec_vertex[0][0] + rec_vertex[2][0]) / 2
    rec_center[1] = (rec_vertex[0][1] + rec_vertex[2][1]) / 2
    return rec_area, rec_vertex, rec_center


# ------------------------------------------------------------------------------------------------------------
# Rotate the Feature Map
def s_rotate(cos, sin, x, y, point_x, point_y):
    """
    Function: 顺时针绕(point_x, point_y)旋转一定度数, 将当前指针向量转至x正方向
    Variable: cos, sin当前指针向量的cos, sin值
    """
    x_rotated = (x-point_x)*cos + (y-point_y)*sin + point_x
    y_rotated = (y-point_y)*cos - (x-point_x)*sin + point_y
    return x_rotated, y_rotated


def adjust_rec(points, vertex, center):
    """
    Function: 将散点按照最小凸包外接矩形调整, 使得矩形没有歪斜
    Variable: vertex-矩形顶点坐标list
              center-矩形中心点坐标list
    """
    rotated_location = list()
    # 录入调整角度信息
    cos = (vertex[0][0] - vertex[1][0]) / sqrt((vertex[0][0] - vertex[1][0]) ** 2 + (vertex[0][1] - vertex[1][1]) ** 2)
    sin = (vertex[0][1] - vertex[1][1]) / sqrt((vertex[0][0] - vertex[1][0]) ** 2 + (vertex[0][1] - vertex[1][1]) ** 2)
    center_x = center[0]
    center_y = center[1]

    for point in points:
        x = point.x
        y = point.y
        x, y = s_rotate(cos, sin, x, y, center_x, center_y)
        rotated_location.append([x, y])

    # 输出的最后5个点为最小外包矩阵四个顶点+重复第一个顶点一次方便做图
    for item in vertex:
        x = item[0]
        y = item[1]
        x, y = s_rotate(cos, sin, x, y, center_x, center_y)
        rotated_location.append([x, y])
    
    return rotated_location

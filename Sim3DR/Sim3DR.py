# coding: utf-8

from . import _init_paths
import numpy as np
import Sim3DR_Cython

import math

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y    
    
    def __add__(self, other):
        np = Point(0, 0)
        np.x = self.x + other.x
        np.y = self.y + other.y
        return np
    
    def __sub__(self, other):
        np = Point(0, 0)
        np.x = self.x - other.x
        np.y = self.y - other.y
        return np

def get_point_weight(weight, p, p0, p1, p2):
    # vectors
    v0 = p2 - p0
    v1 = p1 - p0
    v2 = p - p0

    # dot products
    dot00 = v0.x * v0.x + v0.y * v0.y #//np.dot(v0.T, v0)
    dot01 = v0.x * v1.x + v0.y * v1.y #//np.dot(v0.T, v1)
    dot02 = v0.x * v2.x + v0.y * v2.y #//np.dot(v0.T, v2)
    dot11 = v1.x * v1.x + v1.y * v1.y #//np.dot(v1.T, v1)
    dot12 = v1.x * v2.x + v1.y * v2.y #//np.dot(v1.T, v2)

    # barycentric coordinates
    if (dot00 * dot11 - dot01 * dot01 == 0):
        inverDeno = 0
    else:
        inverDeno = 1 / (dot00 * dot11 - dot01 * dot01)

    u = (dot11 * dot02 - dot01 * dot12) * inverDeno
    v = (dot00 * dot12 - dot01 * dot02) * inverDeno

    # weight
    weight[0] = 1 - u - v
    weight[1] = v
    weight[2] = u

    return weight

def get_normal(vertices, triangles):
    normal = np.zeros_like(vertices, dtype=np.float32)
    Sim3DR_Cython.get_normal(normal, vertices, triangles, vertices.shape[0], triangles.shape[0])
    return normal

def rasterize_numpy(image, vertices, triangles, colors, depth_buffer,
        ntri, h, w, c, index_tri, alpha = 1, reverse = False):

    p = Point(0, 0)
    p0 = Point(0, 0)
    p1 = Point(0, 0)
    p2 = Point(0, 0)
    weight = [0, 0, 0]

    #print(index_tri.shape, ntri, vertices.shape)
    for i in range(ntri):
        tri_p0_ind = triangles[i][0]
        tri_p1_ind = triangles[i][1]
        tri_p2_ind = triangles[i][2]
        p0.x = vertices[tri_p0_ind][0]
        p0.y = vertices[tri_p0_ind][1]
        p0_depth = vertices[tri_p0_ind][2]
        p1.x = vertices[tri_p1_ind][0]
        p1.y = vertices[tri_p1_ind][1]
        p1_depth = vertices[tri_p1_ind][2]
        p2.x = vertices[tri_p2_ind][0]
        p2.y = vertices[tri_p2_ind][1]
        p2_depth = vertices[tri_p2_ind][2]

        x_min = max(int(math.ceil(min(p0.x, min(p1.x, p2.x)))), 0)
        x_max = min(int(math.floor(max(p0.x, max(p1.x, p2.x)))), w - 1)

        y_min = max(int(math.ceil(min(p0.y, min(p1.y, p2.y)))), 0)
        y_max = min(int(math.floor(max(p0.y, max(p1.y, p2.y)))), h - 1)

        if (x_max < x_min or y_max < y_min):
            continue

        for y in range(y_min, y_max+1):
            for x in range(x_min, x_max+1):
                p.x = float(x)
                p.y = float(y)

                # call get_point_weight function once
                weight = get_point_weight(weight, p, p0, p1, p2)

                # and judge is_point_in_tri by below line of code
                if (weight[2] >= 0 and weight[1] >= 0 and weight[0] > 0):
                    #get_point_weight(weight, p, p0, p1, p2);
                    p_depth = weight[0] * p0_depth + weight[1] * p1_depth + weight[2] * p2_depth

                    if ((p_depth > depth_buffer[y][x])):
                        for k in range(c):
                            p0_color = colors[tri_p0_ind][k]
                            p1_color = colors[tri_p1_ind][k]
                            p2_color = colors[tri_p2_ind][k]

                            p_color = weight[0] * p0_color + weight[1] * p1_color + weight[2] * p2_color
                            if (reverse):
                                image[(h - 1 - y)][x][k] = int(
                                        (1 - alpha) * image[(h - 1 - y)][x][k] + alpha * 255 * p_color)%256
#                                image[(h - 1 - y) * w * c + x * c + k] = (unsigned char) (255 * p_color);
                            else:
                                image[y][x][k] = int(
                                        (1 - alpha) * image[y][x][k] + alpha * 255 * p_color)%256
#                                image[y * w * c + x * c + k] = (unsigned char) (255 * p_color);

                        depth_buffer[y][x] = p_depth
                        index_tri[y][x] = i
    return image, depth_buffer, index_tri

def rasterize(vertices, triangles, colors, bg=None, index_tri=None,
              height=None, width=None, channel=None,
              reverse=False):
    if bg is not None:
        height, width, channel = bg.shape
    else:
        assert height is not None and width is not None and channel is not None
        bg = np.zeros((height, width, channel), dtype=np.uint8)

    buffer = np.zeros((height, width), dtype=np.float32) - 1e8

    if colors.dtype != np.float32:
        colors = colors.astype(np.float32)
    # print(triangles.shape)
    Sim3DR_Cython.rasterize(bg, vertices, triangles, colors, buffer, index_tri, triangles.shape[0], height, width, channel,
                            reverse=reverse)

    #bg, buffer, index_tri = rasterize_numpy(bg, vertices, triangles, colors, buffer, triangles.shape[0], height, width, channel, index_tri, 
    #                        reverse=reverse)

    return index_tri

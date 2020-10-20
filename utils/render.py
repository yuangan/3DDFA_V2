# coding: utf-8

__author__ = 'cleardusk'

import sys

sys.path.append('..')

import cv2
import numpy as np

from Sim3DR import RenderPipeline
from utils.functions import plot_image
from .tddfa_util import _to_ctype

import math
cfg = {
    'intensity_ambient': 0.3,
    'color_ambient': (1, 1, 1),
    'intensity_directional': 0.6,
    'color_directional': (1, 1, 1),
    'intensity_specular': 0.1,
    'specular_exp': 5,
    'light_pos': (0, 0, 5),
    'view_pos': (0, 0, 5)
}

render_app = RenderPipeline(**cfg)


def render(img, ver_lst, tri, alpha=0.6, show_flag=False, wfp=None, with_bg_flag=True):
    if with_bg_flag:
        overlap = img.copy()
    else:
        overlap = np.zeros_like(img)

    index_tri = np.zeros_like(img[:,:,0]).astype(np.int32)
    index_out = np.zeros_like(img)
    if len(ver_lst) > 1:
        print('There are two faces, which will cause fault...')
        assert(0)
    for ver_ in ver_lst:
        ver = _to_ctype(ver_.T)  # transpose
        index_tri = render_app(ver, tri, overlap, index_tri)

    # if with_bg_flag:
    #     res = cv2.addWeighted(img, 1 - alpha, overlap, alpha, 0)
    # else:
    #     res = overlap

    # if wfp is not None:
    #     cv2.imwrite(wfp, res)
    #     #index_out[:,:,0] = index_tri%256
    #     #index_out[:,:,1] = np.floor(index_tri/256)%256
    #     #index_out[:,:,2] = np.floor((index_tri/256)/256)
    #     #cv2.imwrite(wfp.replace('_3d', '_index'), index_out)
    #     cv2.imwrite(wfp.replace('_3d', '_ori'), img)
    #     print(f'Save visualization result to {wfp}')

    # if show_flag:
    #     plot_image(res)

    return index_tri

__author__ = 'gy'

import argparse
import sys
import os 

import cv2
import numpy as np
import yaml

from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from utils.depth import depth
from utils.functions import draw_landmarks, get_suffix
from utils.pncc import pncc
from utils.pose import viz_pose
from utils.render import render
from utils.serialization import ser_to_obj, ser_to_ply
from utils.tddfa_util import str2bool
from utils.uv import uv_tex

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

cfg = yaml.load(open('configs/mb1_120x120.yml'), Loader=yaml.SafeLoader)
# Init FaceBoxes and TDDFA, recommend using onnx flag
# if args.onnx:
#     import os
#     os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
#     os.environ['OMP_NUM_THREADS'] = '4'

#     from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
#     from TDDFA_ONNX import TDDFA_ONNX

#     face_boxes = FaceBoxes_ONNX()
#     tddfa = TDDFA_ONNX(**cfg)
# else:
gpu_mode = True
tddfa = TDDFA(gpu_mode=gpu_mode, **cfg)
face_boxes = FaceBoxes()

def main(args):
    global tddfa
    global face_boxes
    # Given a still image path and load to BGR channel
    img = cv2.imread(args.img_fp)

    # Detect faces, get 3DMM params and roi boxes
    boxes = face_boxes(img)
    n = len(boxes)
    if n == 0:
        print(f'No face detected, exit')
        return
        #sys.exit(-1)
    print(f'Detect {n} faces')

    param_lst, roi_box_lst = tddfa(img, boxes)

    # Visualization and serialization
    dense_flag = args.opt in ('2d_dense', '3d', 'depth', 'pncc', 'uv_tex', 'ply', 'obj')
    old_suffix = get_suffix(args.img_fp)
    new_suffix = f'.{args.opt}' if args.opt in ('ply', 'obj') else '.jpg'

    wfp = f'examples/results/{args.img_fp.split("/")[-1].replace(old_suffix, "")}_{args.opt}' + new_suffix

    ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)
    # print(len(ver_lst),ver_lst[0].shape)
    if args.opt == '3d':
        return render(img, ver_lst, tddfa.tri, alpha=0.6, show_flag=args.show_flag, wfp=wfp)
    else:
        raise ValueError(f'Unknown opt {args.opt}')

def method_searchsort(from_values, arr, to_values1, to_values2):
    sort_idx = np.argsort(from_values)
    idx = np.searchsorted(from_values,arr[mask],sorter = sort_idx)
    outx = to_values1[sort_idx][idx]
    outy = to_values2[sort_idx][idx]
    return outx, outy


# searchsorted + in1d(mask)
def method_search2(from_values, arr, to_values1, to_values2):
    datax = np.zeros_like(arr)
    datay = np.zeros_like(arr)
    mask = np.in1d(arr, from_values)
    idx = np.searchsorted(from_values, arr[mask])
    datax[mask] = to_values1[idx]   # Replace elements
    datay[mask] = to_values2[idx]   # Replace elements

    return datax.reshape(448,448), datay.reshape(448,448)

def method_list_comprehension(from_values, arr, to_values1, to_values2):
    d1 = dict(zip(from_values, to_values1))
    d2 = dict(zip(from_values, to_values2))
    outx = [d1[i] for i in arr]
    outy = [d2[i] for i in arr]
    return outx, outy

def optical_cal(tri1, tri2):
    outpath = f'examples/results/_flow_search.jpg'
    resx = np.zeros_like(tri2)
    resy = np.zeros_like(tri2)

    tri1_view = tri1.ravel() # 2d to 1d array
    tri2_view = tri2.ravel()

    val, tri1_diff, tri2_diff = np.intersect1d(tri1_view, tri2_view, assume_unique=False, return_indices=True)
    val = val[1:]
    tri1_diff = tri1_diff[1:]
    tri2_diff = tri2_diff[1:]
    # print(tri1_diff)
    w = tri1.shape[0]
    h = tri1.shape[1]
    # print(w, h)


    y1 = tri1_diff%w
    x1 = np.floor(tri1_diff/w).astype(int)
    y2 = tri2_diff%w
    x2 = np.floor(tri2_diff/w).astype(int)
    # hsv = np.zeros((h, w, 3), np.uint8)
    detax = x1 - x2
    detay = y1 - y2
    
    # broadcast
    #for i in range(1, len(val)):
    #    resx[tri1 == val[i]] = detax[i]
    #    resy[tri1 == val[i]] = detay[i]

    #resx, resy = method_searchsort(val, tri1, detax, detay)
    #resx, resy = method_list_comprehension(val, tri1, detax, detay)
    resx, resy = method_search2(val, tri1_view, detax, detay)


    flow = np.asarray([resx, resy]).transpose(1,2,0)
    print(flow.shape)
    flow = show_flow_hsv(flow.astype(float))
    #res[x1, y1] = detax%255 + detay%255
    #res2[x2, y2] = 255
    cv2.imwrite(outpath, flow)
    #cv2.imwrite(outpath.replace('ti','ti2'), res2)

def show_flow_hsv(flow, show_style=1):
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1]) # 将直角坐标系光流场转成极坐标系

    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), np.uint8)

    # 光流可视化的颜色模式
    if show_style == 1:
        hsv[..., 0] = ang * 180 / np.pi / 2 # angle弧度转角度
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX) # magnitude归到0～255之间
    elif show_type == 2:
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        hsv[..., 2] = 255

    #hsv转bgr
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The demo of still image of 3DDFA_V2')
    parser.add_argument('-c', '--config', type=str, default='configs/mb1_120x120.yml')
    parser.add_argument('-i', '--img_fp', type=str, default='/home/wei/exp/data-IJB-C-7/GT/27751/1158591/im2.png')
    parser.add_argument('-r', '--img_ref', type=str, default='/home/wei/exp/data-IJB-C-7/GT/27751/1158591/im4.png')
    parser.add_argument('-m', '--mode', type=str, default='cpu', help='gpu or cpu mode')
    parser.add_argument('-o', '--opt', type=str, default='3d',
                        choices=['2d_sparse', '2d_dense', '3d', 'depth', 'pncc', 'uv_tex', 'pose', 'ply', 'obj'])
    parser.add_argument('--show_flag', type=str2bool, default='true', help='whether to show the visualization result')
    parser.add_argument('--onnx', action='store_true', default=False)

    args = parser.parse_args()
    img_tri = main(args)
    args.img_fp = args.img_ref
    img_ref_tri = main(args)
    optical_cal(img_tri, img_ref_tri)
import os
import os.path as osp
import random
import numpy as np
import torch
import argparse
from easydict import EasyDict as edict


def ycrcb_conversion(im, format='[bs x 3 x 2D]', reverse=False):
    mat = torch.FloatTensor([
        [ 65.481/255, 128.553/255,  24.966/255], # ranged_from [0, 219/255]
        [-37.797/255, -74.203/255, 112.000/255], # ranged_from [-112/255, 112/255]
        [112.000/255, -93.786/255, -18.214/255], # ranged_from [-112/255, 112/255]
    ]).to(im.device)

    if reverse:
        mat = mat.inverse()

    if format == '[bs x 3 x 2D]':
        im = im.permute(0, 2, 3, 1)
        im = torch.matmul(im, mat.T)
        im = im.permute(0, 3, 1, 2).contiguous()
        return im
    elif format == '[2D x 3]':
        im = torch.matmul(im, mat.T)
        return im
    else:
        raise ValueError

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument("--config", type=str)
    parser.add_argument("--experiment", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--target", type=str, help="target image path")
    parser.add_argument('--log_dir', metavar='DIR', default="log/debug")
    parser.add_argument('--initial', type=str, default="random", choices=['random', 'circle'])
    parser.add_argument('--signature', nargs='+', type=str)
    parser.add_argument('--seginit', nargs='+', type=str)
    parser.add_argument("--num_segments", type=int, default=4)
    # parser.add_argument("--num_paths", type=str, default="1,1,1")
    # parser.add_argument("--num_iter", type=int, default=500)
    # parser.add_argument('--free', action='store_true')
    # Please ensure that image resolution is divisible by pool_size; otherwise the performance would drop a lot.
    # parser.add_argument('--pool_size', type=int, default=40, help="the pooled image size for next path initialization")
    # parser.add_argument('--save_loss', action='store_true')
    # parser.add_argument('--save_init', action='store_true')
    # parser.add_argument('--save_image', action='store_true')
    # parser.add_argument('--save_video', action='store_true')
    # parser.add_argument('--print_weight', action='store_true')
    # parser.add_argument('--circle_init_radius',  type=float)
    cfg = edict()
    args = parser.parse_args()
    cfg.debug = args.debug
    cfg.config = args.config
    cfg.experiment = args.experiment
    cfg.seed = args.seed
    cfg.target = args.target
    cfg.log_dir = args.log_dir
    cfg.initial = args.initial
    cfg.signature = args.signature
    # set cfg num_segments in command
    cfg.num_segments = args.num_segments
    if args.seginit is not None:
        cfg.seginit = edict()
        cfg.seginit.type = args.seginit[0]
        if cfg.seginit.type == 'circle':
            cfg.seginit.radius = float(args.seginit[1])
    return cfg


def get_experiment_id(debug=False):
    if debug:
        return 999999999999
    import time
    time.sleep(0.5)
    return int(time.time()*100)

def get_sdf(phi, method='skfmm', dx = 0.5, **kwargs):

    if method == 'skfmm':
        import skfmm
        phi = (phi-0.5)*2
        if (phi.max() <= 0) or (phi.min() >= 0):
            return np.zeros(phi.shape).astype(np.float32)
        sd = skfmm.distance(phi, dx=dx)

        flip_negative = kwargs.get('flip_negative', True)
        if flip_negative:
            sd = np.abs(sd)

        truncate = kwargs.get('truncate', 10)
        sd = np.clip(sd, -truncate, truncate)
        # print(f"max sd value is: {sd.max()}")

        zero2max = kwargs.get('zero2max', True)
        if zero2max and flip_negative:
            sd = sd.max() - sd
        elif zero2max:
            raise ValueError

        normalize = kwargs.get('normalize', 'sum')
        if normalize == 'sum':
            sd /= sd.sum()
        elif normalize == 'to1':
            sd /= sd.max()
        # plot_sdf(sd)
        return sd
import matplotlib.pyplot as plt
def plot_sdf(sdf, title='SDF Visualization'):
    """
    Plot the SDF values using matplotlib.

    Parameters:
    - sdf (np.ndarray): The computed SDF values to visualize.
    - title (str): Title of the plot. Default is 'SDF Visualization'.
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(sdf, cmap='jet')
    plt.colorbar(label='SDF Value')
    plt.title(title)
    plt.axis('off')
    plt.show()

def get_bezier_circle(radius=1, segments=4, bias=None):
    points = []
    if bias is None:
        bias = (random.random(), random.random())
    avg_degree = 360 / (segments*3)
    for i in range(0, segments*3):
        point = (np.cos(np.deg2rad(i * avg_degree)),
                    np.sin(np.deg2rad(i * avg_degree)))
        points.append(point)
    points = torch.tensor(points)
    points = (points)*radius + torch.tensor(bias).unsqueeze(dim=0)
    points = points.type(torch.FloatTensor)
    return points

def get_path_schedule(type, **kwargs):
    if type == 'repeat':
        max_path = kwargs['max_path']
        schedule_each = kwargs['schedule_each']
        return [schedule_each] * max_path
    elif type == 'list':
        schedule = kwargs['schedule']
        return schedule
    elif type == 'exp':
        import math
        base = kwargs['base']
        max_path = kwargs['max_path']
        max_path_per_iter = kwargs['max_path_per_iter']
        schedule = []
        cnt = 0
        while sum(schedule) < max_path:
            proposed_step = min(
                max_path - sum(schedule), 
                base**cnt, 
                max_path_per_iter)
            cnt += 1
            schedule += [proposed_step]
        return schedule
    else:
        raise ValueError

def edict_2_dict(x):
    if isinstance(x, dict):
        xnew = {}
        for k in x:
            xnew[k] = edict_2_dict(x[k])
        return xnew
    elif isinstance(x, list):
        xnew = []
        for i in range(len(x)):
            xnew.append( edict_2_dict(x[i]) )
        return xnew
    else:
        return x

def check_and_create_dir(path):
    pathdir = osp.split(path)[0]
    if osp.isdir(pathdir):
        pass
    else:
        os.makedirs(pathdir)

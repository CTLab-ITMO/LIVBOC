from utils import *
import torch
from easydict import EasyDict as edict
import yaml
import pydiffvg
import PIL
import PIL.Image
import os.path as osp
import numpy as np
import numpy.random as npr
import copy
import random

def make_configs():
    ###############
    # make config #
    ###############

    cfg_arg = parse_args()
    with open(cfg_arg.config, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg_default = edict(cfg['default'])
    cfg = edict(cfg[cfg_arg.experiment])
    cfg.update(cfg_default)
    cfg.update(cfg_arg)
    cfg.exid = get_experiment_id(cfg.debug)

    cfg.experiment_dir = \
        osp.join(cfg.log_dir, '{}_{}'.format(cfg.exid, '_'.join(cfg.signature)))
    configfile = osp.join(cfg.experiment_dir, 'config.yaml')
    check_and_create_dir(configfile)
    with open(osp.join(configfile), 'w') as f:
        yaml.dump(edict_2_dict(cfg), f)

    # Use GPU if available
    pydiffvg.set_use_gpu(torch.cuda.is_available())
    device = pydiffvg.get_device()

    gt = np.array(PIL.Image.open(cfg.target))
    print(f"Input image shape is: {gt.shape}")
    if len(gt.shape) == 2:
        print("Converting the gray-scale image to RGB.")
        gt = gt.unsqueeze(dim=-1).repeat(1, 1, 3)
    if gt.shape[2] == 4:
        print("Input image includes alpha channel, simply dropout alpha channel.")
        gt = gt[:, :, :3]
    gt = (gt / 255).astype(np.float32)
    gt = torch.FloatTensor(gt).permute(2, 0, 1)[None].to(device)
    if cfg.use_ycrcb:
        gt = ycrcb_conversion(gt)
    h, w = gt.shape[2:]

    path_schedule = get_path_schedule(**cfg.path_schedule)

    if cfg.seed is not None:
        random.seed(cfg.seed)
        npr.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
    render = pydiffvg.RenderFunction.apply

    shapes_record, shape_groups_record = [], []

    region_loss = None
    loss_matrix = []

    para_point, para_color = {}, {}
    para_stroke_width = None
    para_stroke_color = None
    if cfg.trainable.stroke:
        para_stroke_width, para_stroke_color = {}, {}

    pathn_record = []
    # Background
    if cfg.trainable.bg:
        para_bg = torch.tensor([1., 1., 1.], requires_grad=True, device=device)
    else:
        if cfg.use_ycrcb:
            para_bg = torch.tensor([219 / 255, 0, 0], requires_grad=False, device=device)
        else:
            para_bg = torch.tensor([1., 1., 1.], requires_grad=False, device=device)
    return gt, cfg, para_bg, h, w, path_schedule, pathn_record, shapes_record, para_stroke_width, para_stroke_color, \
           shape_groups_record, render, device


def init_shapes(num_paths,
                num_segments,
                canvas_size,
                seginit_cfg,
                shape_cnt,
                pos_init_method=None,
                trainable_stroke=False,
                gt = None):
    shapes = []
    shape_groups = []
    h, w = canvas_size

    for i in range(num_paths):
        num_control_points = [2] * num_segments
        radius = seginit_cfg.radius
        if radius is None:
            radius = npr.uniform(0.5, 1)
        pos_init = pos_init_method()
        # center = pos_init['center']
        points = pos_init['border_points'].type(torch.FloatTensor)
        color_ref = copy.deepcopy(pos_init)['mean_color']
        print("got color ref: ", color_ref)
        num_control_points = [2] * int(points.shape[0]/3)
        path = pydiffvg.Path(num_control_points=torch.LongTensor(num_control_points),
                             points=points,
                             stroke_width=torch.tensor(0.0),
                             is_closed=True)
        shapes.append(path)
        # !!!!!!problem is here. the shape group shape_ids is wrong

        if gt is not None:
            mean_color = color_ref
            fill_color_init = list(mean_color) # + [1.]
            # print('fill_color_init is: ', fill_color_init, color_ref)
            fill_color_init = torch.FloatTensor(fill_color_init)
            stroke_color_init = torch.FloatTensor(npr.uniform(size=[4]))
        else:
            fill_color_init = torch.FloatTensor(npr.uniform(size=[4]))
            stroke_color_init = torch.FloatTensor(npr.uniform(size=[4]))
        path_group = pydiffvg.ShapeGroup(
            shape_ids=torch.LongTensor([shape_cnt + i]),
            fill_color=fill_color_init,
            stroke_color=stroke_color_init,
        )
        shape_groups.append(path_group)

    point_var = []
    color_var = []

    for path in shapes:
        path.points.requires_grad = True
        point_var.append(path.points)
    for group in shape_groups:
        group.fill_color.requires_grad = True
        color_var.append(group.fill_color)
    if trainable_stroke:
        stroke_width_var = []
        stroke_color_var = []
        for path in shapes:
            path.stroke_width.requires_grad = True
            stroke_width_var.append(path.stroke_width)
        for group in shape_groups:
            group.stroke_color.requires_grad = True
            stroke_color_var.append(group.stroke_color)
        return shapes, shape_groups, point_var, color_var, stroke_width_var, stroke_color_var
    else:
        return shapes, shape_groups, point_var, color_var

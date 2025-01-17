import pydiffvg
import time
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim.lr_scheduler import  LambdaLR
import warnings
warnings.filterwarnings("ignore")

from smoothness_loss import smoothness_loss, local_angle_penalty

from classes import *

pydiffvg.set_print_timing(False)
gamma = 1.0
from utils import *
from functions import *

pydiffvg.set_print_timing(False)
if __name__ == "__main__":

    final_l2_loss = 0.0
    start_time = time.time()
    gt, cfg, para_bg, h, w, path_schedule, pathn_record, shapes_record, para_stroke_width, para_stroke_color, \
        shape_groups_record, render, device  = make_configs()
    ##################
    # start_training #
    ##################

    loss_weight = None
    loss_weight_keep = 0

    pos_init_method = Contour_path_init(
        para_bg.view(1, -1, 1, 1).repeat(1, 1, h, w), gt)

    lrlambda_f = linear_decay_lrlambda_f(cfg.num_iter, 0.05)
    optim_schedular_dict = {}
    prev_img = None
    img = None
    path_group_num=0
    for path_idx, pathn in enumerate(path_schedule):
        loss_list = []
        print("=> Adding [{}] paths, [{}] ...".format(pathn, cfg.seginit.type))
        pathn_record.append(pathn)
        pathn_record_str = '-'.join([str(i) for i in pathn_record])
        # initialize new shapes related stuffs.
        # if cfg.trainable.stroke:
        #     shapes, shape_groups, point_var, color_var, stroke_width_var, stroke_color_var = init_shapes(
        #         pathn, cfg.num_segments, (h, w),
        #         cfg.seginit, len(shapes_record),
        #         pos_init_method,
        #         trainable_stroke=True,
        #         gt=gt, )
        #     para_stroke_width[path_idx] = stroke_width_var
        #     para_stroke_color[path_idx] = stroke_color_var
        # else:
        #     shapes, shape_groups, point_var, color_var = init_shapes(
        #         pathn, cfg.num_segments, (h, w),
        #         cfg.seginit, len(shapes_record),
        #         pos_init_method,
        #         trainable_stroke=False,
        #         gt=gt, )
        try:
            if cfg.trainable.stroke:
                shapes, shape_groups, point_var, color_var, stroke_width_var, stroke_color_var = init_shapes(
                    pathn, cfg.num_segments, (h, w),
                    cfg.seginit, len(shapes_record),
                    pos_init_method,
                    trainable_stroke=True,
                    gt=gt, )
                para_stroke_width[path_idx] = stroke_width_var
                para_stroke_color[path_idx] = stroke_color_var
            else:
                shapes, shape_groups, point_var, color_var = init_shapes(
                    pathn, cfg.num_segments, (h, w),
                    cfg.seginit, len(shapes_record),
                    pos_init_method,
                    trainable_stroke=False,
                    gt=gt, )
        except:
            end_time = time.time()  # Capture end time
            elapsed_time = end_time - start_time  # Calculate elapsed time
            print(f"Total time taken for the process: {elapsed_time:.2f} seconds")
            log_dir = cfg.experiment_dir
            log_file_path = os.path.join(log_dir, "training_time_log.txt")
            # Write the total time to the log file
            with open(log_file_path, "a") as log_file:
                log_file.write(f"Total time for the process: {elapsed_time:.2f} seconds\n and l2 loss: {final_l2_loss}")
        shapes_record += shapes
        shape_groups_record += shape_groups
        prev_iter_img = None
        if cfg.save.init:
            filename = os.path.join(
                cfg.experiment_dir, "svg-init",
                "{}-init.svg".format(pathn_record_str))
            check_and_create_dir(filename)
            pydiffvg.save_svg(
                filename, w, h,
                shapes_record, shape_groups_record)

        para = {}
        if (cfg.trainable.bg) and (path_idx == 0):
            para['bg'] = [para_bg]
        para['point'] = point_var
        para['color'] = color_var
        if cfg.trainable.stroke:
            para['stroke_width'] = stroke_width_var
            para['stroke_color'] = stroke_color_var

        pg = [{'params' : para[ki], 'lr' : cfg.lr_base[ki]} for ki in sorted(para.keys())]

        optim = torch.optim.Adam(pg, betas=(0.9, 0.999))
        if cfg.trainable.record:
            scheduler = LambdaLR(
                optim, lr_lambda=lrlambda_f, last_epoch=-1)
        else:
            scheduler = LambdaLR(
                optim, lr_lambda=lrlambda_f, last_epoch=cfg.num_iter)
        optim_schedular_dict[path_idx] = (optim, scheduler)

        # Inner loop training
        t_range = tqdm(range(cfg.num_iter))
        for t in t_range:

            for _, (optim, _) in optim_schedular_dict.items():
                optim.zero_grad()

            # Forward pass: render the image.
            scene_args = pydiffvg.RenderFunction.serialize_scene(
                w, h, shapes_record, shape_groups_record)

            img = render(w, h, 2, 2, t, None, *scene_args)
            # Compose img with white background
            img = img[:, :, 3:4] * img[:, :, :3] + \
                para_bg * (1 - img[:, :, 3:4])
            if prev_img is None:
                prev_img = img.unsqueeze(0).permute(0, 3, 1, 2)  # HWC -> NCHW
            if cfg.save.video:
                filename = os.path.join(
                    cfg.experiment_dir, "video-png",
                    "{}-iter{}.png".format(pathn_record_str, t))
                check_and_create_dir(filename)
                if cfg.use_ycrcb:
                    imshow = ycrcb_conversion(
                        img, format='[2D x 3]', reverse=True).detach().cpu()
                else:
                    imshow = img.detach().cpu()
                pydiffvg.imwrite(imshow, filename, gamma=gamma)

            x = img.unsqueeze(0).permute(0, 3, 1, 2) # HWC -> NCHW
            if cfg.use_ycrcb:
                color_reweight = torch.FloatTensor([255/219, 255/224, 255/255]).to(device)
                print("we are using ycrcb")
                loss = ((x-gt)*(color_reweight.view(1, -1, 1, 1)))**2
            else:
                loss = ((x-gt)**2).sum(1)
                final_l2_loss = loss.mean().item()
                # print('final_l2_loss: ' ,final_l2_loss)
            #overlap loss
            overlap_loss = 0
            if prev_img is not None:
                prev_loss = ((prev_img - gt)**2).sum(1)
                curr_loss = ((x - gt)**2).sum(1)
                mask = ((curr_loss < prev_loss)&((curr_loss < 0.01))).expand_as(prev_img)
                prev_img[mask] = x[mask]
                overlap_loss = curr_loss[(curr_loss > prev_loss)].sum() * cfg.loss.overlap_weight #0.08
                # print("overlap_loss: ", overlap_loss.item())
            if cfg.loss.use_l1_loss:
                loss = abs(x-gt)

            if cfg.loss.use_distance_weighted_loss:
                if cfg.use_ycrcb:
                    raise ValueError
                shapes_forsdf = copy.deepcopy(shapes)
                shape_groups_forsdf = copy.deepcopy(shape_groups)
                for si in shapes_forsdf:
                    si.stroke_width = torch.FloatTensor([0]).to(device)
                for sg_idx, sgi in enumerate(shape_groups_forsdf):
                    sgi.fill_color = torch.FloatTensor([1, 1, 1, 1]).to(device)
                    sgi.shape_ids = torch.LongTensor([sg_idx]).to(device)

                sargs_forsdf = pydiffvg.RenderFunction.serialize_scene(
                    w, h, shapes_forsdf, shape_groups_forsdf)
                with torch.no_grad():
                    torch.cuda.empty_cache()
                    im_forsdf = render(w, h, 2, 2, 0, None, *sargs_forsdf)
                # use alpha channel is a trick to get 0-1 image
                im_forsdf = (im_forsdf[:, :, 3]).detach().cpu().numpy()
                loss_weight = get_sdf(im_forsdf, normalize='lo1', dx=1.0)
                loss_weight += loss_weight_keep
                loss_weight = np.clip(loss_weight, 0, 1)
                loss_weight = torch.FloatTensor(loss_weight).to(device)

            if cfg.save.loss:
                save_loss = loss.squeeze(dim=0).mean(dim=0,keepdim=False).cpu().detach().numpy()
                save_weight = loss_weight.cpu().detach().numpy()
                save_weighted_loss = save_loss*save_weight
                # normalize to [0,1]
                save_loss = (save_loss - np.min(save_loss))/np.ptp(save_loss)
                save_weight = (save_weight - np.min(save_weight))/np.ptp(save_weight)
                save_weighted_loss = (save_weighted_loss - np.min(save_weighted_loss))/np.ptp(save_weighted_loss)

                # save
                plt.imshow(save_loss, cmap='Reds')
                plt.axis('off')
                # plt.colorbar()
                filename = os.path.join(cfg.experiment_dir, "loss", "{}-iter{}-mseloss.png".format(pathn_record_str, t))
                check_and_create_dir(filename)
                plt.savefig(filename, dpi=800)
                plt.close()

                plt.imshow(save_weight, cmap='Greys')
                plt.axis('off')
                # plt.colorbar()
                filename = os.path.join(cfg.experiment_dir, "loss", "{}-iter{}-sdfweight.png".format(pathn_record_str, t))
                plt.savefig(filename, dpi=800)
                plt.close()

                plt.imshow(save_weighted_loss, cmap='Reds')
                plt.axis('off')
                # plt.colorbar()
                filename = os.path.join(cfg.experiment_dir, "loss", "{}-iter{}-weightedloss.png".format(pathn_record_str, t))
                plt.savefig(filename, dpi=800)
                plt.close()


            # print("overlap loss is: ", overlap_loss.mean().item())
            if loss_weight is None:
                loss = loss.sum(1).mean() + overlap_loss
            else:
                loss = (loss*loss_weight).mean()*10 + overlap_loss

            # if (cfg.loss.bis_loss_weight is not None)  and (cfg.loss.bis_loss_weight > 0):
            #     loss_bis = bezier_intersection_loss(point_var[0]) * cfg.loss.bis_loss_weight
            #     loss = loss + loss_bis
            if (cfg.loss.smoothness_loss_weight is not None) \
                    and (cfg.loss.smoothness_loss_weight > 0):
                loss_smoothness = 0
                loss_smoothness = smoothness_loss(point_var) * cfg.loss.smoothness_loss_weight
                local_angle_loss = 0
                # for points in point_var:
                #     local_angle_loss+= local_angle_penalty(points)
                # # print("smoothness loss: ", loss_smoothness.item())
                # loss = loss + loss_smoothness + local_angle_loss
            loss_list.append(loss.item())
            t_range.set_postfix({'loss': loss.item()})
            loss.backward()

            # step
            for _, (optim, scheduler) in optim_schedular_dict.items():
                optim.step()
                scheduler.step()

            for group in shape_groups_record:
                group.fill_color.data.clamp_(0.0, 1.0)
            prev_iter_img = x
            # prev_img = img.unsqueeze(0).permute(0, 3, 1, 2)  # HWC -> NCHW
        # prev_img = img.unsqueeze(0).permute(0, 3, 1, 2)  # HWC -> NCHW
        if cfg.loss.use_distance_weighted_loss:
            loss_weight_keep = loss_weight.detach().cpu().numpy() * 1
        prev_img = prev_iter_img
        if not cfg.trainable.record:
            for _, pi in pg.items():
                for ppi in pi:
                    pi.require_grad = False
            optim_schedular_dict = {}

        if cfg.save.image:
            filename = os.path.join(
                cfg.experiment_dir, "demo-png", "{}.png".format(pathn_record_str))
            check_and_create_dir(filename)
            if cfg.use_ycrcb:
                imshow = ycrcb_conversion(
                    img, format='[2D x 3]', reverse=True).detach().cpu()
            else:
                imshow = img.detach().cpu()
            pydiffvg.imwrite(imshow, filename, gamma=gamma)

        if cfg.save.output:
            filename = os.path.join(
                cfg.experiment_dir, "output-svg", "{}.svg".format(pathn_record_str))
            check_and_create_dir(filename)
            pydiffvg.save_svg(filename, w, h, shapes_record, shape_groups_record)


        # calculate the pixel loss
        # pixel_loss = ((x-gt)**2).sum(dim=1, keepdim=True).sqrt_() # [N,1,H, W]
        # region_loss = adaptive_avg_pool2d(pixel_loss, cfg.region_loss_pool_size)
        # loss_weight = torch.softmax(region_loss.reshape(1, 1, -1), dim=-1)\
        #     .reshape_as(region_loss)


        pos_init_method = Contour_path_init(x, gt)

        if cfg.save.video:
            print("saving iteration video...")
            img_array = []
            for ii in range(0, cfg.num_iter):
                filename = os.path.join(
                    cfg.experiment_dir, "video-png",
                    "{}-iter{}.png".format(pathn_record_str, ii))
                img = cv2.imread(filename)
                img_array.append(img)

            videoname = os.path.join(
                cfg.experiment_dir, "video-avi",
                "{}.avi".format(pathn_record_str))
            check_and_create_dir(videoname)
            out = cv2.VideoWriter(
                videoname,
                # cv2.VideoWriter_fourcc(*'mp4v'),
                cv2.VideoWriter_fourcc(*'FFV1'),
                20.0, (w, h))
            for iii in range(len(img_array)):
                out.write(img_array[iii])
            out.release()
            # shutil.rmtree(os.path.join(cfg.experiment_dir, "video-png"))
        print("The last loss is: {}".format(final_l2_loss))
        end_time = time.time()  # Capture end time
        elapsed_time = end_time - start_time  # Calculate elapsed time
        print(f"Total time taken for the process: {elapsed_time:.2f} seconds")
        log_dir = cfg.experiment_dir
        log_file_path = os.path.join(log_dir, f"training_time_log{path_group_num}.txt")
        # Write the total time to the log file
        with open(log_file_path, "a") as log_file:
            log_file.write(f"Total time for the process: {elapsed_time:.2f} seconds\n and l2 loss: {final_l2_loss}")
        path_group_num+=1
    print("The last loss is: {}".format(loss.item()))
    end_time = time.time()  # Capture end time
    elapsed_time = end_time - start_time  # Calculate elapsed time
    print(f"Total time taken for the process: {elapsed_time:.2f} seconds")
    log_dir = cfg.experiment_dir
    log_file_path = os.path.join(log_dir, "training_time_log.txt")
    # Write the total time to the log file
    with open(log_file_path, "a") as log_file:
        log_file.write(f"Total time for the process: {elapsed_time:.2f} seconds\n and l2 loss: {final_l2_loss}")
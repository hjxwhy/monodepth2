from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as funct

from layers import disp_to_depth
from utils import readlines
from options import MonodepthOptions
import datasets
import networks

np.random.seed(0)

cv2.setNumThreads(
    0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)

splits_dir = os.path.join(os.path.dirname(__file__), "splits")
PRINT_ONCE = False
NUM_SCALE = 700
NOISE = 20
SPECIFY_SEQ = 'MSBuild2018_200'


# Models which were trained with stereo supervision were trained with a nominal
# baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore,
# to convert our stereo predictions to real-world scale we multiply our depths by 5.4.


def get_ratio(pred_depth, gt_depth, num_used_pts=-1, noise_on_gt=0):
    global PRINT_ONCE
    if not PRINT_ONCE:
        if num_used_pts < 0:
            print(f'{pred_depth.shape[0]} points are used for scale')
        else:
            print(f'{num_used_pts} points are used for scale')
        print(f'{noise_on_gt}[m] noise added.')
        PRINT_ONCE = True

    # gauss noise(m)
    if noise_on_gt > 0:
        noise_gt = gt_depth + np.random.normal(0, noise_on_gt,
                                               gt_depth.shape[0])
    else:
        noise_gt = gt_depth
    if num_used_pts < 0:
        ratio = noise_gt / pred_depth
    else:
        num_pts = pred_depth.shape[0]
        choosed_idx = np.random.choice(num_pts, num_used_pts)
        ratio = noise_gt[choosed_idx] / pred_depth[choosed_idx]
    return np.median(ratio)


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())
    mae = np.abs(gt - pred).mean()

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, mae, rmse, rmse_log, a1, a2, a3


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp

def fuse_inv_depth(inv_depth, inv_depth_hat, method='mean'):
    """
    Fuse inverse depth and flipped inverse depth maps

    Parameters
    ----------
    inv_depth : torch.Tensor [B,1,H,W]
        Inverse depth map
    inv_depth_hat : torch.Tensor [B,1,H,W]
        Flipped inverse depth map produced from a flipped image
    method : str
        Method that will be used to fuse the inverse depth maps

    Returns
    -------
    fused_inv_depth : torch.Tensor [B,1,H,W]
        Fused inverse depth map
    """
    if method == 'mean':
        return 0.5 * (inv_depth + inv_depth_hat)
    elif method == 'max':
        return torch.max(inv_depth, inv_depth_hat)
    elif method == 'min':
        return torch.min(inv_depth, inv_depth_hat)
    else:
        raise ValueError('Unknown post-process method {}'.format(method))

def post_process_inv_depth(inv_depth, inv_depth_flipped, method='mean'):
    """
    Post-process an inverse and flipped inverse depth map

    Parameters
    ----------
    inv_depth : torch.Tensor [B,1,H,W]
        Inverse depth map
    inv_depth_flipped : torch.Tensor [B,1,H,W]
        Inverse depth map produced from a flipped image
    method : str
        Method that will be used to fuse the inverse depth maps

    Returns
    -------
    inv_depth_pp : torch.Tensor [B,1,H,W]
        Post-processed inverse depth map
    """
    B, C, H, W = inv_depth.shape
    inv_depth_hat = torch.flip(inv_depth_flipped, [3])
    inv_depth_fused = fuse_inv_depth(inv_depth, inv_depth_hat, method=method)
    xs = torch.linspace(0., 1., W, device=inv_depth.device,
                        dtype=inv_depth.dtype).repeat(B, C, H, 1)
    mask = 1.0 - torch.clamp(20. * (xs - 0.05), 0., 1.)
    mask_hat = torch.flip(mask, [3])
    return mask_hat * inv_depth + mask * inv_depth_hat + \
           (1.0 - mask - mask_hat) * inv_depth_fused

def compute_depth_metrics(opt, gt_depth, pred, use_gt_scale):
    # Resize pred to gt resolution
    pred = funct.interpolate(pred, gt_depth.shape[-2:], mode='bilinear', align_corners=True)
    # For each depth map
    errors = []
    for pred_i, gt_i in zip(pred, gt_depth):
        gt_i, pred_i = torch.squeeze(gt_i), torch.squeeze(pred_i)
        # Keep valid pixels (min/max depth and crop)
        valid = (gt_i > opt.min_depth) & (gt_i < opt.max_depth)
        # Stop if there are no remaining valid pixels
        if valid.sum() == 0:
            continue
        # Keep only valid pixels
        gt_i, pred_i = gt_i[valid], pred_i[valid]
        # Ground-truth median scaling if needed
        if use_gt_scale:
            pred_i = pred_i * torch.median(gt_i) / torch.median(pred_i)
        # Clamp predicted depth values to min/max values
        pred_i = pred_i.clamp(opt.min_depth, opt.max_depth)
        errors.append(compute_errors(gt_i.cpu().numpy(), pred_i.cpu().numpy()))
    errors = np.stack(errors)
    return np.mean(errors, axis=0)


def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 230

    filenames = readlines(
        os.path.join(splits_dir, opt.eval_split, "val_files.txt"))
    if opt.ext_disp_to_eval is None:

        opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

        assert os.path.isdir(opt.load_weights_folder), \
            "Cannot find a folder at {}".format(opt.load_weights_folder)

        print("-> Loading weights from {}".format(opt.load_weights_folder))

        encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
        decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

        encoder_dict = torch.load(encoder_path)

        dataset = datasets.UAVDataset(opt.data_path,
                                      filenames,
                                      encoder_dict['height'],
                                      encoder_dict['width'], [0],
                                      4,
                                      is_train=False,
                                      img_ext='.png' if opt.png else '.jpg')
        dataloader = DataLoader(dataset,
                                16,
                                shuffle=False,
                                num_workers=opt.num_workers,
                                pin_memory=True,
                                drop_last=False)

        encoder = networks.ResnetEncoder(opt.num_layers, False)
        depth_decoder = networks.DepthDecoder(encoder.num_ch_enc)

        model_dict = encoder.state_dict()
        encoder.load_state_dict(
            {k: v for k, v in encoder_dict.items() if k in model_dict})
        depth_decoder.load_state_dict(torch.load(decoder_path))

        encoder.cuda()
        encoder.eval()
        depth_decoder.cuda()
        depth_decoder.eval()

        pred_disps = []

        print("-> Computing predictions with size {}x{}".format(
            encoder_dict['width'], encoder_dict['height']))

        with torch.no_grad():
            for data in dataloader:
                input_color = data[("color", 0, 0)].cuda()

                if opt.post_process:
                    # Post-processed results require each image to have two forward passes
                    input_color = torch.cat(
                        (input_color, torch.flip(input_color, [3])), 0)

                output = depth_decoder(encoder(input_color))

                pred_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth,
                                             opt.max_depth)
                pred_disp = pred_disp.cpu()[:, 0].numpy()

                if opt.post_process:
                    N = pred_disp.shape[0] // 2
                    pred_disp = batch_post_process_disparity(
                        pred_disp[:N], pred_disp[N:, :, ::-1])

                pred_disps.append(pred_disp)

        pred_disps = np.concatenate(pred_disps)
    else:
        filenames = readlines(
            os.path.join(splits_dir, opt.eval_split, "val_files.txt"))
        # Load predictions from file
        print("-> Loading predictions from {}".format(opt.ext_disp_to_eval))
        pred_disps = np.load(opt.ext_disp_to_eval)

        if opt.eval_eigen_to_benchmark:
            eigen_to_benchmark_ids = np.load(
                os.path.join(splits_dir, "benchmark",
                             "eigen_to_benchmark_ids.npy"))

            pred_disps = pred_disps[eigen_to_benchmark_ids]

    if opt.save_pred_disps:
        output_path = os.path.join(opt.load_weights_folder,
                                   "disps_{}_split.npy".format(opt.eval_split))
        print("-> Saving predicted disparities to ", output_path)
        np.save(output_path, pred_disps)

    print("-> Evaluating")

    print("   Mono evaluation - using median scaling")

    errors = []
    ratios = []

    if len(SPECIFY_SEQ) > 0:
        print('Select', SPECIFY_SEQ)
    for i in range(pred_disps.shape[0]):
        if len(SPECIFY_SEQ) > 0 and not SPECIFY_SEQ in filenames[i]:
            continue
        gt_depth = cv2.imread(
            os.path.join(opt.data_path,
                         filenames[i].replace('rgb', 'depth').replace(' ',
                                                                      '/')),
            cv2.IMREAD_ANYDEPTH)
        gt_depth = gt_depth / 255.
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1 / pred_disp

        mask = gt_depth > 0
        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]

        pred_depth *= opt.pred_depth_scale_factor
        if not opt.disable_median_scaling:
            ratio = get_ratio(pred_depth, gt_depth, NUM_SCALE, NOISE)
            ratios.append(ratio)
            pred_depth *= ratio

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

        errors.append(compute_errors(gt_depth, pred_depth))

    if not opt.disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(
            " Scaling ratios | med: {:0.3f} | min: {:0.3f} | max: {:0.3f} | std: {:0.3f}"
                .format(med, ratios.min(), ratios.max(), np.std(ratios / med)))

    print(f'Total samples: {len(errors)}')
    mean_errors = np.array(errors).mean(0)

    print("\n  " + ("{:>8} | " * 8).format(
        "abs_rel", "sq_rel", "mae(m)", "rmse(m)", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 8).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")


def evaluateV2(opt):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 230

    filenames = readlines(
        os.path.join(splits_dir, opt.eval_split, "val_files.txt"))
    if opt.ext_disp_to_eval is None:

        opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

        assert os.path.isdir(opt.load_weights_folder), \
            "Cannot find a folder at {}".format(opt.load_weights_folder)

        print("-> Loading weights from {}".format(opt.load_weights_folder))

        encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
        decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

        encoder_dict = torch.load(encoder_path)

        dataset = datasets.UAVDataset(opt.data_path,
                                      filenames,
                                      encoder_dict['height'],
                                      encoder_dict['width'], [0],
                                      4,
                                      is_train=False,
                                      img_ext='.png' if opt.png else '.jpg')
        dataloader = DataLoader(dataset,
                                16,
                                shuffle=False,
                                num_workers=opt.num_workers,
                                pin_memory=True,
                                drop_last=False)

        encoder = networks.ResnetEncoder(opt.num_layers, False, 1 if not opt.use_sparse else 1.5)
        depth_decoder = networks.DepthDecoder(encoder.num_ch_enc)

        model_dict = encoder.state_dict()
        encoder.load_state_dict(
            {k: v for k, v in encoder_dict.items() if k in model_dict})
        depth_decoder.load_state_dict(torch.load(decoder_path))

        encoder.cuda()
        encoder.eval()
        depth_decoder.cuda()
        depth_decoder.eval()

        pred_disps = []

        print("-> Computing predictions with size {}x{}".format(
            encoder_dict['width'], encoder_dict['height']))
        metrics_modes = ('', '_pp', '_gt', '_pp_gt')
        metrics = OrderedDict()
        with torch.no_grad():
            for data in dataloader:
                input_color = data[("color", 0, 0)].cuda()
                if opt.use_sparse:
                    input_color = torch.cat([input_color, data['sparse']])
                # Post-processed results require each image to have two forward passes
                input_color_pp = torch.flip(input_color, [3])
                output_pp = depth_decoder(encoder(input_color_pp))

                output = depth_decoder(encoder(input_color))

                dis, _ = disp_to_depth(output[("disp", 0)], opt.min_depth,
                                         opt.max_depth)
                dis_pp, _ = disp_to_depth(output_pp[("disp", 0)], opt.min_depth,
                                            opt.max_depth)
                dis_pp = post_process_inv_depth(dis, dis_pp)

                depth = 1 / dis
                depth_pp = 1 / dis_pp

                for mode in metrics_modes:
                    metrics.setdefault(opt.model_name + mode, []).append(compute_depth_metrics(opt, gt_depth=data['depth_gt'],
                                                                           pred=depth_pp if 'pp' in mode else depth,
                                                                           use_gt_scale='gt' in mode))
            for key, value in metrics.items():
                print("\n  " + key + '  ' + ("{:>8} | " * 8).format(
                    "abs_rel", "sq_rel", "mae(m)", "rmse(m)", "rmse_log", "a1", "a2", "a3"))
                print(np.mean(np.stack(value, axis=0), axis=0))


if __name__ == "__main__":
    options = MonodepthOptions()
    evaluateV2(options.parse())

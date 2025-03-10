from __future__ import absolute_import, division, print_function

import os
import cv2
import glob
import argparse
from PIL import ImageShow
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

import torch
from torchvision import transforms, datasets

import networks
from layers import disp_to_depth
from evaluate_depth import STEREO_SCALE_FACTOR


def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for Monodepthv2 models.')

    parser.add_argument('--image_path',
                        type=str,
                        help='path to a test image or folder of images',
                        required=True)
    parser.add_argument('--model_folder',
                        type=str,
                        help="name of model to load")
    parser.add_argument('--ext',
                        type=str,
                        help='image extension to search for in folder',
                        default="jpg")
    parser.add_argument("--no_cuda",
                        help='if set, disables CUDA',
                        action='store_true')
    parser.add_argument("--show", help='if set, show gui', action='store_true')
    return parser.parse_args()


def test_depth(args):
    """Function to predict for a single image or folder of images
    """

    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model_path = args.model_folder
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # # LOADING PRETRAINED MODEL
    # print("   Loading pretrained encoder")
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {
        k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()
    }
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc,
                                          scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval()

    output_directory = '/tmp/visulize'
    # FINDING INPUT IMAGES
    if os.path.isfile(args.image_path):
        # Only testing on a single image
        paths = [args.image_path]
        # output_directory = os.path.dirname(args.image_path)
    elif os.path.isdir(args.image_path):
        # Searching folder for images
        paths = glob.glob(os.path.join(args.image_path,
                                       '*.{}'.format(args.ext)))
        output_directory = os.path.join(args.image_path, '../depth')
    else:
        raise Exception("Can not find args.image_path: {}".format(
            args.image_path))
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    print("-> Predicting on {:d} test images".format(len(paths)))

    # PREDICTING ON EACH IMAGE IN TURN
    viewer = ImageShow.EogViewer()
    with torch.no_grad():
        for idx, image_path in enumerate(paths):

            if image_path.endswith("_disp.jpg"):
                # don't try to predict disparity for a disparity image!
                continue

            # Load image and preprocess
            input_image = pil.open(image_path).convert('RGB')
            original_width, original_height = input_image.size
            rgb = transforms.ToTensor()(input_image).permute(1, 2, 0).numpy()
            rgb = (rgb * 255).astype(np.uint8)
            input_image = input_image.resize((feed_width, feed_height),
                                             pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)

            # PREDICTION
            input_image = input_image.to(device)
            features = encoder(input_image)
            outputs = depth_decoder(features)

            disp = outputs[("disp", 0)]
            disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width),
                mode="bilinear",
                align_corners=False)

            # Saving numpy file
            output_name = os.path.splitext(os.path.basename(image_path))[0]
            scaled_disp, depth = disp_to_depth(disp, 0.1, 100)
            disp_resized_np = disp_resized.squeeze().cpu().numpy()
            # output_npfile = os.path.join(output_directory,
            #                              "{}_disp.npy".format(output_name))
            # np.save(output_npfile, disp_resized_np)
            output_npfile = os.path.join(output_directory,
                                         "{}_disp.exr".format(output_name))
            cv2.imwrite(output_npfile, disp_resized_np)

            # Saving colormapped depth image
            vmax = np.percentile(disp_resized_np, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(),
                                              vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] *
                              255).astype(np.uint8)

            # concate
            vis = np.concatenate([rgb, colormapped_im], 0)
            # mask
            alpha = 0.8
            vis = (rgb * (1 - alpha) + colormapped_im * alpha).astype(np.uint8)

            im = pil.fromarray(vis)
            name_dest_im = os.path.join(output_directory,
                                        "{}_disp.jpg".format(output_name))
            # im.save(name_dest_im)
            if args.show:
                viewer.show(image=im)

            # name_dest_im = os.path.join(output_directory,
            #                             "{}_disp.jpeg".format(output_name))
            # im.save(name_dest_im)

            # print("   Processed {:d} of {:d} images - saved predictions to:".
            #       format(idx + 1, len(paths)))
            # print("   - {}".format(name_dest_im))
            # print("   - {}".format(name_dest_npy))

    print('-> Done!')


if __name__ == '__main__':
    args = parse_args()
    test_depth(args)

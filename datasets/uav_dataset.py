import numpy as np
import os
import PIL.Image as pil
import random
import torch
from torchvision import transforms
import cv2

from .mono_dataset import MonoDataset
from tools.add_noise import add_gausian_noise


class UAVDataset(MonoDataset):
    """Superclass for UAV dataset loaders
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size.
        # To normalize you need to scale the first row by 1 / image_width and the second row
        # by 1 / image_height. Monodepth2 assumes a principal point to be exactly centered.
        # If your principal point is far from the center you might need to disable the horizontal
        # flip augmentation.
        self.K = np.array(
            [[0.5, 0, 0.5, 0], [0, 0.5, 0.5, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float32)

        # self.full_res_shape = (1242, 375)
        self.full_res_shape = (1024, 540)
        self.idx = list(range(self.__len__()))
        self.load_sparse = True

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        if self.is_train:
            if index == 0:
                random.shuffle(self.idx)
        index = self.idx[index]
        inputs = {}
        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        line = self.filenames[index].split()
        folder = line[0]
        seq = folder.split('/')[0]

        frame_index = int(line[1].rsplit('.', 1)[0])
        side = None

        for i in self.frame_idxs:
            if i == "s":
                other_side = {"r": "l", "l": "r"}[side]
                inputs[("color", i,
                        -1)] = self.get_color(folder, frame_index, other_side,
                                              do_flip)
            else:
                inputs[("color", i,
                        -1)] = self.get_color(folder, frame_index + i, side,
                                              do_flip)

        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = self.K.copy()

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        if do_color_aug:
            color_aug = self.color_aug
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)

        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]

        if self.load_depth and not self.is_train:
            depth_gt = self.get_depth(seq, frame_index, side, do_flip)
            inputs["depth_gt"] = self.to_tensor(depth_gt)
        if self.load_sparse:
            sparse_depth = self.get_sparse(seq, frame_index, side, do_flip)
            height, width = np.where(sparse_depth > 0)
            points = np.stack([width, height], axis=1)
            depth = sparse_depth[height, width][np.newaxis, :]
            new_points, new_sparse_depth = add_gausian_noise(points, depth, inputs[("K", 0)][:3, :3].numpy())
            mask = np.logical_and.reduce(
                [(new_points[:, 1] < self.height-5), (new_points[:, 1] > 5), (new_points[:, 0] < self.width-5),
                 (new_points[:, 0] > 5)])
            new_points = new_points[mask, :]
            new_sparse_depth = new_sparse_depth[mask]
            new_depth = np.zeros_like(sparse_depth)
            new_depth[new_points[:, 1], new_points[:, 0]] = 1 / np.clip(new_sparse_depth[:, 0], 0., 200)
            inputs['sparse'] = self.to_tensor(new_depth)
            # image = np.array(self.get_color(folder, frame_index, side,
            #                                   do_flip))
            # image = cv2.resize(image,(self.width, self.height))
            # for (y,x) in points:
            #     image = cv2.circle(image, (y, x), 2, (0, 0, 255), -1)
            # for (y, x) in new_points:
            #     image = cv2.circle(image, (y, x), 2, (255, 0, 0), -1)
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # cv2.imshow('ii', image)
            # cv2.waitKey(0)

        if "s" in self.frame_idxs:
            stereo_T = np.eye(4, dtype=np.float32)
            baseline_sign = -1 if do_flip else 1
            side_sign = -1 if side == "l" else 1
            stereo_T[0, 3] = side_sign * baseline_sign * 0.1

            inputs["stereo_T"] = torch.from_numpy(stereo_T)

        return inputs

    def check_depth(self):
        if len(self.filenames) == 0:
            return False
        line = self.filenames[0].split()
        scene_name = line[0].split('/')[0]
        frame_index = int(line[1].rsplit('.', 1)[0])

        velo_filename = os.path.join(
            self.data_path, scene_name,
            "depth/{:05d}.png".format(int(frame_index)))
        return os.path.isfile(velo_filename)

    def get_image_path(self, folder, frame_index):
        f_str = "{:05d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(self.data_path, folder, f_str)
        return image_path

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index))
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

    def get_sparse(self, folder, frame_index, side, do_flip):
        f_str = "{:05d}.png".format(frame_index)
        sparse_depth_path = os.path.join(
            self.data_path,
            folder,
            'sparse',
            f_str)

        sparse_depth = pil.open(sparse_depth_path)
        sparse_depth = sparse_depth.resize((self.width, self.height), pil.NEAREST)
        sparse_depth = np.array(sparse_depth).astype(np.float32) / 255.
        if do_flip:
            sparse_depth = np.fliplr(sparse_depth)

        return sparse_depth

    def get_depth(self, folder, frame_index, side, do_flip):
        f_str = "{:05d}.png".format(frame_index)
        depth_path = os.path.join(
            self.data_path,
            folder,
            "depth",
            f_str)
        depth_gt = pil.open(depth_path)
        # depth_gt = depth_gt.resize(self.full_res_shape, pil.NEAREST)
        depth_gt = depth_gt.resize((self.width, self.height), pil.NEAREST)
        depth_gt = np.array(depth_gt).astype(np.float32) / 255.

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt

##############################################
'''
This file and main2.cpp is modified from Indoor-SfMlearner
'''
##############################################
import numpy as np
import ctypes
import os
import cv2
import argparse

so_path = os.path.join(os.path.dirname(__file__), 'libtest.so')
lib = ctypes.cdll.LoadLibrary(so_path)
c_float_p = ctypes.POINTER(ctypes.c_float)


class PixelSelector:
    def __init__(self, size=(640, 480)):
        # self.so_path = os.path.join(os.path.dirname(__file__), 'libtest.so')
        self.size = size
        self.so_path = '/home/hjx/Indoor-SfMLearner/datasets/libtest.so'
        self.lib = ctypes.cdll.LoadLibrary(self.so_path)
        self.c_float_p = ctypes.POINTER(ctypes.c_float)

    def extract_points(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.resize(image, self.size)
        data_p = image.astype(np.float32).ctypes.data_as(self.c_float_p)

        result = np.zeros((2500, 2)).astype(np.float32)
        result_p = result.ctypes.data_as(self.c_float_p)
        point_num = self.lib.main(self.size[0], self.size[1], data_p, 2000, 2500, result_p)
        points = result[:int(point_num), :]
        # points = points.astype(np.int16)
        # for (y, x) in points:
        #     image = cv2.circle(image, (x, y), 2, (255, 0, 255), -1)
        #
        # cv2.imshow("ii", image)
        # cv2.waitKey(0)
        return points

    def get_depth(self, points, depth):
        points = points.astype(np.int16)
        d = depth[points[:, 0], points[:, 1]]
        return d

    def generate_sparse_depth(self, image, depth):
        depth = cv2.resize(depth, self.size, interpolation=cv2.INTER_AREA)
        sparse_mask = np.zeros_like(depth)
        points = self.extract_points(image)
        points = points.astype(np.int16)
        sparse_mask[points[:, 0], points[:, 1]] = 1
        sparse_depth = sparse_mask * depth
        return sparse_depth

    def save_sparse_depth(self, file_path, sparse_depth):
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        cv2.imwrite(file_path, sparse_depth)
        print('save sparse depth {} to {}'.format(file_path.split('/')[-1], dir_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=False, default='/home/hjx/Documents/airsim_data')
    parser.add_argument('--save', required=False, type=bool, default=True)
    args = parser.parse_args()
    seqs = os.listdir(args.data_dir)
    seqs.sort()
    ps = PixelSelector(size=(960, 540))

    for seq in seqs:
        print(f'generate sparse depth for {seq}')
        image_files = os.listdir(os.path.join(args.data_dir, seq, 'rgb'))
        image_files.sort()
        for f in image_files:
            image_path = os.path.join(args.data_dir, seq, 'rgb', f)
            depth_path = os.path.join(args.data_dir, seq, 'depth', f)
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            assert depth is not None, 'depth don\'t have the same file, it a invalid data!!!'
            if args.save:
                sparse_depth = ps.generate_sparse_depth(image, depth)
                sparse_path = os.path.join(args.data_dir, seq, 'sparse', f)
                ps.save_sparse_depth(sparse_path, sparse_depth)
            else:
                points = ps.extract_points(image)
                points = points.astype(np.int16)
                image = cv2.resize(image, ps.size)
                for (y, x) in points:
                    image = cv2.circle(image, (x, y), 2, (255, 0, 255), -1)

                cv2.imshow("ii", image)
                cv2.waitKey(0)






import os
import stainNorm_Reinhard
from skimage import color, io
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import matplotlib.pyplot as plt


def compute_mean_std(data_path, file_list, ratio=1.0):
    total_sum = np.zeros(3)  # total sum of all pixel values in each channel
    total_square_sum = np.zeros(3)
    num_pixel = 0  # total num of all pixels

    random.shuffle(file_list)
    N = len(file_list)
    files_to_be_processed = file_list[:int(ratio*N)]

    for file_name in files_to_be_processed:
        img_name = os.path.join(data_path, file_name)
        img = io.imread(img_name)
        if len(img.shape) != 3 or img.shape[2] < 3:
            continue
        img = img[:, :, :3]

        img = color.rgb2lab(img)

        total_sum += img.sum(axis=(0, 1))
        total_square_sum += (img ** 2).sum(axis=(0, 1))
        num_pixel += img.shape[0] * img.shape[1]

    # compute the mean values of each channel
    mean_values = total_sum / num_pixel
    # compute the standard deviation
    std_values = np.sqrt(total_square_sum / num_pixel - mean_values ** 2)

    return mean_values, std_values


def main():
    parser = argparse.ArgumentParser(description='The start and end positions in the file list')
    parser.add_argument('--start', type=float, default=0.0, help='start position')
    parser.add_argument('--end', type=float, default=0.01, help='end position')
    args = parser.parse_args()

    dataset = 'TCGA-BRCA'
    data_dir = '../data/{:s}'.format(dataset)
    all_patch_indices = np.load('{:s}/20x_512x512/clustering_results/all_patch_indices_refined.npy'.format(data_dir), allow_pickle=True).item()

    save_dir = '{:s}/20x_normalized'.format(data_dir)
    slides_list = pd.read_csv('{:s}/slide_selection_final.txt'.format(data_dir), header=None)
    slides_list = list(slides_list[0].values)

    N = len(slides_list)
    slides_to_be_processed = slides_list[int(N * args.start):int(N * args.end)]

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # compute the mean and std of the reference slide
    ref_slide = 'TCGA-AN-A0AS-01Z-00-DX1'
    mean_std_file_path = '../data/{:s}/color_norm_mean_std_{:s}.npy'.format(dataset, ref_slide)
    if os.path.exists(mean_std_file_path):
        print('Loading the mean and std of reference slide: {:s} ...'.format(ref_slide))
        means, stds = np.load(mean_std_file_path)
    else:
        print('Computing the mean and std of reference slide: {:s} ...'.format(ref_slide))
        ref_img_list = os.listdir('{:s}/{:s}'.format(data_dir, ref_slide))
        means, stds = compute_mean_std('{:s}/{:s}'.format(data_dir, ref_slide), ref_img_list, ratio=0.2)
        np.save(mean_std_file_path, np.array([means, stds]))

    # initialize the normalizer
    normalizer = stainNorm_Reinhard.normalizer(means, stds)

    print('Starting color normalization ...')
    for slide_name in tqdm(slides_to_be_processed):
        slide_folder = '{:s}/20x_512x512/{:s}'.format(data_dir, slide_name)
        img_list = os.listdir(slide_folder)

        # add on 4/8/2020: consider tumor patches only #
        tumor_img_list = []
        for img_name in img_list:
            if int(img_name.split('.')[0]) in all_patch_indices[slide_name]['tumor']:
                tumor_img_list.append(img_name)
        # ------------------ #

        save_folder = '{:s}/{:s}'.format(save_dir, slide_name)
        # check if already done color normalization
        if os.path.exists(save_folder):
            img_list_normed = os.listdir(save_folder)
            if len(img_list_normed) == len(tumor_img_list):
                continue

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        slide_mean, slide_std = compute_mean_std(slide_folder, tumor_img_list, ratio=0.2)

        for i in range(0, len(tumor_img_list)):
            img_name = tumor_img_list[i]
            img_path = '{:s}/{:s}'.format(slide_folder, img_name)
            img = io.imread(img_path)
            img = img[:, :, :3]

            # perform reinhard color normalization
            img_normalized = normalizer.transform(img, slide_mean, slide_std)

            io.imsave('{:s}/{:s}.png'.format(save_folder, img_name[:-4]), img_normalized)


if __name__ == '__main__':
    main()

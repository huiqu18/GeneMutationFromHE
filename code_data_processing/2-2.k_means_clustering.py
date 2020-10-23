import os
import pandas as pd
from skimage import io
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
import argparse
import random
import matplotlib.pyplot as plt


dataset = 'TCGA-BRCA'
original_data_dir = '../data/{:s}/20x_512x512'.format(dataset)
thumbnail_dir = '../data/{:s}/thumbnails'.format(dataset)
save_dir = '../data/{:s}/20x_512x512/clustering_results'.format(dataset)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
# tumor_patches_folder = '{:s}/tumor_patches'.format(original_data_dir)

slides_list = pd.read_csv('../data/{:s}/slide_selection_final.txt'.format(dataset), header=None)
slides_list = list(slides_list[0].values)

slides_info = pd.read_pickle('../data/{:s}/slide_size_info.pickle'.format(dataset))


def main():
    parser = argparse.ArgumentParser(description='The start and end positions in the file list')
    parser.add_argument('--start', type=float, default=0.0, help='start position')
    parser.add_argument('--end', type=float, default=1.0, help='end position')
    args = parser.parse_args()

    N = len(slides_list)
    slides_to_be_processed = slides_list[int(N * args.start):int(N * args.end)]

    # clustering
    for slide_name in tqdm(slides_to_be_processed):
        k_means_clustering(slide_name, original_data_dir, '{:s}/npy_files'.format(save_dir))

    # project clustering results
    for slide_name in tqdm(slides_to_be_processed):
        filename = '{:s}/npy_files/{:s}_cluster.npy'.format(save_dir, slide_name)
        clustering_results = np.load(filename, allow_pickle=True).item()
        thumbnail = io.imread('{:s}/{:s}_thumbnail.png'.format(thumbnail_dir, slide_name))
        project_clustering_results(slide_name, clustering_results, thumbnail, '{:s}/png_files'.format(save_dir))

    all_slides_list = pd.read_csv('../data/{:s}/slide_selection_final.txt'.format(dataset), header=None)
    all_slides_list = list(all_slides_list[0].values)
    generate_all_indices(all_slides_list, original_data_dir, save_dir)

    # select cluster manually and save the results
    selection_results = pd.read_csv('{:s}/tumor_clusters.txt'.format(save_dir), delimiter=' ')
    generate_tumor_normal_indices(selection_results, '{:s}/npy_files'.format(save_dir), save_dir)


def k_means_clustering(slide_name, original_data_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    # ----- perform k-means clustering ----- #
    file_list = os.listdir('{:s}/{:s}'.format(original_data_dir, slide_name))

    features = []
    index_list = []
    for file in file_list:
        index = int(file.split('.')[0])
        image = Image.open('{:s}/{:s}/{:s}'.format(original_data_dir, slide_name, file))
        # image2 = image.resize((56, 56))   # for 224x224 patch
        image2 = image.resize((128, 128))   # for 512x512 patch
        image2 = np.array(image2)[:, :, :3]
        # plt.figure(1)
        # plt.imshow(image)
        # plt.figure(2)
        # plt.imshow(image2)
        # plt.show()

        features.append(image2.flatten())
        index_list.append(index)

    kmeans = KMeans(n_clusters=2, random_state=0).fit(np.array(features))
    index_list = np.array(index_list)
    cluster1 = index_list[kmeans.labels_ == 0]
    cluster2 = index_list[kmeans.labels_ == 1]

    cluster_results = {'0': cluster1, '1': cluster2}
    np.save('{:s}/{:s}_cluster.npy'.format(save_dir, slide_name), cluster_results)


def project_clustering_results(slide_name, clustering_results, thumbnail, save_dir, color=(255, 255, 255),
                               on_thumbnail=True):
    os.makedirs(save_dir, exist_ok=True)
    patch_size = 512  # the size of patches extracted from 20x svs image
    target_mag = 20
    slide_info = slides_info[slides_info.Slide_name == slide_name]

    mag, h, w = slide_info['Mag'].values[0], slide_info['Height'].values[0], slide_info['Width'].values[0]
    extract_patch_size = int(patch_size * mag / target_mag)

    N_patch_row = h // extract_patch_size
    stride = round(float(thumbnail.shape[0]) / N_patch_row)

    color_mask = np.ones(thumbnail.shape, dtype=np.float) * 50.0 / 255
    for j in range(0, thumbnail.shape[1], stride):
        for i in range(0, thumbnail.shape[0], stride):
            index = (j//stride) * N_patch_row + (i//stride) + 1
            if index in clustering_results['0']:
                color_mask[i:i + stride, j:j + stride, :] = np.array(color) * 0.5 / 255
            elif index in clustering_results['1']:
                color_mask[i:i + stride, j:j + stride, :] = np.array(color) / 255

    if on_thumbnail:
        thumbnail = thumbnail.astype(np.float) / 255
        result = thumbnail * color_mask
    else:
        result = color_mask

    io.imsave('{:s}/{:s}_cluster.png'.format(save_dir, slide_name), (result*255).astype(np.uint8))


def generate_all_indices(slide_names, original_data_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    all_indices = {}
    for slide_name in slide_names:
        file_list = os.listdir('{:s}/{:s}'.format(original_data_dir, slide_name))

        index_list = []
        for file in file_list:
            index = int(file.split('.')[0])
            index_list.append(index)
        all_indices[slide_name] = np.array(index_list)

    np.save('{:s}/all_indices.npy'.format(save_dir), all_indices)


def generate_tumor_normal_indices(selection_results, cluster_data_dir, save_dir):
    all_patch_indices = {}
    N_tumor = 0
    N_normal = 0
    all_indices = np.load('{:s}/all_indices.npy'.format(save_dir), allow_pickle=True).item()
    for i in tqdm(range(selection_results.shape[0])):
        slide_name, tumor_cluster = selection_results.iloc[i]
        if slide_name not in slides_list:
            all_patch_indices[slide_name] = {'normal': [],
                                             'tumor': np.array(all_indices[slide_name]).astype(np.int)}
            continue
        clustering_results = np.load('{:s}/{:s}_cluster.npy'.format(cluster_data_dir, slide_name),
                                     allow_pickle=True).item()
        if tumor_cluster == 0:
             tumor_indices = clustering_results['0']
             normal_indices = clustering_results['1']
        elif tumor_cluster == 1:
            tumor_indices = clustering_results['1']
            normal_indices = clustering_results['0']
        else:
            tumor_indices = np.append(clustering_results['0'], clustering_results['1'])
            normal_indices = []
        all_patch_indices[slide_name] = {'normal': np.array(normal_indices).astype(np.int),
                                         'tumor': np.array(tumor_indices).astype(np.int)}
        N_tumor += len(tumor_indices)
        N_normal += len(normal_indices)
    print('Number of patches: tumor {:d}\tnormal {:d}'.format(N_tumor, N_normal))
    np.save('{:s}/all_patch_indices_initial.npy'.format(save_dir), all_patch_indices)


if __name__ == '__main__':
    main()
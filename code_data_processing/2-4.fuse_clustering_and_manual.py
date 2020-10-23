import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage import io
from PIL import Image
# import cv2

dataset = 'TCGA-BRCA'
data_dir = '../data/{:s}'.format(dataset)
thumbnail_dir = '{:s}/thumbnails'.format(data_dir)
anno_dir = '{:s}/20x_512x512/manual_annotation'.format(data_dir)
clustering_results_dir = '{:s}/20x_512x512/clustering_results'.format(data_dir)
save_dir = '{:s}/20x_512x512/manual_refined'.format(data_dir)
os.makedirs(save_dir, exist_ok=True)
patch_size = 512  # the size of patches extracted from 20x svs image
target_mag = 20

anno_list = os.listdir(anno_dir)
# unchanged_list = pd.read_csv('{:s}/unchanged_list.txt'.format(anno_dir), header=None)
# anno_list.remove('unchanged_list.txt')
slides_info = pd.read_pickle('{:s}/slide_size_info.pickle'.format(data_dir))
initial_indices = np.load('{:s}/all_patch_indices_initial.npy'.format(clustering_results_dir), allow_pickle=True).item()
tumor_clusters = pd.read_csv('{:s}/tumor_clusters.txt'.format(clustering_results_dir), delimiter=' ')
refined_indices = {}

for slide_name in tqdm(slides_info['Slide_name']):
    # keep the clustering results if not manually refined
    if '{:s}_anno.png'.format(slide_name) not in anno_list:
        refined_indices[slide_name] = initial_indices[slide_name]
        continue

    slide_info = slides_info[slides_info['Slide_name'] == slide_name]
    anno_file = io.imread('{:s}/{:s}_anno.png'.format(anno_dir, slide_name))
    thumbnail = io.imread('{:s}/{:s}_thumbnail.png'.format(thumbnail_dir, slide_name))

    # assert anno_file.shape == thumbnail.shape
    if anno_file.shape[0] != thumbnail.shape[0] or anno_file.shape[1] != anno_file.shape[1]:
        anno_file = Image.fromarray(anno_file).resize((thumbnail.shape[1], thumbnail.shape[0]), resample=Image.NEAREST)
        #anno_file = cv2.resize(anno_file, (thumbnail.shape[1], thumbnail.shape[0]), interpolation=cv2.INTER_NEAREST)
        anno_file = np.array(anno_file).astype(np.float)

    mag, h, w = slide_info['Mag'].values[0], slide_info['Height'].values[0], slide_info['Width'].values[0]
    extract_patch_size = int(patch_size * mag / target_mag)
    N_patch_row = h // extract_patch_size
    stride = round(float(thumbnail.shape[0]) / N_patch_row)

    anno_indices = []
    for j in range(0, thumbnail.shape[1], stride):
        for i in range(0, thumbnail.shape[0], stride):
            index = (j // stride) * N_patch_row + (i // stride) + 1
            anno_patch = anno_file[i:i+stride, j:j+stride]
            if np.sum(anno_patch) / (stride*stride) > 0.5:
                anno_indices.append(index)

    # find the intersection of annotated indices and the initial tumor indices
    initial_tumor_indices = initial_indices[slide_name]['tumor']
    tumor_cluster = tumor_clusters[tumor_clusters['Slide_name'] == slide_name]['tumor_cluster'].values[0]
    all_indices = np.append(initial_indices[slide_name]['tumor'], initial_indices[slide_name]['normal'])
    if tumor_cluster != 2:
        refined_tumor_indices = np.intersect1d(initial_tumor_indices, np.array(anno_indices))
    else:
        refined_tumor_indices = np.intersect1d(all_indices, np.array(anno_indices))
    refined_normal_indices = np.setdiff1d(all_indices, refined_tumor_indices)
    refined_indices[slide_name] = {'tumor': np.array(refined_tumor_indices).astype(np.int),
                                   'normal': np.array(refined_normal_indices).astype(np.int)}


# # for unchanged slides, copy the initial result
# for slide_name in tqdm(unchanged_list.iloc[:, 0].values):
#     refined_indices[slide_name] = initial_indices[slide_name]

np.save('{:s}/all_patch_indices_refined.npy'.format(clustering_results_dir), refined_indices)


# ----- project tumor patches on thumbnail ----- #
color = [255, 255, 255]
for slide_name in tqdm(slides_info['Slide_name']):
    # if slide_name != 'TCGA-D8-A13Z-01Z-00-DX1':
    #     continue
    slide_info = slides_info[slides_info['Slide_name'] == slide_name]
    thumbnail = io.imread('{:s}/{:s}_thumbnail.png'.format(thumbnail_dir, slide_name))

    mag, h, w = slide_info['Mag'].values[0], slide_info['Height'].values[0], slide_info['Width'].values[0]
    extract_patch_size = int(patch_size * mag / target_mag)

    N_patch_row = h // extract_patch_size
    stride = round(float(thumbnail.shape[0]) / N_patch_row)

    color_mask = np.ones(thumbnail.shape, dtype=np.float) * 50.0 / 255
    for j in range(0, thumbnail.shape[1], stride):
        for i in range(0, thumbnail.shape[0], stride):
            index = (j//stride) * N_patch_row + (i//stride) + 1
            if index in refined_indices[slide_name]['normal']:
                color_mask[i:i + stride, j:j + stride, :] = np.array(color) * 0.5 / 255
            elif index in refined_indices[slide_name]['tumor']:
                color_mask[i:i + stride, j:j + stride, :] = np.array(color) / 255

    thumbnail = thumbnail.astype(np.float) / 255
    result = thumbnail * color_mask

    io.imsave('{:s}/{:s}_tumor.png'.format(save_dir, slide_name), (result*255).astype(np.uint8))

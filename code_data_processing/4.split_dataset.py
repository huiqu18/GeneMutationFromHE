
import os, shutil
import numpy as np
import random
import pandas as pd
import h5py
from tqdm import tqdm


dataset = 'TCGA-BRCA'
root_data_dir = '../data/{:s}'.format(dataset)


def main():

    slides_list = pd.read_csv('{:s}/slide_selection_final.txt'.format(root_data_dir), header=None)
    slides_list = list(slides_list[0].values)
    data_split = split_dataset(slides_list, root_data_dir, dataset)

    # build_hdf5_files_all(root_data_dir, slides_list)
    build_hdf5_files(root_data_dir, data_split)


def split_dataset(slides_list, save_dir, dataset):
    create_folder(save_dir)
    # split
    N_slides = len(slides_list)
    random.seed(2)
    random.shuffle(slides_list)

    if dataset == 'BRCA':
        idx1, idx2 = int(0.7*N_slides), int(0.85*N_slides)
        slides_train = slides_list[:idx1]
        slides_val = slides_list[idx1:idx2]
        slides_test = slides_list[idx2:]
        data_split = {'train': slides_train, 'val': slides_val, 'test': slides_test}
    else:
        idx = int(0.2 * N_slides)
        slides_train = slides_list[:idx]
        slides_test = slides_list[idx:]
        data_split = {'train': slides_train, 'test': slides_test}
    np.save('{:s}/data_split.npy'.format(save_dir), data_split)
    return data_split


def build_hdf5_files(root_data_dir, data_split):
    """ put all features into a large file """

    data_dir = '{:s}/20x_features_resnet101'.format(root_data_dir)
    save_dir = '{:s}/20x_features_resnet101_hdf5'.format(root_data_dir)
    create_folder(save_dir)

    indices_all = np.load('{:s}/20x_512x512/clustering_results/all_patch_indices_refined.npy'.format(root_data_dir), allow_pickle=True).item()

    with h5py.File('{:s}/train_tumor.h5'.format(save_dir), 'w') as hf:
        for slide_name in tqdm(sorted(data_split['train'])):
            indices = indices_all[slide_name]['tumor']
            features = np.load('{:s}/{:s}.npy'.format(data_dir, slide_name), allow_pickle=True).item()
            assert len(np.intersect1d(indices, features['indices'])) == len(indices)
            hf.create_dataset('{:s}/features'.format(slide_name), data=features['features'])
            hf.create_dataset('{:s}/indices'.format(slide_name), data=features['indices'])

    with h5py.File('{:s}/val_tumor.h5'.format(save_dir), 'w') as hf:
        for slide_name in tqdm(sorted(data_split['val'])):
            hf.create_group(slide_name)
            indices = indices_all[slide_name]['tumor']
            features = np.load('{:s}/{:s}.npy'.format(data_dir, slide_name), allow_pickle=True).item()
            assert len(np.intersect1d(indices, features['indices'])) == len(indices)
            hf.create_dataset('{:s}/features'.format(slide_name), data=features['features'])
            hf.create_dataset('{:s}/indices'.format(slide_name), data=features['indices'])

    with h5py.File('{:s}/test_tumor.h5'.format(save_dir), 'w') as hf:
        for slide_name in tqdm(sorted(data_split['test'])):
            hf.create_group(slide_name)
            indices = indices_all[slide_name]['tumor']
            features = np.load('{:s}/{:s}.npy'.format(data_dir, slide_name), allow_pickle=True).item()
            assert len(np.intersect1d(indices, features['indices'])) == len(indices)
            hf.create_dataset('{:s}/features'.format(slide_name), data=features['features'])
            hf.create_dataset('{:s}/indices'.format(slide_name), data=features['indices'])


def build_hdf5_files_all(slides_list):
    """ put all features into a large file """

    data_dir = '{:s}/20x_features_resnet101'.format(root_data_dir)
    save_dir = '{:s}/20x_features_resnet101_hdf5'.format(root_data_dir)
    create_folder(save_dir)

    indices_all = np.load('{:s}/20x_512x512/clustering_results/all_patch_indices_refined.npy'.format(root_data_dir), allow_pickle=True).item()

    with h5py.File('{:s}/all_tumor.h5'.format(save_dir), 'w') as hf:
        for slide_name in tqdm(sorted(slides_list)):
            indices = indices_all[slide_name]['tumor']
            features = np.load('{:s}/{:s}.npy'.format(data_dir, slide_name), allow_pickle=True).item()
            assert len(np.intersect1d(indices, features['indices'])) == len(indices)
            hf.create_dataset('{:s}/features'.format(slide_name), data=features['features'])
            hf.create_dataset('{:s}/indices'.format(slide_name), data=features['indices'])


def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


if __name__ == '__main__':
    main()

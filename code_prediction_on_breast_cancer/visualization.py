import os
import numpy as np
from tqdm import tqdm
import utils
import pandas as pd
from models import AttnClassifier
import torch
from skimage import io
import time
import random
import shutil


def project_attn(dataset, data_type, gene_or_pathway, img_data_dir=None, save_top_weighted_patches=False):
    data_dir = './experiments/{:s}/{:s}/{:s}/test_best'.format(dataset, data_type, gene_or_pathway)
    save_dir = '{:s}/attn_maps'.format(data_dir)
    os.makedirs(save_dir, exist_ok=True)

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    model_path = './experiments/{:s}/{:s}/{:s}/checkpoint_best.pth.tar'.format(dataset, data_type, gene_or_pathway)
    model = AttnClassifier(2048, 2)
    model = model.cuda()
    best_checkpoint = torch.load(model_path)
    model.load_state_dict(best_checkpoint['state_dict'])
    gamma = model.attn.gamma.item()

    data_split = np.load('../data/{:s}/data_split.npy'.format(dataset), allow_pickle=True).item()

    test_attns = np.load('{:s}/test_attn_maps.npy'.format(data_dir), allow_pickle=True).item()
    test_probs = np.load('{:s}/test_prob_results.npy'.format(data_dir), allow_pickle=True).item()
    indices = {}
    attn_weights_all = {}
    labels = {}
    for slide_name in data_split['test']:
        if slide_name not in test_attns.keys():
            continue
        attn = np.array(test_attns[slide_name]['attn'])
        indices[slide_name] = np.array(test_attns[slide_name]['true_indices'])

        attn_weights = np.sum(attn, axis=0) * gamma + 1
        attn_weights_normalized = np.log(attn_weights) / np.max(np.log(attn_weights))

        # save top 20 patches
        if save_top_weighted_patches:
            label = test_probs[slide_name]['label']
            os.makedirs('{:s}/top_weighted_patches/{:d}_{:s}'.format(data_dir, label, slide_name), exist_ok=True)
            sorted_attn_weights = sorted(attn_weights)[::-1]
            index_sorted = np.argsort(attn_weights)[::-1]
            for i in range(20):
                index = indices[slide_name][index_sorted[i]]
                img_path = '{:s}/{:s}/{:d}.png'.format(img_data_dir, slide_name, index)
                shutil.copy2(img_path, '{:s}/top_weighted_patches/{:d}_{:s}/{:d}-{:d}-{:.4f}.png'
                             .format(data_dir, label, slide_name, i+1, index, sorted_attn_weights[i]))

        attn_weights_all[slide_name] = attn_weights_normalized
        labels[slide_name] = test_probs[slide_name]['label']

    utils.project_results(dataset, data_split['test'], indices, save_dir, color=(0, 255, 0), on_thumbnail=False,
                          slides_probs=attn_weights_all, slides_labels=labels)


if __name__ == '__main__':
    dataset = 'TCGA-BRCA'
    gene_or_pathway = 'TP53'
    data_type = 'mutation'
    img_data_dir = '../data/{:s}/20x_512x512'.format(dataset)
    project_attn(dataset, data_type, gene_or_pathway, img_data_dir, save_top_weighted_patches=False)
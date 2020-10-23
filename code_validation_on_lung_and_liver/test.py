
import os
import random
from sklearn import metrics

import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt

from options import Options
from datafolder import SlideLevelDataset
from models import AttnClassifier


def main():
    opt = Options(isTrain=False)
    opt.parse()

    os.makedirs(opt.test['save_dir'], exist_ok=True)

    opt.save_options()
    opt.print_options()

    mode = 'test'
    get_probs(opt, mode)

    # compute accuracy
    save_dir = opt.test['save_dir']
    test_prob = np.load('{:s}/{:s}_prob_results.npy'.format(save_dir, mode), allow_pickle=True).item()

    txt_file = open('{:s}/{:s}_results.txt'.format(save_dir, mode), 'w')
    acc, auc, auc_CIs = compute_accuracy(test_prob)
    message = '{:11s}:  Acc: {:5.2f}\tAUC: {:5.2f} ({:5.2f}, {:5.2f})'.format('All', acc * 100, auc * 100, auc_CIs[0] * 100, auc_CIs[1] * 100)
    print(message)
    txt_file.write('{:s}\n'.format(message))
    txt_file.close()


def get_probs(opt, mode='test'):
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in opt.test['gpus'])

    model_path = opt.test['model_path']
    save_dir = opt.test['save_dir']

    # create model
    if opt.model['name'].lower() == 'attnclassifier':
        model = AttnClassifier(2048, 2)
    else:
        raise NotImplemented
    model = model.cuda()

    # print("=> loading trained model")
    best_checkpoint = torch.load(model_path)
    model.load_state_dict(best_checkpoint['state_dict'])
    print('Model obtained in epoch: {:d}'.format(best_checkpoint['epoch']))

    class_info_test = np.load('{:s}/{:s}/{:s}_class_info_{:s}.npy'.format(opt.test['data_dir'], mode, opt.data_type, opt.gene_or_pathway),
                              allow_pickle=True).item()
    test_set = SlideLevelDataset('{:s}/20x_features_resnet101_hdf5/{:s}_tumor.h5'.format(opt.root_data_dir, mode), class_info_test)

    print("=> Test begins:")
    # switch to evaluate mode
    model.eval()
    prob_results = {}
    attn_maps = {}
    slide_names = list(test_set.keys)
    for i in range(len(test_set)):
        input, target, true_indices = test_set[i]

        flag = '' if i < len(test_set)-1 else '\n'
        print('\r\t{:d}/{:d}'.format(i+1, len(test_set)), end=flag)

        with torch.no_grad():
            output, attn = model(input.unsqueeze(1).cuda())
            probs = nn.functional.softmax(output, dim=1)
            probs = probs.cpu().numpy()

            attn_map = attn[0].detach().cpu().numpy()

        slide_name = slide_names[i]
        prob_results[slide_name] = {'prob': probs, 'label': target.item()}
        attn_maps[slide_name] = {'attn': attn_map, 'true_indices': np.array(true_indices)}

        if i % 10 == 0:
            np.save('{:s}/{:s}_attn_maps.npy'.format(save_dir, mode), attn_maps)
            np.save('{:s}/{:s}_prob_results.npy'.format(save_dir, mode), prob_results)

    np.save('{:s}/{:s}_prob_results.npy'.format(save_dir, mode), prob_results)
    np.save('{:s}/{:s}_attn_maps.npy'.format(save_dir, mode), attn_maps)


def compute_accuracy(slides_probs):
    all_probs = []
    all_labels = []
    for slide_name in slides_probs.keys():
        probs = np.array(slides_probs[slide_name]['prob'])
        label = slides_probs[slide_name]['label']

        all_probs.extend(probs[:, 1])
        all_labels.append(label)

    all_probs = np.array(all_probs)
    all_pred = (all_probs > 0.5).astype(np.float)

    acc = metrics.accuracy_score(all_labels, all_pred)
    if np.unique(np.array(all_labels)).size == 1:
        auc = -0.01
        auc_CIs = [-0.01, -0.01]
    else:
        auc, auc_CIs = bootstrap_AUC_CIs(all_probs, all_labels)
    return acc, auc, auc_CIs


def plot_roc_curve(slides_probs, save_filename):
    all_probs = []
    all_labels = []
    for slide_name in slides_probs.keys():
        probs = np.array(slides_probs[slide_name]['prob'])
        label = slides_probs[slide_name]['label']

        all_probs.extend(probs[:, 1])
        all_labels.append(label)

    all_probs = np.array(all_probs)

    fpr, tpr, thresholds = metrics.roc_curve(all_labels, all_probs)
    auc = metrics.auc(fpr, tpr)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(fpr, tpr)
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.set_aspect('equal')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('ROC curve, AUC: {:.2f})'.format(auc * 100))
    fig.savefig(save_filename)
    plt.close()


def bootstrap_AUC_CIs(probs, labels):
    probs = np.array(probs)
    labels = np.array(labels)
    N_slide = len(probs)
    index_list = np.arange(0, N_slide)
    AUC_list = []
    i = 0
    while i < 1000:
        sampled_indices = random.choices(index_list, k=N_slide)
        sampled_probs = probs[sampled_indices]
        sampled_labels = labels[sampled_indices]

        if np.unique(sampled_labels).size == 1:   # reject the sample if there is only one class
            continue

        auc_bs = metrics.roc_auc_score(sampled_labels, sampled_probs)
        AUC_list.append(auc_bs)
        i += 1

    assert len(AUC_list) == 1000
    AUC_list = np.array(AUC_list)
    auc_avg = np.mean(AUC_list)
    auc_CIs = [np.percentile(AUC_list, 2.5), np.percentile(AUC_list, 97.5)]
    return auc_avg, auc_CIs


if __name__ == '__main__':
    main()
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
import parse
from skimage import io


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def project_results(dataset, slide_names, slides_indices, save_dir, slides_probs=None, color=(255, 255, 255),
                    on_thumbnail=True, slides_labels=None, postfix='projected'):
    patch_size = 512  # the size of patches extracted from 20x svs image
    target_mag = 20
    slides_info = pd.read_pickle('../data/{:s}/slide_size_info.pickle'.format(dataset))
    for slide_name in tqdm(sorted(slide_names)):
        slide_info = slides_info[slides_info.Slide_name == slide_name]
        if slide_name not in slides_indices.keys():
            continue
        indices = np.array(slides_indices[slide_name])

        mag, h, w = slide_info['Mag'].values[0], slide_info['Height'].values[0], slide_info['Width'].values[0]
        extract_patch_size = int(patch_size * mag / target_mag)

        thumbnail = io.imread('../data/{:s}/thumbnails/{:s}_thumbnail.png'.format(dataset, slide_name))
        N_patch_row = h // extract_patch_size
        # N_patch_col = w // extract_patch_size
        stride = int(float(thumbnail.shape[0]) / N_patch_row)

        if slides_probs is None:
            probs = np.ones(indices.shape).astype(np.float)
            color_mask = np.zeros(thumbnail.shape, dtype=np.float)
        else:
            probs = slides_probs[slide_name]
            color_mask = np.ones(thumbnail.shape, dtype=np.float)

        for j in range(0, thumbnail.shape[1], stride):
            for i in range(0, thumbnail.shape[0], stride):
                index = (j//stride) * N_patch_row + (i//stride) + 1
                if index in indices:
                    prob = probs[indices == index]
                    color_mask[i:i + stride, j:j + stride, :] = np.array(color) / 255 * prob

        if on_thumbnail:
            thumbnail = thumbnail.astype(np.float) / 255
            result = thumbnail * color_mask
        else:
            result = color_mask

        if slides_labels is not None:
            label = int(slides_labels[slide_name])
            io.imsave('{:s}/{:s}_{:d}_{:s}.png'.format(save_dir, slide_name, label, postfix), (result*255).astype(np.uint8))
        else:
            io.imsave('{:s}/{:s}_{:s}.png'.format(save_dir, slide_name, postfix), (result*255).astype(np.uint8))


def parse_test_results(file_path):
    with open(file_path, 'r') as file:
        results = file.readline()
        Acc = parse.search('Acc: {:f}', results)[0]
        AUC, CI_lb, CI_ub = parse.search('AUC: {:f} ({:f}, {:f})', results)
    return Acc, AUC, [CI_lb, CI_ub]


def obtain_test_results():
    dataset = 'TCGA-BRCA'
    data_type = 'cna'

    if data_type in ['mutation', 'cna']:
        mutation_info = pd.read_pickle('../data/{:s}/patient_{:s}.pickle'.format(dataset, data_type))
        gene_or_pathway_names = list(mutation_info.columns)
        gene_or_pathway_names.remove('Sample_ID')
    elif data_type in ['pathway_expression', 'pathway_cna']:
        gene_or_pathway_names = ['pi3k', 'p53', 'notch', 'rtk', 'nrf2', 'wnt', 'tgfb', 'myc', 'cellcycle', 'hippo']

    txt_file = open('./experiments/{:s}/{:s}_results.txt'.format(dataset, data_type), 'w')
    for gene in sorted(gene_or_pathway_names):
        print(gene)
        file_path = './experiments/{:s}/{:s}/{:s}/test_best/test_results.txt'.format(dataset, data_type, gene)
        Acc, AUC, CIs = parse_test_results(file_path)
        txt_file.write('{:s}\t{:.2f}\t{:.2f} ({:.2f}, {:.2f})\n'.format(gene, Acc, AUC, CIs[0], CIs[1]))


def show_figures(imgs, new_flag=False):
    import matplotlib.pyplot as plt
    if new_flag:
        for i in range(len(imgs)):
            plt.figure()
            plt.imshow(imgs[i])
    else:
        for i in range(len(imgs)):
            plt.figure(i+1)
            plt.imshow(imgs[i])

    plt.show()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def setup_logger(opt):
    mode = 'w'

    # create logger for training information
    logger = logging.getLogger('train_logger')
    logger.setLevel(logging.DEBUG)
    # create console handler and file handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler('{:s}/train_log.txt'.format(opt.train['save_dir']), mode=mode)
    file_handler.setLevel(logging.DEBUG)
    # create formatter
    formatter = logging.Formatter('%(asctime)s\t%(message)s', datefmt='%m-%d %I:%M')
    # add formatter to handlers
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    # add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # create logger for epoch results
    logger_results = logging.getLogger('results')
    logger_results.setLevel(logging.DEBUG)

    # set up logger for each result
    file_handler2 = logging.FileHandler('{:s}/epoch_results.txt'.format(opt.train['save_dir']), mode=mode)
    file_handler2.setFormatter(logging.Formatter('%(message)s'))
    logger_results.addHandler(file_handler2)

    logger.info('***** Training starts *****')
    logger.info('save directory: {:s}'.format(opt.train['save_dir']))
    if mode == 'w':
        logger_results.info('epoch | Train_loss Train_acc   || Val_loss   Val_acc     Val_AUC')

    return logger, logger_results


def plot_roc_curve(save_filename, dataset, data_type):
    from sklearn import metrics
    import matplotlib.pyplot as plt
    plt.switch_backend('agg')

    gene_names = ['CDH1', 'RB1', 'TP53', 'MAP3K1', 'TBX3', 'GATA3', 'NOTCH2']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for gene in gene_names:
        slides_probs = np.load('./experiments/{:s}/{:s}/{:s}/test_best/test_prob_results.npy'.format(dataset, data_type, gene), allow_pickle=True).item()

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

        ax.plot(fpr, tpr, label=gene)
        ax.set_aspect('equal')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        # ax.set_title('ROC curve, AUC: {:.2f})'.format(auc * 100))
    ax.legend()
    fig.savefig(save_filename)
    plt.close()


if __name__ == '__main__':
    obtain_test_results()
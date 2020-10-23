import shutil
import time
import os

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from models import AttnClassifier
import numpy as np
from sklearn import metrics

from options import Options
from datafolder import SlideLevelDataset
import utils


def main():
    global best_score, logger, logger_results, slide_weights
    opt = Options(isTrain=True)
    opt.parse()
    opt.save_options()

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in opt.train['gpus'])

    # set up logger
    logger, logger_results = utils.setup_logger(opt)
    opt.print_options(logger)

    if opt.train['random_seed'] >= 0:
        logger.info("=> Using random seed {:d}".format(opt.train['random_seed']))
        torch.manual_seed(opt.train['random_seed'])
        torch.cuda.manual_seed(opt.train['random_seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(opt.train['random_seed'])
    else:
        torch.backends.cudnn.benchmark = True

    # ---------- Create model ---------- #
    if opt.model['name'].lower() == 'attnclassifier':
        model = AttnClassifier(2048, 2)   # feature dimension is 2048 for ResNet101
                                          # number of class is 2 for all prediction tasks
    else:
        raise NotImplementedError

    model = model.cuda()
    logger.info(model)
    # ---------- End create model ---------- #

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    my_list = ['attn.gamma']
    params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] in my_list, model.named_parameters()))))
    base_params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] not in my_list, model.named_parameters()))))

    optimizer = torch.optim.Adam([{'params': base_params}, {'params': params, 'lr': opt.train['gamma_lr']}],
                                 lr=opt.train['lr'], weight_decay=opt.train['weight_decay'])

    # ---------- Data loading ---------- #
    class_info_train = np.load('{:s}/train/{:s}_class_info_{:s}.npy'.format(opt.train['data_dir'], opt.data_type, opt.gene_or_pathway), allow_pickle=True).item()
    class_info_val = np.load('{:s}/val/{:s}_class_info_{:s}.npy'.format(opt.train['data_dir'], opt.data_type, opt.gene_or_pathway), allow_pickle=True).item()
    train_set = SlideLevelDataset('{:s}/20x_features_resnet101_hdf5/train_tumor.h5'.format(opt.root_data_dir), class_info_train)
    val_set = SlideLevelDataset('{:s}/20x_features_resnet101_hdf5/val_tumor.h5'.format(opt.root_data_dir), class_info_val)

    class_weights_all = np.load('{:s}/class_weights_{:s}.npy'.format(opt.train['data_dir'],
                                                                     opt.data_type), allow_pickle=True).item()
    class_weights = class_weights_all[opt.gene_or_pathway]
    slide_weights = assign_slide_weight(train_set.class_info, class_weights)
    # ---------- End Data loading ---------- #

    # ----- Start training ---- #
    best_score = 0
    for epoch in range(opt.train['epochs']):
        # train and validate for one epoch
        train_loss, train_acc = train(opt, train_set, model, criterion, optimizer, epoch)
        val_loss, val_acc, val_auc = validate(opt, val_set, model, criterion, epoch)

        # remember best accuracy and save checkpoint
        is_best = val_auc > best_score
        best_score = max(val_auc, best_score)
        cp_flag = False
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_score': best_score,
        }, is_best, opt.train['save_dir'], cp_flag, epoch+1)

        # save training results
        logger_results.info('{:<6d}| {:<12.4f}{:<12.4f}||  {:<12.4f}{:<12.4f}{:<12.4f}'
                            .format(epoch, train_loss, train_acc, val_loss, val_acc, val_auc))

    for i in list(logger.handlers):
        logger.removeHandler(i)
        i.flush()
        i.close()
    for i in list(logger_results.handlers):
        logger_results.removeHandler(i)
        i.flush()
        i.close()


def train(opt, train_set, model, criterion, optimizer, epoch):
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    acc = utils.AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    idx_list = torch.multinomial(torch.Tensor(slide_weights), len(train_set), replacement=True)

    for i in range(0, len(idx_list), opt.train['batch_size']):
        N_slide = min(opt.train['batch_size'], len(idx_list)-i)
        loss = 0
        for k in range(N_slide):
            idx = idx_list[i + k]
            input, target, _ = train_set[idx]
            output, _ = model(input.unsqueeze(1).cuda())
            loss += criterion(output, target.cuda())

            # measure accuracy
            probs = nn.functional.softmax(output, dim=1)
            pred = torch.argmax(probs, dim=1).cpu()
            accuracy = (pred == target).sum().numpy()

            acc.update(accuracy)
            del output

        loss /= N_slide

        losses.update(loss.item(), N_slide)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % opt.train['log_interval'] == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Data Time: {data_time.avg:.3f}\t'
                        'Batch Time: {batch_time.avg:.3f}\t'
                        'Loss: {loss.avg:.3f}\t'
                        'Acc: {acc.avg:.4f}'
                        .format(epoch, i, len(train_set), data_time=data_time, batch_time=batch_time,
                                loss=losses, acc=acc))

    logger.info('=> Train Avg: Loss: {loss.avg:.3f}\t\tAcc: {acc.avg:.4f}'
                .format(loss=losses, acc=acc))

    return losses.avg, acc.avg


def validate(opt, val_set, model, criterion, epoch):
    batch_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    acc = utils.AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    probs_all = []
    targets_all = []
    for i in range(len(val_set)):
        input, target, _ = val_set[i]

        # compute output
        with torch.no_grad():
            output, _ = model(input.unsqueeze(1).cuda())
            loss = criterion(output, target.cuda())

        # measure accuracy and record loss
        probs = nn.functional.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).detach().cpu()
        accuracy = (pred == target).sum().detach().cpu().numpy()

        # measure accuracy and record loss
        losses.update(loss.item())
        acc.update(accuracy)

        probs_all.append(probs.detach().cpu())
        targets_all.append(target.detach().cpu())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        del output, probs

        if i % opt.train['log_interval'] == 0:
            logger.info('Val: [{0}][{1}/{2}]\t'
                        'Time: {batch_time.avg:.3f}\t'
                        'Loss: {loss.avg:.3f}'
                        .format(epoch, i, len(val_set), batch_time=batch_time, loss=losses))

    probs_all = torch.cat(probs_all, dim=0).numpy()
    targets_all = torch.cat(targets_all, dim=0).numpy()

    pred = np.argmax(probs_all, axis=1)
    tp = np.sum((pred == 1) * (targets_all == 1))
    tn = np.sum((pred == 0) * (targets_all == 0))
    fp = np.sum((pred == 1) * (targets_all == 0))
    fn = np.sum((pred == 0) * (targets_all == 1))
    acc = metrics.accuracy_score(targets_all, pred)
    auc = metrics.roc_auc_score(targets_all, probs_all[:, 1])

    logger.info('Val Avg: {}\tLoss: {:.3f}\tAcc: {:.4f}\tAUC: {:.4f}\n'
                'TP: {:d}\tTN: {:d}\tFP: {:d}\tFN: {:d}\n'
                .format(epoch, losses.avg, acc, auc, tp, tn, fp, fn))

    return losses.avg, acc, auc


def save_checkpoint(state, is_best, save_dir, cp_flag, epoch):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    filename = '{:s}/checkpoint.pth.tar'.format(save_dir)
    torch.save(state, filename)
    if cp_flag:
        shutil.copyfile(filename, '{:s}/checkpoint_{:d}.pth.tar'.format(save_dir, epoch))
    if is_best:
        shutil.copyfile(filename, '{:s}/checkpoint_best.pth.tar'.format(save_dir))


def assign_slide_weight(class_info, weights):
    keys = list(class_info.keys())
    slide_weights = np.zeros(len(keys))
    for i in range(len(keys)):
        target = class_info[keys[i]]
        slide_weights[i] = weights[0] if target == 0 else weights[1]
    return slide_weights


if __name__ == '__main__':
    main()

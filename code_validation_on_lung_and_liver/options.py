import os
import numpy as np
import argparse


class Options:
    def __init__(self, isTrain):
        self.isTrain = isTrain  # train or test mode
        self.dataset = 'TCGA-LUAD'
        self.root_data_dir = '../data/{:s}'.format(self.dataset)
        self.gene_or_pathway = 'RB1'             # gene name | pathway name
        self.data_type = 'mutation'    # mutation | cna | pathway_expression | pathway_cna

        # --- model hyper-parameters --- #
        self.model = dict()
        self.model['name'] = 'AttnClassifier'

        # --- training params --- #
        self.train = dict()
        self.train['random_seed'] = -1
        self.train['data_dir'] = '{:s}/data_for_train'.format(self.root_data_dir)  # path to data
        self.train['save_dir'] = './experiments/{:s}/{:s}/{:s}'.format(self.dataset, self.data_type, self.gene_or_pathway)  # path to save results
        self.train['input_size'] = 224          # input size of the image
        self.train['epochs'] = 30         # number of slide-level training epochs
        self.train['batch_size'] = 8     # batch size of slide-level training
        self.train['checkpoint_freq'] = 30      # epoch to save checkpoints
        self.train['lr'] = 0.0001                # initial learning rate
        self.train['gamma_lr'] = 0.001       # learning rate of gamma in attention
        self.train['weight_decay'] = 0.0001       # weight decay
        self.train['log_interval'] = 100         # iterations to print training results
        self.train['workers'] = 4               # number of workers to load images
        self.train['gpus'] = [1]              # select gpu devices
        self.train['checkpoint'] = '../code_prediction_on_breast_cancer/experiments/TCGA-BRCA/{:s}/{:s}/checkpoint_best.pth.tar'.format(self.data_type, self.gene_or_pathway)

        # --- test parameters --- #
        self.test = dict()
        self.test['test_epoch'] = '30'
        self.test['gpus'] = [1]
        self.test['batch_size'] = 8
        self.test['data_dir'] = '{:s}/data_for_train'.format(self.root_data_dir)
        self.test['save_flag'] = False
        self.test['save_dir'] = './experiments/{:s}/{:s}/{:s}/test_{:s}'.format(self.dataset, self.data_type, self.gene_or_pathway, self.test['test_epoch'])
        self.test['model_path'] = './experiments/{:s}/{:s}/{:s}/checkpoint_{:s}.pth.tar'.format(self.dataset, self.data_type, self.gene_or_pathway, self.test['test_epoch'])

    def parse(self):
        """ Parse the options, replace the default value if there is a new input """
        parser = argparse.ArgumentParser(description='')
        if self.isTrain:
            parser.add_argument('--dataset', type=str, default=self.dataset, help='dataset')
            parser.add_argument('--root-data-dir', type=str, default=self.root_data_dir, help='root-data-dir')
            parser.add_argument('--gene-or-pathway', type=str, default=self.gene_or_pathway, help='gene or pathway name')
            parser.add_argument('--data-type', type=str, default=self.data_type, help='data type')
            parser.add_argument('--model', type=str, default=self.model['name'], help='model name')
            parser.add_argument('--random-seed', type=int, default=self.train['random_seed'], help='random seed for reproducibility')
            parser.add_argument('--epochs', type=int, default=self.train['epochs'], help='number of epochs to train')
            parser.add_argument('--batch-size', type=int, default=self.train['batch_size'],
                                help='input batch size for training')
            parser.add_argument('--lr', type=float, default=self.train['lr'], help='learning rate')
            parser.add_argument('--log-interval', type=int, default=self.train['log_interval'], help='how many batches to wait before logging training status')
            parser.add_argument('--gpus', type=int, nargs='+', default=self.train['gpus'], help='GPUs for training')
            parser.add_argument('--data-dir', type=str, default=self.train['data_dir'], help='directory of training data')
            parser.add_argument('--save-dir', type=str, default=self.train['save_dir'], help='directory to save training results')
            parser.add_argument('--checkpoint-path', type=str, default=self.train['checkpoint'], help='path to the checkpoint')
            args = parser.parse_args()

            self.gene_or_pathway = args.gene_or_pathway
            self.dataset = args.dataset
            self.data_type = args.data_type
            self.root_data_dir = args.root_data_dir
            self.model['name'] = args.model
            self.train['random_seed'] = args.random_seed
            self.train['batch_size'] = args.batch_size
            self.train['epochs'] = args.epochs
            self.train['lr'] = args.lr
            self.train['log_interval'] = args.log_interval
            self.train['gpus'] = args.gpus
            self.train['data_dir'] = args.data_dir
            self.train['save_dir'] = args.save_dir
            self.train['checkpoint'] = args.checkpoint_path
            if not os.path.exists(self.train['save_dir']):
                os.makedirs(self.train['save_dir'], exist_ok=True)

        else:
            parser.add_argument('--dataset', type=str, default=self.dataset, help='dataset')
            parser.add_argument('--root-data-dir', type=str, default=self.root_data_dir, help='root-data-dir')
            parser.add_argument('--gene-or-pathway', type=str, default=self.gene_or_pathway, help='gene or pathway name')
            parser.add_argument('--data-type', type=str, default=self.data_type, help='data type')
            parser.add_argument('--model', type=str, default=self.model['name'], help='model name')
            parser.add_argument('--gpus', type=int, nargs='+', default=self.test['gpus'], help='GPUs for test')
            parser.add_argument('--batch-size', type=int, default=self.test['batch_size'], help='input batch size for test')
            parser.add_argument('--save-flag', type=bool, default=self.test['save_flag'], help='flag to save the network outputs and predictions')
            parser.add_argument('--data-dir', type=str, default=self.test['data_dir'], help='directory of test images')
            parser.add_argument('--save-dir', type=str, default=self.test['save_dir'], help='directory to save test results')
            parser.add_argument('--model-path', type=str, default=self.test['model_path'], help='train model to be evaluated')
            args = parser.parse_args()
            self.gene_or_pathway = args.gene_or_pathway
            self.dataset = args.dataset
            self.root_data_dir = args.root_data_dir
            self.data_type = args.data_type
            self.model['name'] = args.model
            self.test['gpus'] = args.gpus
            self.test['batch_size'] = args.batch_size
            self.test['save_flag'] = args.save_flag
            self.test['data_dir'] = args.data_dir
            self.test['save_dir'] = args.save_dir
            self.test['model_path'] = args.model_path

            if not os.path.exists(self.test['save_dir']):
                os.makedirs(self.test['save_dir'], exist_ok=True)

    def print_options(self, logger=None):
        message = '\n'
        message += self._generate_message_from_options()
        if not logger:
            print(message)
        else:
            logger.info(message)

    def save_options(self):
        if self.isTrain:
            filename = '{:s}/train_options.txt'.format(self.train['save_dir'])
        else:
            filename = '{:s}/test_options.txt'.format(self.test['save_dir'])
        message = self._generate_message_from_options()
        file = open(filename, 'w')
        file.write(message)
        file.close()

    def _generate_message_from_options(self):
        message = ''
        message += '# {str:s} Options {str:s} #\n'.format(str='-' * 25)
        train_groups = ['model', 'train', 'transform']
        test_groups = ['model', 'test', 'transform']
        cur_group = train_groups if self.isTrain else test_groups

        for group, options in self.__dict__.items():
            if group not in train_groups + test_groups:
                message += '{:>15}: {:<35}\n'.format(group, str(options))
            elif group in cur_group:
                message += '\n{:s} {:s} {:s}\n'.format('*' * 15, group, '*' * 15)
                if group == 'transform':
                    for name, val in options.items():
                        # message += '{:s}:\n'.format(name)
                        val = str(val).replace('\n', ',\n{:17}'.format(''))
                        message += '{:>15}: {:<35}\n'.format(name, str(val))
                else:
                    for name, val in options.items():
                        message += '{:>15}: {:<35}\n'.format(name, str(val))
        message += '# {str:s} End {str:s} #\n'.format(str='-' * 26)
        return message



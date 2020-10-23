import numpy as np
import pandas as pd
import os


def main():
    dataset = 'TCGA-LUAD'
    data_type = 'cna'   # mutation or cna
    root_data_dir = '../data/{:s}'.format(dataset)
    data_split = np.load('{:s}/data_split.npy'.format(root_data_dir), allow_pickle=True).item()
    mutation_data = pd.read_pickle('{:s}/patient_{:s}.pickle'.format(root_data_dir, data_type))
    save_dir = '{:s}/data_for_train'.format(root_data_dir)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs('{:s}/train'.format(save_dir), exist_ok=True)
    os.makedirs('{:s}/test'.format(save_dir), exist_ok=True)

    if dataset == 'TCGA-LUAD':
        if data_type == 'mutation':
            gene_names = ['ARID1A', 'KMT2C', 'NF1', 'NOTCH2', 'PIK3CA',  'RB1',  'RYR2',  'TP53', 'USH2A']
        else:
            gene_names = ['APH1A', 'CDKN2A', 'CDKN2B', 'EIF4EBP1', 'FGFR1', 'HEY1', 'KAT6A', 'MCL1', 'MYC', 'NCSTN', 'NOTCH2', 'PTK2', 'RAB25', 'RIT1']
    elif dataset == 'TCGA-LIHC':
        if data_type == 'mutation':
            gene_names = ['ARID1A', 'KMT2C', 'NF1', 'RB1', 'RYR2', 'TP53', 'USH2A']
        else:
            gene_names = ['AKT3', 'APH1A', 'CCND1', 'CDKN2A', 'CDKN2B', 'E2F5', 'EIF4EBP1', 'FGFR1', 'HEY1', 'IRF2BP2',
                          'KAT6A', 'MCL1', 'MDM4', 'MMP16', 'MYC', 'NCSTN', 'NOTCH2', 'PARP1', 'PSEN2', 'PTK2', 'RAB25',
                          'RIT1', 'RYR2', 'TGFB2', 'USH2A']
    for gene_name in sorted(gene_names):
        print(gene_name)
        generate_original_labels(gene_name, mutation_data, data_split, save_dir, data_type)


def generate_original_labels(gene_name, mutation_data, data_split, save_dir, data_type):
    train_samples = [x[:15] for x in data_split['train']]
    test_samples = [x[:15] for x in data_split['test']]
    train_data = mutation_data[mutation_data['Sample_ID'].isin(train_samples)]
    test_data = mutation_data[mutation_data['Sample_ID'].isin(test_samples)]
    
    print_label_distribution(mutation_data[gene_name], train_data[gene_name], test_data[gene_name])

    # calculate the weight for each class
    freqs = np.array([(train_data[gene_name] == 0).sum() / len(train_data[gene_name]),
                      (train_data[gene_name] == 1).sum() / len(train_data[gene_name])])
    if not os.path.isfile('{:s}/class_weights_{:s}.npy'.format(save_dir, data_type)):
        weights = {}
    else:
        weights = np.load('{:s}/class_weights_{:s}.npy'.format(save_dir, data_type), allow_pickle=True).item()
    weights[gene_name] = 0.5/freqs
    np.save('{:s}/class_weights_{:s}.npy'.format(save_dir, data_type), weights)

    train_dict = pd.Series(train_data[gene_name].values, index=train_data['Sample_ID']).to_dict()
    test_dict = pd.Series(test_data[gene_name].values, index=test_data['Sample_ID']).to_dict()

    # test if the transformation is correct
    for key, val in train_dict.items():
        assert val == train_data[gene_name][train_data['Sample_ID'] == key].values[0]
    for key, val in test_dict.items():
        assert val == test_data[gene_name][test_data['Sample_ID'] == key].values[0]

    np.save('{:s}/train/{:s}_class_info_{:s}.npy'.format(save_dir, data_type, gene_name), train_dict)
    np.save('{:s}/test/{:s}_class_info_{:s}.npy'.format(save_dir, data_type, gene_name), test_dict)


def print_label_distribution(all_data, train_data, test_data):
    N_mutated = (all_data == 1).sum()
    N_non_mutated = (all_data == 0).sum()
    print('Total ({:4.1f} %):\t0:{:>4d}\t1:{:>4d}'.format(100.0*N_mutated/(N_mutated+N_non_mutated), N_non_mutated, N_mutated))
    N_mutated = (train_data == 1).sum()
    N_non_mutated = (train_data == 0).sum()
    print('Train ({:4.1f} %):\t0:{:>4d}\t1:{:>4d}'.format(100.0*N_mutated/(N_mutated+N_non_mutated), N_non_mutated, N_mutated))
    N_mutated = (test_data == 1).sum()
    N_non_mutated = (test_data == 0).sum()
    print('Test  ({:4.1f} %):\t0:{:>4d}\t1:{:>4d}'.format(100.0*N_mutated/(N_mutated+N_non_mutated), N_non_mutated, N_mutated))


if __name__ == '__main__':
    main()
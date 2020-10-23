
import os
import pandas as pd
import numpy as np

# From Figure 2 of Sanchez-Vega F, et al. Oncogenic signaling pathways in the cancer genome atlas. Cell 173, 321-337. e310 (2018).
pathways = dict()
pathways['wnt'] = {'genes': ['WIF1', 'SFRP1', 'SFRP2', 'SFRP3', 'SFRP4', 'SFRP5', 'RNF43', 'ZNRF3', 'LRP5',
                             'LRP6', 'DKK1', 'DKK2', 'DKK3', 'DKK4', 'GSK3B', 'APC', 'CTNNB1', 'AXIN1', 'AXIN2',
                             'AMER1', 'TCF7', 'TCF7L1', 'TCF7L2', 'TLE1', 'TLE2', 'TLE3', 'TLE4'],
                   'signs': [-1, -1, -1, -1, -1, -1, -1, -1, -1,
                             -1, -1, -1, -1, -1, -1, -1, 1, -1, -1,
                             -1, -1, -1, -1, -1, -1, -1, -1]}
pathways['pi3k'] = {'genes': ['PTEN', 'INPP4B', 'PIK3R2', 'PIK3R1', 'PIK3R3', 'PIK3CA', 'PIK3CB', 'AKT1', 'AKT2',
                              'AKT3', 'PPP2R1A', 'STK11', 'TSC1', 'TSC2', 'RHEB', 'RICTOR', 'MTOR', 'RPTOR'],
                    'signs': [-1, -1, 1, -1, -1, 1, 1, 1, 1,
                              1, -1, -1, -1, -1, 1, 1, 1, 1]}
pathways['p53'] = {'genes': ['CDKN2A', 'MDM2', 'MDM4', 'TP53', 'ATM', 'CHEK2', 'RPS6KA3'],
                   'signs': [-1, 1, 1, -1, -1, -1, -1]}
pathways['notch'] = {'genes': ['NOV', 'CNTN6', 'JAG2', 'ARRDC1', 'NOTCH1', 'NOTCH2', 'NOTCH3', 'NOTCH4', 'DNER',
                               'PSEN2', 'FBXW7', 'CUL1', 'MAML3', 'KAT2B', 'CREBBP', 'EP300', 'NCOR1', 'NCOR2',
                               'SPEN', 'KDM5A', 'HES-X', 'HEY-X'],
                     'signs': [-1, -1, -1, 1, -1, -1, -1, -1, -1,
                               -1, -1, -1, -1, -1, -1, -1, -1, -1,
                               -1, 1, -1, -1]}
pathways['rtk'] = {'genes': ['EGFR', 'ERBB2', 'ERBB3', 'ERBB4', 'MET', 'PDGFRA', 'FGFR1', 'FGFR2', 'FGFR3', 'FGFR4',
                             'KIT', 'IGF1R', 'RET', 'ROS1', 'ALK', 'FLT3', 'NTRK1', 'NTRK2', 'NTRK3', 'JAK2',
                             'CBL', 'ERRFI1', 'ABL1', 'SOS1', 'NF1', 'RASA1', 'PTPN11', 'KRAS', 'HRAS', 'NRAS',
                             'RIT1', 'ARAF', 'BRAF', 'RAF1', 'RAC1', 'MARK1', 'MAP2K1', 'MAP2K2'],
                   'signs': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                             -1, -1, 1, 1, -1, -1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1]}
pathways['nrf2'] = {'genes': ['KEAP1', 'CUL3', 'NFE2L2'],
                    'signs': [-1, -1, 1]}
pathways['tgfb'] = {'genes': ['TGFBR1', 'TGFBR2', 'ACVR2A', 'ACVR1B', 'SMAD2', 'SMAD3', 'SMAD4'],
                    'signs': [-1, -1, -1, -1, -1, -1, -1]}
pathways['myc'] = {'genes': ['MYC', 'MYCN', 'MAX', 'MGA', 'MXI1', 'MNT', 'MLX'],
                   'signs': [1, 1, -1, -1, -1, -1, -1]}
pathways['cellcycle'] = {'genes': ['CDKN1A', 'CDKN1B', 'CDKN2A', 'CDKN2B', 'CDKN2C', 'CCNE1', 'CCND1', 'CCND2',
                                   'CCND3', 'CDK2', 'CDK4', 'CDK6', 'RB1', 'E2F1', 'E2F3'],
                         'signs': [-1, -1, -1, -1, -1, 1, 1, 1,
                                   1, 1, 1, 1, -1, 1, 1]}
pathways['hippo'] = {'genes': ['DCHS1', 'DCHS2', 'FAT1', 'FAT2', 'FAT3', 'FAT4', 'TAOK1', 'TAOK2', 'TAOK3',
                               'SAV1', 'STK3', 'STK4', 'LATS1', 'LATS2', 'MOB1A', 'MOB1B', 'NF2', 'WWC1',
                               'CRB1', 'CRB2', 'YAP1', 'PTPN14', 'CSNK1E', 'CSNK1D', 'TEAD2'],
                     'signs': [-1, -1, -1, -1, -1, -1, -1, -1, -1,
                               -1, -1, -1, -1, -1, -1, -1, -1, -1,
                               -1, -1, 1, -1, -1, -1, 1]}

dataset = 'TCGA-LUAD'
data_cna = pd.read_pickle('../data/{:s}/data_cna_shared.pickle'.format(dataset))
data_expression = pd.read_pickle('../data/{:s}/data_expression_shared.pickle'.format(dataset))

data_split = np.load('../data/{:s}/data_split.npy'.format(dataset), allow_pickle=True).item()
save_dir = '../data/{:s}/data_for_train'.format(dataset)
os.makedirs(save_dir, exist_ok=True)


def main():
    pathway_omics_data = extract_pathway_data()
    features = extract_pathway_feature(pathway_omics_data)
    labels_exp = generate_labels(features['expression'])
    labels_cna = generate_labels(features['cna'])
    split_labels_into_train_val_test(labels_exp, save_dir, 'pathway_expression')
    split_labels_into_train_val_test(labels_cna, save_dir, 'pathway_cna')


def extract_pathway_data():
    pathway_omics_data = {}
    for k, data in pathways.items():
        genes, signs = data['genes'], data['signs']
        print('Extracting {:s} pathway data...'.format(k))
        exist_genes = []
        exist_gene_signs = []
        for gene, sign in zip(genes, signs):
            if gene in list(data_cna.columns):
                exist_genes.append(gene)
                exist_gene_signs.append(sign)
            else:
                print('\tMissing {:s}'.format(gene))
        data_cna_k = data_cna[exist_genes]
        data_expression_k = data_expression[exist_genes]

        pathway_omics_data[k] = {'cna': data_cna_k, 'expression': data_expression_k,
                                 'sign': exist_gene_signs}

        # data_cna_k.to_csv('{:s}/cna_{:s}.txt'.format(save_dir, k), header=True, index=True, sep='\t')
        # data_expression_k.to_csv('{:s}/expression_{:s}.txt'.format(save_dir, k), header=True, index=True, sep='\t')
    return pathway_omics_data


def extract_pathway_feature(pathway_omics_data):
    expression_features = {}
    cna_features = {}
    for pathway_name, data in pathway_omics_data.items():
        data_exp, data_cna, sign = data['expression'], data['cna'], data['sign']
        expression_features[pathway_name] = (data_exp * sign).mean(axis=1).to_dict()
        cna_features[pathway_name] = (data_cna * sign).mean(axis=1).to_dict()

        # plt.hist(list(cna_features[k].values()), bins='auto')
        # plt.xlabel("mean Z-score")
        # plt.ylabel("Frequency")
        # plt.savefig('./hist/cna/hist_{:s}.png'.format(k))
        # plt.close()

    features = {'expression': expression_features, 'cna': cna_features}
    return features


def generate_labels(features):
    labels_all = {}
    for pathway_name, data in features.items():
        labels = {}

        # assign labels according to fixed cutoff values
        for patient_name, val in data.items():
            label = 0 if val < 0 else 1
            labels[patient_name] = label

        labels_all[pathway_name] = labels
    return labels_all


def split_labels_into_train_val_test(labels, data_train_dir, data_type):
    train_patients = [x[:15] for x in data_split['train']]
    test_patients = [x[:15] for x in data_split['test']]

    class_weights = {}
    os.makedirs('{:s}/train'.format(data_train_dir), exist_ok=True)
    os.makedirs('{:s}/test'.format(data_train_dir), exist_ok=True)
    for pathway, data in labels.items():
        class_info_train = {}
        class_info_val = {}
        class_info_test = {}
        for patient_name, label in data.items():
            if patient_name in train_patients:
                class_info_train[patient_name] = label
            elif patient_name in test_patients:
                class_info_test[patient_name] = label

        print('Label distribution in pathway {:s}:'.format(pathway))
        freq = print_distribution(data)
        freq_train = print_distribution(class_info_train)
        freq_test = print_distribution(class_info_test)

        class_weights[pathway] = 1/freq_train

        np.save('{:s}/train/{:s}_class_info_{:s}.npy'.format(data_train_dir, data_type, pathway), class_info_train)
        np.save('{:s}/test/{:s}_class_info_{:s}.npy'.format(data_train_dir, data_type, pathway), class_info_test)
    np.save('{:s}/class_weights_{:s}.npy'.format(data_train_dir, data_type), class_weights)


def print_distribution(data):
    n1, n2 = 0, 0
    for k, v in data.items():
        if v == 0:
            n1 += 1
        elif v == 1:
            n2 += 1

    freq = np.array([n1, n2])
    freq = freq / np.sum(freq)
    print('0: {:.4f}\t1: {:.4f}'.format(freq[0], freq[1]))
    return freq


if __name__ == '__main__':
    main()
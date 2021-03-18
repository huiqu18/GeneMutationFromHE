import os
import pandas as pd
import numpy as np


def main():
    dataset = 'TCGA-BRCA'
    omics_data_dir = '../data/cbioportal/{:s}_tcga_pan_can_atlas_2018'.format(dataset.split('-')[1].lower())
    save_dir = '../data/{:s}'.format(dataset)

    shared_patients = np.load('{:s}/shared_patients.npy'.format(omics_data_dir))
    get_mutation_info('../data/cbioportal/{:s}_mutation_matrix.txt'.format(dataset.split('-')[1].lower()), shared_patients, save_dir)
    get_cna_info('../data/cbioportal/{:s}_cna_matrix.txt'.format(dataset.split('-')[1].lower()), shared_patients, save_dir)


def get_mutation_info(filename, sample_names, save_dir):
    # read mutation data from file
    mutation_data = pd.read_csv(filename, delimiter='\t')
    mutation_data.rename(columns={'studyID:sampleId': 'Sample_ID'}, inplace=True)
    for i in range(mutation_data.shape[0]):
        mutation_data.at[i, 'Sample_ID'] = mutation_data.at[i, 'Sample_ID'][-15:]
    # extract mutation info
    pat_mutation_data = mutation_data.loc[mutation_data['Sample_ID'].isin(sample_names)]

    # remove genes with mutation percentage < 3%
    N = pat_mutation_data.shape[0]
    print('Number of samples: {:d}'.format(N))
    gene_names = list(pat_mutation_data.columns[2:])
    for gene in gene_names:
        N_not_mutated = (pat_mutation_data[gene] == 0).sum()
        mut_percent = 1 -  N_not_mutated / N
        print('{:s}\t{:.2f} %\t{:d}'.format(gene, mut_percent * 100., N - N_not_mutated))
        # if mut_percent < 0.03:
        #     pat_mutation_data = pat_mutation_data.drop(gene, 1)

    pat_mutation_data = pat_mutation_data.drop('Altered', 1)
    pat_mutation_data.to_pickle('{:s}/patient_mutation.pickle'.format(save_dir))


def get_cna_info(filename, sample_names, save_dir):
    # read cn data from file
    cna_data = pd.read_csv(filename, delimiter='\t')
    cna_data.rename(columns={'studyID:sampleId': 'Sample_ID'}, inplace=True)
    for i in range(cna_data.shape[0]):
        cna_data.at[i, 'Sample_ID'] = cna_data.at[i, 'Sample_ID'][-15:]
    # extract cna info
    pat_cna_data = cna_data.loc[cna_data['Sample_ID'].isin(sample_names)]

    # remove genes with cna percentage < 5%
    N = pat_cna_data.shape[0]
    gene_names = list(pat_cna_data.columns[2:])
    for gene in gene_names:
        N_not_altered = (pat_cna_data[gene] == 0).sum()
        mut_percent = 1 - N_not_altered / N
        print('{:s}\t{:.2f} %\t{:d}'.format(gene, mut_percent * 100., N - N_not_altered))
        # if mut_percent < 0.05:
        #     pat_cna_data = pat_cna_data.drop(gene, 1)

    pat_cna_data = pat_cna_data.drop('Altered', 1)
    pat_cna_data.to_pickle('{:s}/patient_cna.pickle'.format(save_dir))


if __name__ == '__main__':
    main()

import os
import pandas as pd
import numpy as np


def main():
    dataset = 'TCGA'
    omics_data_dir = '../data/cbioportal/{:s}_tcga_pan_can_atlas_2018'.format(dataset.lower())
    save_dir = '../data/{:s}'.format(dataset)

    shared_patients = np.load('{:s}/shared_patients.npy'.format(omics_data_dir))
    get_mutation_info('../data/cbioportal/mutations_{:s}.txt'.format(dataset.lower()), shared_patients, save_dir)
    get_cna_info('../data/cbioportal/cna_{:s}.txt'.format(dataset.lower()), shared_patients)


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


# def get_PARADIGM_pathway_info():
#     import matplotlib.pyplot as plt
#     pathway_data = pd.read_pickle('../data/omics/ipl_pathway_median.pickle')
#
#     pathway_data = pathway_data.T
#     pathway_data.columns = pathway_data.iloc[0].apply(int)
#     pathway_data = pathway_data[1:]
#     pathway_data = pathway_data.sort_index(axis=0)
#     pathway_data['Patient_ID'] = pathway_data.index
#     cols = pathway_data.columns.to_list()
#     cols = cols[-1:] + cols[:-1]
#     pathway_data = pathway_data[cols]
#
#     # read slides names
#     slides_list = pd.read_csv('/data/hui/gene-predict-v2/data/slide_selection_step4_659-659.txt', header=None)
#     slides_list = list(slides_list[0].values)
#     patient_names = [x[:12] for x in slides_list]
#
#     # extract cna info of 659 patients
#     pat_pathway_data = pathway_data.loc[pathway_data['Patient_ID'].isin(patient_names)]
#
#     # pathway_names = list(pat_pathway_data.columns[1:])
#     # for pathway_name in pathway_names:
#     #     plt.hist(pat_pathway_data[pathway_name], bins='auto')
#     #     plt.savefig('./hists_median/pathway_ipl_{:d}.png'.format(int(pathway_name)))
#     #     plt.close()
#
#     pat_pathway_data.to_pickle('../data/patient_pathway_median.pickle')
#
#
# def get_survival_info():
#     # read clinical data from file
#     clinical_data = pd.read_csv('../data/data_clinical_patient.txt', delimiter='\t')
#     clinical_data = clinical_data.iloc[4:, :]
#     clinical_data.rename(columns={'#Patient Identifier': 'Patient_ID',
#                                   'Overall Survival Status': 'Survival_Censor',
#                                   'Overall Survival (Months)': 'Survival_Months'}, inplace=True)
#
#     # read slides names
#     slides_list = pd.read_csv('/data/hui/gene-predict-v2/data/slide_selection_step4_659-659.txt', header=None)
#     slides_list = list(slides_list[0].values)
#     patient_names = [x[:12] for x in slides_list]
#
#     # subtypes and survival
#     pat_survival_data = clinical_data[['Patient_ID', 'Subtype', 'Survival_Months', 'Survival_Censor']]
#     pat_survival_data = pat_survival_data.loc[pat_survival_data['Patient_ID'].isin(patient_names)]
#     pat_survival_data.to_pickle('../data/patient_survival.pickle')


if __name__ == '__main__':
    main()

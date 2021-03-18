import pandas as pd
from tqdm import tqdm
import numpy as np
from functools import reduce


dataset = 'TCGA-BRCA'
root_data_dir = '../data/{:s}'.format(dataset)
slides_all = pd.read_csv('{:s}/slide_selection_step1.txt'.format(root_data_dir), header=None)
slide_names = list(slides_all[0])
patient_names = [slide_name[:15] for slide_name in slide_names]

omics_data_dir = '../data/cbioportal/{:s}_tcga_pan_can_atlas_2018'.format(dataset.split('-')[1].lower())


def main():
    get_cna(patient_names)
    get_gene_expression(patient_names)
    extract_shared_patients_and_genes()

    shared_patients = np.load('{:s}/shared_patients.npy'.format(root_data_dir))

    slides_final = [x for x in slide_names if x[:15] in shared_patients]
    with open('{:s}/slide_selection_final.txt'.format(root_data_dir), 'w') as file:
        for i in slides_final:
            file.write('{:s}\n'.format(i))

    diff_patients = np.setdiff1d(patient_names, shared_patients)
    print(diff_patients.shape)
    print(diff_patients)


def get_cna(patient_names):
    # copy number variation
    data_cna = pd.read_csv('{:s}/data_log2CNA.txt'.format(omics_data_dir), delimiter='\t')
    data_cna = data_cna.dropna()
    data_cna = data_cna.set_index('Hugo_Symbol').transpose().drop('Entrez_Gene_Id')
    data_cna['patient_name'] = data_cna.index
    data_cna.to_pickle('{:s}/data_log2cna_original.pickle'.format(root_data_dir))

    data_cna_selected = data_cna.loc[data_cna['patient_name'].isin(patient_names)]
    data_cna_selected.to_pickle('{:s}/data_log2cna_step2.pickle'.format(root_data_dir))


def get_gene_expression(patient_names):
    # gene expression
    data_exp = pd.read_csv('{:s}/data_RNA_Seq_v2_mRNA_median_Zscores.txt'.format(omics_data_dir), delimiter='\t')
    data_exp = data_exp.dropna()
    data_exp = data_exp.set_index('Hugo_Symbol').transpose().drop('Entrez_Gene_Id')
    data_exp['patient_name'] = data_exp.index
    data_exp.to_pickle('{:s}/data_expression_original.pickle'.format(root_data_dir))

    data_exp_selected = data_exp.loc[data_exp['patient_name'].isin(patient_names)]
    data_exp_selected.to_pickle('{:s}/data_expression_step2.pickle'.format(root_data_dir))


def extract_shared_patients_and_genes():
    # load data
    data_cna_selected = pd.read_pickle('{:s}/data_log2cna_step2.pickle'.format(root_data_dir))
    data_expression_selected = pd.read_pickle('{:s}/data_expression_step2.pickle'.format(root_data_dir))

    shared_patients = reduce(np.intersect1d, (list(data_cna_selected['patient_name']),
                                              list(data_expression_selected['patient_name'])))
    shared_genes = reduce(np.intersect1d, (list(data_cna_selected.columns),
                                           list(data_expression_selected.columns)))
    print('{:d} x {:d}'.format(len(shared_patients), len(shared_genes)))

    data_cna = data_cna_selected.loc[data_cna_selected['patient_name'].isin(shared_patients)]
    data_cna = data_cna[shared_genes].copy()
    data_expression = data_expression_selected.loc[data_expression_selected['patient_name'].isin(shared_patients)]
    data_expression = data_expression[shared_genes].copy()

    # remove duplicated columns
    data_cna = data_cna.loc[:, ~data_cna.columns.duplicated()]
    data_expression = data_expression.loc[:, ~data_expression.columns.duplicated()]

    print(data_cna.shape)
    print(data_expression.shape)

    # save data
    data_cna.to_pickle('{:s}/data_cna_shared.pickle'.format(root_data_dir))
    data_expression.to_pickle('{:s}/data_expression_shared.pickle'.format(root_data_dir))
    np.save('{:s}/shared_patients.npy'.format(root_data_dir), shared_patients)
    np.save('{:s}/shared_genes.npy'.format(root_data_dir), shared_genes)


if __name__ == '__main__':
    main()

import numpy as np
import pandas as pd
import math


def main():
    dataset = 'TCGA-BRCA'
    omics_data_dir = '../data/cbioportal/{:s}_tcga_pan_can_atlas_2018'.format(dataset.lower())

    clinical_data = pd.read_csv('{:s}/data_clinical_patient.txt'.format(omics_data_dir), delimiter='\t')
    clinical_data = clinical_data.iloc[4:, :]
    data_split = np.load('../data/{:s}/data_split.npy'.format(dataset), allow_pickle=True).item()

    train_patient_names = [x[:12] for x in data_split['train']]
    val_patient_names = [x[:12] for x in data_split['val']]
    test_patient_names = [x[:12] for x in data_split['test']]
    all_patient_names = train_patient_names + val_patient_names + test_patient_names
    # all_patient_names = train_patient_names + test_patient_names

    clinical_data_train = clinical_data.loc[clinical_data['#Patient Identifier'].isin(train_patient_names)]
    clinical_data_val = clinical_data.loc[clinical_data['#Patient Identifier'].isin(val_patient_names)]
    clinical_data_test = clinical_data.loc[clinical_data['#Patient Identifier'].isin(test_patient_names)]
    clinical_data_all = clinical_data.loc[clinical_data['#Patient Identifier'].isin(all_patient_names)]

    print('Patient characterisitcs in the whole set:')
    print_characteristics(clinical_data_all, dataset)
    print('Patient characterisitcs in the train set:')
    print_characteristics(clinical_data_train, dataset)
    print('Patient characterisitcs in the val set:')
    print_characteristics(clinical_data_val, dataset)
    print('Patient characterisitcs in the test set:')
    print_characteristics(clinical_data_test, dataset)


def print_characteristics(data, dataset):
    N = data.shape[0]
    Age_data = data['Diagnosis Age'].values

    Ages = []
    for x in Age_data:
        if isinstance(x, float) and math.isnan(x):
            continue
        Ages.append(int(x))

    # Age_data = [int(x) for x in Age_data]
    print('\tAge: mean (min, max): {:.1f} ({:d}, {:d})'.format(np.mean(np.array(Ages)), min(Ages), max(Ages)))

    Sex_data = data['Sex'].values
    N_male, N_female = np.sum(Sex_data == 'Male'), np.sum(Sex_data == 'Female')
    print('\tSex: Male Female: {:d} ({:.1f}%) {:d} ({:.1f}%)'.format(N_male, N_male/N*100, N_female, N_female/N*100))

    Stage_data = data['Neoplasm Disease Stage American Joint Committee on Cancer Code'].values
    N_1 = np.sum(np.isin(Stage_data, ['STAGE I', 'STAGE IA', 'STAGE IB']))
    N_2 = np.sum(np.isin(Stage_data, ['STAGE II', 'STAGE IIA', 'STAGE IIB']))
    N_3 = np.sum(np.isin(Stage_data, ['STAGE III', 'STAGE IIIA', 'STAGE IIIB', 'STAGE IIIC']))
    N_4 = np.sum(Stage_data == 'STAGE IV')
    N_10 = np.sum(Stage_data == 'STAGE X')
    N_NA = np.sum([1 if isinstance(x, float) and np.isnan(x) else 0 for x in Stage_data])
    print('\tStage:\n\t\tI/IA/IB: {:d} ({:.1f}%)\n\t\tII/IIA/IIB: {:d} ({:.1f}%)\n'
          '\t\tIII/IIIA/IIIB/IIIC: {:d} ({:.1f}%)\n\t\tIV: {:d} ({:.1f}%)\n\t\tX: {:d} ({:.1f})\n'
          '\t\tN/A: {:d} ({:.1f}%)'.format(N_1, N_1/N*100, N_2, N_2/N*100, N_3, N_3/N*100, N_4, N_4/N*100,
                                           N_10, N_10/N*100, N_NA, N_NA/N*100))

    if dataset == 'BRCA':
        Subtype_data = data['Subtype'].values
        N_LumA = np.sum(Subtype_data == 'BRCA_LumA')
        N_LumB = np.sum(Subtype_data == 'BRCA_LumB')
        N_Her2 = np.sum(Subtype_data == 'BRCA_Her2')
        N_Basal = np.sum(Subtype_data == 'BRCA_Basal')
        N_Normal = np.sum(Subtype_data == 'BRCA_Normal')
        N_NA = np.sum([1 if isinstance(x, float) and np.isnan(x) else 0 for x in Subtype_data])
        print('\tSubtype:\n\t\tLumA: {:d} ({:.1f}%)\n\t\tLumB: {:d} ({:.1f}%)\n\t\tHer2: {:d} ({:.1f}%)'
              '\n\t\tBasal: {:d} ({:.1f}%)\n\t\tNormal: {:d} ({:.1f}%)\n\t\tN/A: {:d} ({:.1f}%)'
              .format(N_LumA, N_LumA/N*100, N_LumB, N_LumB/N*100, N_Her2, N_Her2/N*100, N_Basal, N_Basal/N*100,
                      N_Normal, N_Normal/N*100, N_NA, N_NA/N*100))


if __name__ == '__main__':
    main()

# Mutation from H&E in breast cancer

This repository contains the code for the following manuscript:

Genetic Mutation and Biological Pathway Prediction based on Whole Slide Images in Breast Carcinoma using Deep Learning, submitted to <i>Nature Precision Oncology</i> for review.


## Introduction
Breast carcinoma is the most common cancer among women worldwide that consists of a heterogeneous group of subtype diseases. The whole slide images (WSIs) can capture the cell-level heterogeneity, and are routinely used for cancer diagnosis by pathologists. However, key driver genetic mutations related to targeted therapies are identified by genomic analysis like high-throughput molecular profiling. In this study, we developed a deep learning model to predict the genetic mutations and biological pathway activities directly from WSIs. Using the histopathology images from the Genomic Data Commons Database, our model can predict the point mutations of six important genes (AUC 0.68-0.85) and copy number alteration of another six genes (AUC 0.69-0.79). Additionally, the trained models can predict the activities of three out of ten canonical pathways (AUC 0.65-0.79). Next, we visualized the weight maps of tumor tiles in WSI to understand the decision making process of deep learning models. We further validated our models on the liver and lung cancers that are related to metastatic breast cancer. Our results provide new insights into the association between pathological image features, molecular outcomes and targeted therapies for breast cancer patients.



## Dependencies

Pytorch 1.1.0

Torchvision 0.3.0

openslide 3.4.1

openslide-python 1.1.0

numpy 1.16.4

pandas 1.0.5

scikit-image 0.15.0

scikit-learn 0.21.3

Pillow 6.1.0

h5py 2.9.0

matplotlib 2.2.2

tqdm 4.32.1



## Usage

### Step 1. Data download

Download the FFPE whole slide images from GDC portal (https://portal.gdc.cancer.gov/) for breast carcinoma (TCGA-BRCA), lung adenocarcinoma (TCGA-LUAD), and liver hepatocellular carcinoma (TCGA-LIHC).

Download corresponding omics data (gene mutation, cna, expression) from cBioPortal (https://www.cbioportal.org/) for each dataset.

### Step 2. Data processing

Using the code under `code_data_processing` to perform

- Slide selection: select high quality WSIs from the original dataset 
- Tile extraction: extract 512x512 tiles from the large WSI at 20x magnification and discard background tiles
- Tumor tile selection: use k-means clustering and manual revision to select tumor tiles
- Color normalization
- Feature extraction: extract a feature vector for each tile using pretrained ResNet-101
- Data split: split the data into train, validation and test sets
- Patients characteristics calculation

Details can be found in the paper and code_data_processing.

### Step 3. Model training for mutation and pathway activity prediction on TCGA-BRCA

Using the code under `code_prediction_on_breast_cancer` to train models to predict point mutation, copy number alteration and pathway activities for breast patients.

Details can be found in the paper and code_prediction_on_breast_cancer.

### Step 4. Model validation (finetuning) for mutation and pathway activity prediction on TCGA-LUAD and TCGA-LIHC

Using the code under `code_validation_on_lung_and_liver` to train models to predict point mutation, copy number alteration and pathway activities for breast patients.

Details can be found in the paper and code_validation_on_lung_and_liver.

## Note

Some intermediate data are put into the folder `data`.
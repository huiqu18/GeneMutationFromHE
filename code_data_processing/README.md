# Data processing steps


## Step 1. Slide selection
* Get slide date and thumbnail for each slide
    ```
    python  1-1.get_slide_data.py
    python  1-1.get_thumbnails.py
    ```
* Read and process omics data to remove non-useful slides: extract expression and copy number variation 
from raw data, remove slides that has no omics data.
    ```
    python 1-2.prepare_omics_data.py
    ```
* Get size and mag information of selected slides
    ```
    python 1-3.get_size_info.py
    ```
* Get mutation and cna information of selected slides
    ```
    python 1-4.get_mutation_info.py
    ```
 
## Step 2. Patch extraction and selection
* Extract patches
    ```
    python 2-1.extract_patches.py
    ```
* Perform k-means clustering
    ```
    python 2-2.k_means_clustering.py
    ```
* Refine the clustering results manually
* Fuse the clustering results with manual annotation
    ```
    python 2-4.fuse_clustering_and_manual.py
    ```

## Step 3. Color norm and feature extraction
* Color normalization
    ``` 
    python 3-1.color_norm.py
    ```
* Extract features for each patch
    ```
    python 3-2.feature_extraction.py
    ```

## Step 4. Data split
``` 
python 4.split_dataset.py 
```

## Step 5. Get the characteristics of all patients and patients in each set
```
python 5.get_patient_characteristics.py
```


<!-- ##Slide selection
-[x] Step 1: Select high quality slides, resulting in 681 slides / 669 patients
-[x] Step 2: Retrieve expression and copy number variation from raw data
-[x] Step 3: Remove slides that has no omics data, resulting in 671 slides / 659 patients
-[x] Step 4: Keep one slide per patient, resulting in 659 slides/patients


## Patch selection
-[x] Step 1: Extract 512x512 patches from 20x images, and remove background patches
-[x] Step 2: Initial k-means clustering
-[x] Step 3: Manual refinement
-[x] Step 4: Fuse clustering and manual annotation

## Patch feature extraction
-[x] Step 1: Color normalization
-[x] Step 2: ResNet-101 feature extraction


## Patient information extraction
-[x] Download mutation data and survival data from cBioPortal
-[x] Get patient info (gene mutation, copy number variation, overall survival, subtypes) from data


## Train, val, test split
-[x] Split the slides/patients into train, val and test sets randomly with 70%, 15%, 15%.
-[x] Put features into hdf5 file to accelerate the speed of data loading
>
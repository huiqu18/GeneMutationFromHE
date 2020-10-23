# Mutation status and pathway activity prediction on breast cancer

After the data processing steps, the extracted features of pathology slides should be stored 
under the `<root_data_dir>/20x_features_resnet101_hdf5`. Before training, we need to generate 
labels.


### Generate labels for point mutation and copy number alteration prediction tasks
Set the data_type in `generate_mutation_labels.py` as mutation or cna, ensure all necessary files
are included in `root_data_dir`, then run the script
```bash
python generate_mutation_labels.py
```

### Generate data and labels for pathway activity prediction tasks
There are two types of pathway activity labels, one from the mRNA expression data and the other from
the copy number alteration data. The pathway activity in each patient is obtained by a weighted sum of 
the genes’ expression data or CNA data in the pathway. Then the activity was binarized as activated 
if it is greater than zero and inactivated otherwise.
```bash
python generate_pathway_labels.py
```

### Training and test
Before training or testing the model, set the options in ```options.py```. Most options are set as default values, 
and a part of them can be also parsed from the command line. An exmaple of running the experiments for mutation
prediction and pathway prediction can be found in `run_mutation.sh` and `run_pathway.sh`, respectively.

### Visualization of weight maps
After testing with the attention weights saved, we can visualize the weight maps by running `visualization.py`.
If you want to save the top weighted tiles/patches, the path to individual patches `img_data_dir` should be provided.
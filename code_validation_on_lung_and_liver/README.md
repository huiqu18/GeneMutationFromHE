# Validation of trained models using breast cancer data on lung and liver cancers

To perform the validation, lung and liver data should be processed as the breast cancer data.

Similarly, the mutation and pathway labels can be generated from `generate_mutation_labels.py` and `generate_pathway_labels.py`.

### Training and test
Before training or testing the model, set the options in ```options.py```. Most options are set as default values, 
and a part of them can be also parsed from the command line. An exmaple of running the experiments for mutation
prediction and pathway prediction can be found in `run_mutation.sh` and `run_pathway.sh`, respectively.

### Visualization of weight maps
After testing with the attention weights saved, we can visualize the weight maps by running `visualization.py`.
If you want to save the top weighted tiles/patches, the path to individual patches `img_data_dir` should be provided.
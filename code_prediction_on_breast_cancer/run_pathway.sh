#!/usr/bin/env bash

dataset=TCGA-BRCA
gpu=2
for data_type in pathway_expression pathway_cna
do
  for pathway in 'pi3k' 'p53' 'notch' 'rtk' 'nrf2' 'wnt' 'tgfb' 'myc' 'cellcycle' 'hippo'
  do
    python train.py --data-type ${data_type} --random-seed -1 --model AttnClassifier --gene-or-pathway ${pathway} --batch-size 8 \
      --gpus ${gpu} --save-dir ./experiments/${dataset}/${data_type}/${pathway}
    python test.py --data-type ${data_type} --model AttnClassifier --gene-or-pathway ${pathway} --batch-size 8 --gpus ${gpu} \
      --model-path ./experiments/${dataset}/${data_type}/${pathway}/checkpoint_best.pth.tar  \
      --save-dir ./experiments/${dataset}/${data_type}/${pathway}/test_best
  done
done


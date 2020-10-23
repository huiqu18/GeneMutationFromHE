#!/usr/bin/env bash


# ----- LUAD ----- #
dataset=TCGA-LUAD
gpu=1
for data_type in pathway_expression pathway_cna
do
  for pathway in 'pi3k' 'p53' 'notch' 'rtk' 'nrf2' 'wnt' 'tgfb' 'myc' 'cellcycle' 'hippo'
  do
    python train.py --dataset ${dataset} --root-data-dir ../data/${dataset} --data-type ${data_type} \
      --random-seed -1 --model AttnClassifier --gene-or-pathway ${pathway} --batch-size 8 --gpus ${gpu}  \
      --data-dir ../data/${dataset}/data_for_train --save-dir ./experiments/${dataset}/${data_type}/${pathway} \
      --checkpoint-path ../code_prediction_on_breast_cancer/experiments/TCGA-BRCA/${data_type}/${pathway}/checkpoint_best.pth.tar

    python test.py --dataset ${dataset} --root-data-dir ../data/${dataset} --data-type ${data_type} \
      --model AttnClassifier --gene-or-pathway ${pathway} --batch-size 8 --gpus ${gpu} \
      --data-dir ../data/${dataset}/data_for_train --save-dir ./experiments/${dataset}/${data_type}/${pathway}/test_best \
      --model-path ./experiments/${dataset}/${data_type}/${pathway}/checkpoint_30.pth.tar
  done
done


# ----- LIHC ----- #
dataset=TCGA-LIHC
gpu=1
for data_type in pathway_expression pathway_cna
do
  for pathway in 'pi3k' 'p53' 'notch' 'rtk' 'nrf2' 'wnt' 'tgfb' 'myc' 'cellcycle' 'hippo'
  do
    python train.py --dataset ${dataset} --root-data-dir ../data/${dataset} --data-type ${data_type} \
      --random-seed -1 --model AttnClassifier --gene-or-pathway ${pathway} --batch-size 8 --gpus ${gpu}  \
      --data-dir ../data/${dataset}/data_for_train --save-dir ./experiments/${dataset}/${data_type}/${pathway} \
      --checkpoint-path ../code_prediction_on_breast_cancer/experiments/TCGA-BRCA/${data_type}/${pathway}/checkpoint_best.pth.tar

    python test.py --dataset ${dataset} --root-data-dir ../data/${dataset} --data-type ${data_type} \
      --model AttnClassifier --gene-or-pathway ${pathway} --batch-size 8 --gpus ${gpu} \
      --data-dir ../data/${dataset}/data_for_train --save-dir ./experiments/${dataset}/${data_type}/${pathway}/test_best \
      --model-path ./experiments/${dataset}/${data_type}/${pathway}/checkpoint_30.pth.tar
  done
done

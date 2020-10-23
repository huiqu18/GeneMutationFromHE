#!/usr/bin/env bash

# ----- LUAD ----- #
dataset=TCGA-LUAD
data_type=mutation
gpu=0
for gene in 'ARID1A' 'KMT2C' 'NF1' 'NOTCH2' 'PIK3CA'   'RB1'  'RYR2'  'TP53' 'USH2A'
do
  python train.py --dataset ${dataset} --root-data-dir ../data/${dataset} --data-type ${data_type} \
    --random-seed -1 --model AttnClassifier --gene-or-pathway ${gene} --batch-size 8 --gpus ${gpu} \
    --data-dir ../data/${dataset}/data_for_train --save-dir ./experiments/${dataset}/${data_type}/${gene} \
    --checkpoint-path ../code_prediction_on_breast_cancer/experiments/TCGA-BRCA/${data_type}/${gene}/checkpoint_best.pth.tar

  python test.py --dataset ${dataset} --root-data-dir ../data/${dataset} --data-type ${data_type} \
    --model AttnClassifier --gene-or-pathway ${gene} --batch-size 8 --gpus ${gpu} \
    --data-dir ../data/${dataset}/data_for_train --save-dir ./experiments/${dataset}/${data_type}/${gene}/test_best \
    --model-path ./experiments/${dataset}/${data_type}/${gene}/checkpoint_30.pth.tar
done

dataset=TCGA-LUAD
data_type=cna
gpu=0
for gene in 'APH1A' 'CDKN2A' 'CDKN2B' 'EIF4EBP1' 'FGFR1' 'HEY1' 'KAT6A' 'MCL1' 'MYC' 'NCSTN' 'NOTCH2' 'PTK2' 'RAB25' 'RIT1'
do
  python train.py --dataset ${dataset} --root-data-dir ../data/${dataset} --data-type ${data_type} \
    --random-seed -1 --model AttnClassifier --gene-or-pathway ${gene} --batch-size 8 --gpus ${gpu} \
    --data-dir ../data/${dataset}/data_for_train --save-dir ./experiments/${dataset}/${data_type}/${gene} \
    --checkpoint-path ../code_prediction_on_breast_cancer/experiments/TCGA-BRCA/${data_type}/${gene}/checkpoint_best.pth.tar

  python test.py --dataset ${dataset} --root-data-dir ../data/${dataset} --data-type ${data_type} \
    --model AttnClassifier --gene-or-pathway ${gene} --batch-size 8 --gpus ${gpu} \
    --data-dir ../data/${dataset}/data_for_train --save-dir ./experiments/${dataset}/${data_type}/${gene}/test_best \
    --model-path ./experiments/${dataset}/${data_type}/${gene}/checkpoint_30.pth.tar
done


# ----- LIHC ----- #
dataset=TCGA-LIHC
data_type=mutation
gpu=0
for gene in 'ARID1A' 'KMT2C' 'NF1' 'RB1'  'RYR2'  'TP53' 'USH2A'
do
  python train.py --dataset ${dataset} --root-data-dir ../data/${dataset} --data-type ${data_type} \
    --random-seed -1 --model AttnClassifier --gene-or-pathway ${gene} --batch-size 8 --gpus ${gpu} \
    --data-dir ../data/${dataset}/data_for_train --save-dir ./experiments/${dataset}/${data_type}/${gene} \
    --checkpoint-path ../code_prediction_on_breast_cancer/experiments/TCGA-BRCA/${data_type}/${gene}/checkpoint_best.pth.tar

  python test.py --dataset ${dataset} --root-data-dir ../data/${dataset} --data-type ${data_type} \
    --model AttnClassifier --gene-or-pathway ${gene} --batch-size 8 --gpus ${gpu} \
    --data-dir ../data/${dataset}/data_for_train --save-dir ./experiments/${dataset}/${data_type}/${gene}/test_best \
    --model-path ./experiments/${dataset}/${data_type}/${gene}/checkpoint_30.pth.tar
done

dataset=TCGA-LIHC
data_type=cna
gpu=0
for gene in 'AKT3' 'APH1A' 'CCND1' 'CDKN2A' 'CDKN2B' 'E2F5' 'EIF4EBP1' 'FGFR1' 'HEY1' 'IRF2BP2' 'KAT6A' 'MCL1' \
            'MDM4' 'MMP16' 'MYC' 'NCSTN' 'NOTCH2' 'PARP1' 'PSEN2' 'PTK2' 'RAB25' 'RIT1' 'RYR2' 'TGFB2' 'USH2A'
do
  python train.py --dataset ${dataset} --root-data-dir ../data/${dataset} --data-type ${data_type} \
    --random-seed -1 --model AttnClassifier --gene-or-pathway ${gene} --batch-size 8 --gpus ${gpu} \
    --data-dir ../data/${dataset}/data_for_train --save-dir ./experiments/${dataset}/${data_type}/${gene} \
    --checkpoint-path ../code_prediction_on_breast_cancer/experiments/TCGA-BRCA/${data_type}/${gene}/checkpoint_best.pth.tar

  python test.py --dataset ${dataset} --root-data-dir ../data/${dataset} --data-type ${data_type} \
    --model AttnClassifier --gene-or-pathway ${gene} --batch-size 8 --gpus ${gpu} \
    --data-dir ../data/${dataset}/data_for_train --save-dir ./experiments/${dataset}/${data_type}/${gene}/test_best \
    --model-path ./experiments/${dataset}/${data_type}/${gene}/checkpoint_30.pth.tar
done
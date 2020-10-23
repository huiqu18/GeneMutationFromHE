#!/usr/bin/env bash

dataset=TCGA-BRCA
data_type=mutation
gpu=0
for gene in 'ARID1A' 'CDH1' 'CTCF' 'ERBB2' 'GATA3' 'KMT2C' 'MAP2K4' 'MAP3K1' 'NF1' 'NOTCH2' 'PIK3CA' 'PTEN' \
  'RB1' 'RUNX1' 'RYR2' 'TBX3' 'TP53' 'USH2A'
do
  python train.py --dataset ${dataset} --root-data-dir ../data/${dataset} --data-type ${data_type} \
    --random-seed -1 --model AttnClassifier --gene-or-pathway ${gene} --batch-size 8 --gpus ${gpu} \
    --data-dir ../data/${dataset}/data_for_train --save-dir ./experiments/${dataset}/${data_type}/${gene}

  python test.py --dataset ${dataset} --root-data-dir ../data/${dataset} --data-type ${data_type} \
    --model AttnClassifier --gene-or-pathway ${gene} --batch-size 8 --gpus ${gpu} \
    --data-dir ../data/${dataset}/data_for_train --save-dir ./experiments/${dataset}/${data_type}/${gene}/test_best \
    --model-path ./experiments/${dataset}/${data_type}/${gene}/checkpoint_best.pth.tar
done

dataset=TCGA-BRCA
data_type=cna
gpu=0
for gene in 'AKT3' 'APH1A' 'BMP7' 'CCND1' 'CDKN2A' 'CDKN2B' 'E2F5' 'EIF4EBP1' 'ERBB2' 'FGFR1' 'HEY1' 'IRF2BP2' \
  'KAT6A' 'MAP3K3' 'MCL1' 'MDM4' 'MMP16' 'MYC' 'NCSTN' 'NOTCH2' 'PAK1' 'PARP1' 'PSEN2' 'PTEN' 'PTK2' 'RAB25' \
  'RIT1' 'RNF43' 'RPS6KB1' 'RPS6KB2' 'RYR2' 'SPOP' 'TGFB2' 'USH2A' 'ZNF217'
do
  python train.py --dataset ${dataset} --root-data-dir ../data/${dataset} --data-type ${data_type} \
    --random-seed -1 --model AttnClassifier --gene-or-pathway ${gene} --batch-size 8 --gpus ${gpu} \
    --data-dir ../data/${dataset}/data_for_train --save-dir ./experiments/${dataset}/${data_type}/${gene}

  python test.py --dataset ${dataset} --root-data-dir ../data/${dataset} --data-type ${data_type} \
    --model AttnClassifier --gene-or-pathway ${gene} --batch-size 8 --gpus ${gpu} \
    --data-dir ../data/${dataset}/data_for_train --save-dir ./experiments/${dataset}/${data_type}/${gene}/test_best \
    --model-path ./experiments/${dataset}/${data_type}/${gene}/checkpoint_best.pth.tar
done

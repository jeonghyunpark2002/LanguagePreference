#!/bin/bash
#SBATCH --job-name  merge_indexes
#SBATCH --time      480:00:00
#SBATCH -c          10
#SBATCH --mem       60G
#SBATCH --constraint ada
#SBATCH --gpus      1
conda activate bergen

INDEX_PATH=../../indexes

python3 merge_indexes.py --dataset_yaml ../../config/dataset/mkqa/mkqa_ko.retrieve_all.yaml --retriever BAAI_bge-m3 --indexes_path $INDEX_PATH

#for lang in ar zh fi fr de ja it ko pt ru es th; do
    #python3 merge_indexes.py --dataset_yaml ../../config/dataset/mkqa/mkqa_${lang}.retrieve_en_${lang}.yaml --retriever BAAI_bge-m3 --indexes_path #$INDEX_PATH
#done
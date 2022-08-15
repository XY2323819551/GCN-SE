#!/usr/bin/env bash

workdir=`realpath ./`
task_name=$1  # STS-B
model_type=$2  # bert; roberta
train_file="train.tsv"
val_file="dev.tsv"
test_file="test.csv"
use_gcn="wo_gcn"
#use_gcn="wi_gcn" # 如果要使用gcn，则此处修改为wi_gcn
incro="cls"
#incro="concat" # 使用gcn时，则此处修改为concat

exp="${workdir}/outputs/${task_name}/${use_gcn}"

# 若使用gcn，需要在参数末尾添加--use_gcn参数
python -m run.main_reg --do_train \
	--task_name ${task_name} \
	--model_path ${workdir}/init_checkpoints/bert-base-uncased \
	--bert_model ${workdir}/init_checkpoints/bert-base-uncased \
	--model_type ${model_type} \
	--train_file parsing_data/toy/${task_name}/${train_file} \
	--val_file parsing_data/toy/${task_name}/${val_file} \
	--test_file parsing_data/toy/${task_name}/${test_file} \
	--device cuda:1 \
	--outdir ${exp} \
	--log ${workdir}/logs/${task_name} \
	--incro ${incro}

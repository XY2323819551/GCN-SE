#!/usr/bin/env bash

workdir=`realpath ./`
task_name=$1  # MRPC
model_type=$2  # bert; roberta
train_file="msr_paraphrase_train.txt"
val_file=""
test_file="msr_paraphrase_test.txt"
use_gcn="wi_gcn"  # 如果要使用gcn，则此处修改为wi_gcn
incro="concat"  #cls、concat 使用gcn时，则此处修改为concat

exp="${workdir}/outputs/${task_name}/${use_gcn}"

# 若使用gcn，需要在参数末尾添加--use_gcn参数
python -m run.main --do_train \
	--task_name ${task_name} \
	--model_path ${workdir}/init_checkpoints/bert-base-uncased \
	--bert_model ${workdir}/init_checkpoints/bert-base-uncased \
	--model_type ${model_type} \
	--label_dim 2 \
	--valset_ratio 0.2 \
	--train_file parsing_data/${task_name}/${train_file} \
	--val_file "" \
	--test_file parsing_data/${task_name}/${test_file} \
	--device cuda:1 \
	--outdir ${exp} \
	--log ${workdir}/logs/${task_name} \
	--incro ${incro} \
	--use_gcn

#!/usr/bin/env bash

workdir=`realpath ./`
task_name=$1  # SICK-R
model_type=$2  # bert; roberta
train_file="SICK_train.txt"
val_file="SICK_trial.txt"
test_file="SICK_test_annotated.txt"
use_gcn="wo_gcn"  # 如果要使用gcn，则此处修改为wi_gcn
#use_gcn="wi_gcn"
incro="cls"  #cls、concat 使用gcn，则此处修改为concat
#incro="concat"

exp="${workdir}/outputs/${task_name}/${use_gcn}"

# 若使用gcn，需要在参数末尾添加--use_gcn参数;若不适用，则去掉此参数
python -m run.main_reg --do_train \
	--task_name ${task_name} \
	--model_path ${workdir}/init_checkpoints/bert-base-uncased \
	--bert_model ${workdir}/init_checkpoints/bert-base-uncased \
	--model_type ${model_type} \
	--train_file parsing_data/SICK/${train_file} \
	--val_file parsing_data/SICK/${val_file} \
	--test_file parsing_data/SICK/${test_file} \
	--device cuda:1 \
	--outdir ${exp} \
	--log ${workdir}/logs/${task_name} \
	--incro ${incro}

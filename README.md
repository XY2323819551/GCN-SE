### Introduction

GCN-SE项目包含两个任务：① 分类任务（对应MRPC & SICK-E数据集）；②回归任务（对应STS-B & SICK-R数据集）。该实验基于上述任务进行，分别对应原始模型和添加了GCN模块的模型进行实验验证，最终在MRPC、SICK-E、SICK-R、STS-B共4个下游任务上相比baseline分别约平均提升了**1%**、**3%**、**4%**、**10%**。其中base模型的结果就可以超越baseline中的 large模型。实验所需数据在[这里]()

### Requirements

需要安装的packages以及对应的版本，`python>=3.6`、`conda install pytorch==1.4.0`

```
transformers==4.6.0
scipy==1.5.2
sklearn
```

### run

**运行的命令train**

```
python main-gcn.py \
	--do_train \
	--task_name choose one task from['MRPC','SICK-E','SICK-R','STS-B'] \
	--model_path model_path(e.g. ./bert-base-uncased) \
	--bert_model model_path(e.g. ./bert-base-uncased) \
	--model_type choose one type from['bert', 'roberta'] \
	--label_dim numbers of classes(only useful for 'MRPC'&'SICK-E') \
	--valset_ratio 0.2(data divide from the training set as the validation set.only useful for 'MPRC') \
	--train_file path to train file \
	--val_file path to val file \
	--test_file path to test file \
	--device cuda:1 or cuda:0(only one GPU)
	--outdir Path to trained models
```



**基于bert-based的分类模型（MRPC、SICK-E）**

- MRPC：二分类。The Microsoft Research Paraphrase Corpus，微软研究院释义语料库。含有标签0和1。数据集包含两句话，每个样本的句子长度都非常长，且数据不均衡，正样本占比68%，负样本仅占32%。MRPC特有的参数是`--valset_ratio 0.2`(因为MRPC没有val set，这个就是从训练集中拿出来一部分当作val set，其他三个数据集都有验证集)，`--model_path`和`--bert_model`的值是一样的，`--model_type`就是`bert` or `roberta`. 运行脚本时需要传入两个参数，①任务名称；②使用的模型。

  ```
  bash run_MRPC.sh MRPC bert
  ```

  tips：如果要使用添加GCN的模型，需要修改脚本中的`use_gcn="wo_gcn"` 为`"wi_gcn"`、`incro="cls"`为`"concat"`，并在运行命令后添加`--use_gcn`，下述任务同理。
  
- SICK-E数据集。SICK-Entailment (SICK-E) 是自然语言蕴含任务， 包含三个类别 (contradiction / entailment / neutral)。

  ```
  bash run_SICK-E.sh SICK-E bert
  ```

- STS-B，求两句话的相似度，回归任务。

  ```
  bash run_STS-B.sh STS-B bert
  ```
  
- SICK-R，SICK-Relatedness (SICK-R) 是语义相关度任务，每个样本包含一对句子，回归任务。

  ```
  bash run_SICK-R.sh SICK-R bert
  ```








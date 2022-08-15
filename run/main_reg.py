import logging
import argparse
import math
import os
import sys
from time import strftime, localtime
import random
import numpy as np
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from transformers import BertModel, BertConfig
from utils import Tokenizer4Bert, ABSADataset
from models.model_reg import Bert, Roberta
from models.model_reg import Bert_gcn, Roberta_gcn
from transformers import RobertaModel, RobertaConfig
from scipy.stats import pearsonr, spearmanr

CONFIG_NAME = 'config.json'
WEIGHTS_NAME = 'pytorch_model.bin'

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


class Instructor:
    def __init__(self, opt):
        self.opt = opt
        tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.bert_model, opt.model_type)
        config = None
        if opt.model_type == 'bert':
            config = BertConfig.from_json_file(os.path.join(opt.bert_model, CONFIG_NAME))
        elif opt.model_type == 'roberta':
            config = RobertaConfig.from_json_file(os.path.join(opt.bert_model, CONFIG_NAME))

        config.num_labels = opt.label_dim
        config.bert_dropout = opt.bert_dropout
        config.device = opt.device
        config.incro = opt.incro
        if opt.use_gcn:
            config.alpha = opt.weight
            if opt.model_type == 'bert':
                self.model = Bert_gcn.from_pretrained(opt.model_path, config=config)
            elif opt.model_type == 'roberta':
                self.model = Roberta_gcn.from_pretrained(opt.model_path, config=config)
        else:
            if opt.model_type == 'bert':
                self.model = Bert.from_pretrained(opt.model_path, config=config)
            elif opt.model_type == 'roberta':
                self.model = Roberta.from_pretrained(opt.model_path, config=config)
        self.model.to(opt.device)
        self.tokenizer = tokenizer
        self.device = opt.device.type
        logger.info(opt)

        self.trainset = ABSADataset(opt.train_file, tokenizer, self.opt)
        self.testset = ABSADataset(opt.test_file, tokenizer, self.opt)
        if os.path.exists(opt.val_file):
            self.valset = ABSADataset(opt.val_file, tokenizer, self.opt)
        elif opt.valset_ratio > 0:
            valset_len = int(len(self.trainset) * opt.valset_ratio)
            self.trainset, self.valset = random_split(self.trainset, (len(self.trainset) - valset_len, valset_len))
        else:
            self.valset = self.testset

        if opt.device.type == 'cuda':
            logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(device=opt.device.index)))

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        logger.info(
            'n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        logger.info('> training arguments:')

        for arg in vars(self.opt):
            logger.info('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

        # save the model parameters 8-31
        saving_path = os.path.join(self.opt.outdir, "model_params.txt")
        with open(saving_path, 'w', encoding='utf-8') as writer:
            writer.write(
                'n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))

    def _reset_params(self):
        for child in self.model.children():
            if type(child) != BertModel and type(child) != RobertaModel:
                print(type(child))
                for p in child.parameters():
                    if p.requires_grad:
                        if len(p.shape) > 1:
                            torch.nn.init.xavier_uniform_(p)
                        else:
                            stdv = 1. / math.sqrt(p.shape[0])
                            torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def save_model(self, save_path, model, args):
        # Save a trained model, configuration and tokenizer
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_path, WEIGHTS_NAME)
        output_config_file = os.path.join(save_path, CONFIG_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)

        config = model_to_save.config
        config.__dict__["device"] = self.device
        with open(output_config_file, "w", encoding='utf-8') as writer:
            writer.write(config.to_json_string())
        output_args_file = os.path.join(save_path, 'training_args.bin')
        torch.save(args, output_args_file)

    def _train(self, criterion, optimizer, train_data_loader, val_data_loader, test_data_loader):
        max_val_corr = -1
        early_steps = 0
        global_step = 0
        path = None
        device = self.opt.device

        results = {"bert_model": self.opt.model_path, "batch_size": self.opt.batch_size,
                   "learning_rate": self.opt.learning_rate, "seed": self.opt.seed,
                   "l2reg": self.opt.l2reg, "dropout_rate": self.opt.bert_dropout, "incro": self.opt.incro}
        for epoch in range(self.opt.num_epoch):
            logger.info('>' * 100)
            logger.info('epoch: {}'.format(epoch))
            # start_time = (datetime.now())
            n_correct, n_total, loss_total = 0, 0, 0
            # switch model to training mode
            self.model.train()
            for i_batch, sample_batched in enumerate(train_data_loader):
                global_step += 1
                # clear gradient accumulators
                optimizer.zero_grad()
                if self.opt.use_gcn:
                    outputs = self.model(sample_batched[0]["input_ids"].to(device),
                                         sample_batched[0]["attention_masks"].to(device),
                                         sample_batched[0]["valid_ids"].to(device),
                                         sample_batched[0]["mem_valid_ids"].to(device),
                                         sample_batched[0]["dep_adj_matrix"].to(device),

                                         sample_batched[1]["input_ids_b"].to(device),
                                         sample_batched[1]["attention_masks_b"].to(device),
                                         sample_batched[1]["valid_ids_b"].to(device),
                                         sample_batched[1]["mem_valid_ids_b"].to(device),
                                         sample_batched[1]["dep_adj_matrix_b"].to(device))
                else:
                    outputs = self.model(sample_batched[0]["input_ids"].to(device),
                                         sample_batched[0]["attention_masks"].to(device),

                                         sample_batched[1]["input_ids_b"].to(device),
                                         sample_batched[1]["attention_masks_b"].to(device))

                targets = sample_batched[0]['sim_score'].to(self.opt.device)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                n_total += 1
                loss_total += loss.item()
                if global_step % self.opt.log_step == 0:
                    train_loss = loss_total / n_total
                    logger.info('epoch: {}, loss: {:.4f}'.format(epoch, train_loss))

            val_pearson, val_spearman = Instructor._evaluate_pearson_spearman(self.model, val_data_loader,
                                                                              self.opt.use_gcn,
                                                                              device=self.opt.device)
            logger.info('>epoch: {}, val_pearson corr: {:.4f}, val_spearman corr: {:.4f}'.format(epoch, val_pearson,
                                                                                                 val_spearman))

            if val_spearman > max_val_corr:  # 根据dev上的spearman 来进行早停
                max_val_corr = val_spearman
                # saving_path = os.path.join(self.opt.outdir, "epoch_{}".format(epoch))
                saving_path = os.path.join(self.opt.outdir, "best_model")
                if not os.path.exists(saving_path):
                    os.makedirs(saving_path)
                # self.save_model(saving_path, self.model, self.opt)

                self.model.eval()
                # saving_path = os.path.join(self.opt.outdir, "epoch_{}_eval.txt".format(epoch))
                saving_path = os.path.join(self.opt.outdir, "best_prediction.txt")
                test_pearson, test_spearman = self._evaluate_pearson_spearman(self.model, test_data_loader,
                                                                              self.opt.use_gcn,
                                                                              device=self.opt.device,
                                                                              saving_path=saving_path)
                logger.info('>> epoch: {}, test_pearson: {:.4f}, test_spearman: {:.4f}'.format(epoch, test_pearson,
                                                                                               test_spearman))

                results["max_val_spearman"] = max_val_corr
                results["test_pearson_corr"] = test_pearson
                results["test_spearman_corr"] = test_spearman

                results["{}_val_pearson_corr".format(epoch)] = val_pearson
                results["{}_val_spearman_corr".format(epoch)] = val_spearman
            else:
                early_steps += 1
                if early_steps >= 5:
                    break
            output_eval_file = os.path.join(self.opt.outdir, "eval_results.txt")
            with open(output_eval_file, "w") as writer:
                for k, v in results.items():
                    writer.write("{}={}\n".format(k, v))
        return path

    @staticmethod
    def _evaluate_pearson_spearman(model, data_loader, use_gcn, device, saving_path=None):
        t_targets_all, t_outputs_all = None, None
        r = np.arange(1, 6)
        # switch model to evaluation mode
        model.eval()

        saving_path_f = open(saving_path, 'w') if saving_path is not None else None

        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(data_loader):
                t_targets = t_sample_batched[0]['sim_score'].to(device)
                t_raw_texts_a = t_sample_batched[0]['raw_text']
                t_raw_texts_b = t_sample_batched[1]['raw_text_b']

                if use_gcn:
                    t_outputs = model(t_sample_batched[0]["input_ids"].to(device),
                                      t_sample_batched[0]["attention_masks"].to(device),
                                      t_sample_batched[0]["valid_ids"].to(device),
                                      t_sample_batched[0]["mem_valid_ids"].to(device),
                                      t_sample_batched[0]["dep_adj_matrix"].to(device),

                                      t_sample_batched[1]["input_ids_b"].to(device),
                                      t_sample_batched[1]["attention_masks_b"].to(device),
                                      t_sample_batched[1]["valid_ids_b"].to(device),
                                      t_sample_batched[1]["mem_valid_ids_b"].to(device),
                                      t_sample_batched[1]["dep_adj_matrix_b"].to(device))
                else:
                    t_outputs = model(t_sample_batched[0]["input_ids"].to(device),
                                      t_sample_batched[0]["attention_masks"].to(device),

                                      t_sample_batched[1]["input_ids_b"].to(device),
                                      t_sample_batched[1]["attention_masks_b"].to(device))

                # 将输出的概率分布变换为similarity score
                t_outputs = np.dot(t_outputs.detach().cpu().numpy(), r)
                t_targets = np.dot(t_targets.detach().cpu().numpy(), r)
                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = np.concatenate((t_targets_all, t_targets), axis=0)
                    t_outputs_all = np.concatenate((t_outputs_all, t_outputs), axis=0)

                if saving_path_f is not None:
                    # saving_path_f.write('prediction\tground-truth\tsentence A\tsentence B\n')
                    for t_target, t_output, t_raw_text_a, t_raw_text_b in zip(t_targets,
                                                                              t_outputs,
                                                                              t_raw_texts_a, t_raw_texts_b):
                        saving_path_f.write(
                            "{}\t{}\t{}\t{}\n".format(round(t_target, 5), round(t_output, 5), t_raw_text_a,
                                                      t_raw_text_b))
        p_corr, _ = pearsonr(t_outputs_all, t_targets_all)
        sp_corr, _ = spearmanr(t_outputs_all, t_targets_all)
        p_corr = round(p_corr, 4)
        sp_corr = round(sp_corr, 4)
        return p_corr, sp_corr

    def train(self):
        # Loss and Optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)
        train_data_loader = DataLoader(dataset=self.trainset, batch_size=self.opt.batch_size, shuffle=True)
        val_data_loader = DataLoader(dataset=self.valset, batch_size=self.opt.batch_size, shuffle=False)
        test_data_loader = DataLoader(dataset=self.testset, batch_size=self.opt.batch_size, shuffle=False)

        self._reset_params()
        self._print_args()
        self._train(criterion, optimizer, train_data_loader, val_data_loader, test_data_loader)


def test(opt):
    logger.info(opt)
    config = BertConfig.from_json_file(os.path.join(opt.bert_model, CONFIG_NAME)) if opt.model_type == 'bert' \
        else RobertaConfig.from_json_file(os.path.join(opt.bert_model, CONFIG_NAME))
    logger.info(config)

    tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.bert_model, opt.model_type)
    if opt.use_gcn:
        model = Bert_gcn.from_pretrained(opt.model_path) if opt.model_type == 'bert' else Roberta_gcn.from_pretrained(
            opt.model_path)
    else:
        model = Bert.from_pretrained(opt.model_path) if opt.model_type == 'bert' else Roberta.from_pretrained(
            opt.model_path)
    model.to(opt.device)

    testset = ABSADataset(opt.test_file, tokenizer, opt)
    test_data_loader = DataLoader(dataset=testset, batch_size=opt.batch_size, shuffle=False)

    test_pcorr, test_spcorr = Instructor._evaluate_pearson_spearman(model, test_data_loader, opt.use_gcn,
                                                                    device=opt.device)
    logger.info('>> test_pearson: {:.4f}, test_spearman: {:.4f}'.format(test_pcorr, test_spcorr))


def get_args():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='', type=str)
    parser.add_argument('--train_file', default='./parsing_data/SICK_ori/SICK_train.txt',
                        type=str)  # SICK_ori/SICK_train.txt
    parser.add_argument('--val_file', default='./parsing_data/SICK_ori/SICK_trial.txt',
                        type=str)  # SICK_ori/SICK_trial.txt
    parser.add_argument('--test_file', default='./parsing_data/SICK_ori/SICK_test_annotated.txt',
                        type=str)  # SICK_ori/SICK_test_annotated.txt
    parser.add_argument('--learning_rate', default='2e-5', type=float)  # 2e-5
    parser.add_argument('--bert_dropout', default=0.0, type=float)
    parser.add_argument('--l2reg', default=0.0, type=float)
    parser.add_argument('--num_epoch', default=15, type=int)  # 15
    parser.add_argument('--batch_size', default=32, type=int)  # 32
    parser.add_argument('--log_step', default=32, type=int)
    parser.add_argument('--max_seq_len', default=64, type=int)
    parser.add_argument('--label_dim', default=5, type=int)
    parser.add_argument('--device', default=None, type=str, help='e.g. cuda:0')
    parser.add_argument('--seed', default=123, type=int, help='set seed for reproducibility')  # 123
    parser.add_argument('--valset_ratio', default=0.0, type=float,
                        help='set ratio between 0 and 1 for validation support')
    parser.add_argument('--log', default='./logs/reg_logs', type=str)
    parser.add_argument('--bert_model', default='./bert-base-uncased', type=str)
    parser.add_argument('--model_path', default='./bert-base-uncased', type=str)
    parser.add_argument('--incro', default='cls', type=str)
    parser.add_argument('--weight', default=-1, type=float, help='the weight of weighted average')
    parser.add_argument('--outdir', default='outputs/GCN-SE_Regression/',
                        type=str)  # ./GCN_Regression/ 06-10,该目录没有对相似度分数进行变换,最后输出神经单元为1,即为score
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--use_gcn", action='store_true', help="Whether to use gcn.")
    parser.add_argument('--task_name', default='SICK', type=str, help='training or evaluating task')
    parser.add_argument('--model_type', default='bert', type=str, help='using bert or roberta?')
    opt = parser.parse_args()

    if opt.do_train:
        now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        opt.outdir = os.path.join(opt.outdir,
                                  "{}/{}task_{}_fusion_method_{}_bts_{}_lr_{}_epoch_{}_l2reg_{}_seed_{}_bert_dropout_{}_{}".format(
                                      opt.model_path.split('/')[1],
                                      opt.model_name,
                                      opt.task_name,
                                      opt.incro,
                                      opt.batch_size,
                                      opt.learning_rate,
                                      opt.num_epoch,
                                      opt.l2reg,
                                      opt.seed,
                                      opt.bert_dropout,
                                      now_time
                                  ))
        if not os.path.exists(opt.outdir):
            os.makedirs(opt.outdir)

    return opt


def main():
    opt = get_args()

    if opt.seed is not None:
        random.seed(opt.seed)
        np.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    log_file = '{}/{}-{}.log'.format(opt.log, opt.incro, strftime("%y%m%d-%H%M", localtime()))
    if not os.path.exists(opt.log):  # 新增,不然找不到./log目录
        os.makedirs(opt.log)
    logger.addHandler(logging.FileHandler(log_file))

    if opt.do_train:
        ins = Instructor(opt)
        ins.train()
    elif opt.do_eval:
        test(opt)


if __name__ == '__main__':
    main()

import logging
import argparse
import math
import os
import sys
from time import strftime, localtime
import random
import numpy as np
import datetime
from sklearn import metrics
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from transformers import BertModel, BertConfig
from utils import Tokenizer4Bert, ABSADataset
from models.model import Bert, Roberta
from models.model import Bert_gcn, Roberta_gcn
from transformers import RobertaModel, RobertaConfig

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
        config.task = opt.task_name
        config.alpha = opt.weight
        if opt.model_type == 'bert':
            if opt.use_gcn:
                self.model = Bert_gcn.from_pretrained(opt.model_path, config=config)
            else:
                self.model = Bert.from_pretrained(opt.model_path, config=config)
        elif opt.model_type == 'roberta':
            if opt.use_gcn:
                self.model = Roberta_gcn.from_pretrained(opt.model_path, config=config)
            else:
                self.model = Roberta.from_pretrained(opt.model_path, config=config)

        self.model.to(opt.device)
        self.tokenizer = tokenizer
        self.device = opt.device.type
        self.task = opt.task_name
        logger.info(opt)
        label2id = ABSADataset.get_label2id(opt)
        logger.info(label2id)
        self.label2id = label2id
        self.trainset = ABSADataset(opt.train_file, tokenizer, self.opt)
        self.testset = ABSADataset(opt.test_file, tokenizer, self.opt)
        if os.path.exists(opt.val_file):
            self.valset = ABSADataset(opt.val_file, tokenizer, self.opt)
        elif opt.valset_ratio > 0:
            print('using random set as valset')
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
            # print('参数类型', type(child))
            # print(child)
            if type(child) != BertModel and type(child) != RobertaModel:  # skip bert params,8-29新增albert
                # if type(child) != BertModel:
                print(type(child))
                for p in child.parameters():
                    # print(p)
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
        config.__dict__["label2id"] = self.label2id
        config.__dict__["device"] = self.device  # 新增
        with open(output_config_file, "w", encoding='utf-8') as writer:
            writer.write(config.to_json_string())
        output_args_file = os.path.join(save_path, 'training_args.bin')
        torch.save(args, output_args_file)

    def _train(self, criterion, optimizer, train_data_loader, val_data_loader, test_data_loader):
        max_val_acc = 0
        early_steps = 0
        global_step = 0
        path = None
        device = self.opt.device

        if self.opt.use_gcn:
            results = {"bert_model": self.opt.model_path,
                       "batch_size": self.opt.batch_size,
                       "learning_rate": self.opt.learning_rate,
                       "seed": self.opt.seed,
                       "l2reg": self.opt.l2reg,
                       "dropout_rate": self.opt.bert_dropout,
                       "incro": self.opt.incro,
                       "weight": self.opt.weight}
        else:
            results = {"bert_model": self.opt.model_path,
                       "batch_size": self.opt.batch_size,
                       "learning_rate": self.opt.learning_rate,
                       "seed": self.opt.seed,
                       "l2reg": self.opt.l2reg,
                       "dropout_rate": self.opt.bert_dropout,
                       "incro": self.opt.incro}
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
                # print(targets)
                loss.backward()

                optimizer.step()

                n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                n_total += len(outputs)
                loss_total += loss.item() * len(outputs)
                if global_step % self.opt.log_step == 0:
                    train_acc = n_correct / n_total
                    train_loss = loss_total / n_total
                    logger.info('epoch: {}, loss: {:.4f}, acc: {:.4f}'.format(epoch, train_loss, train_acc))

            val_acc, val_f1 = Instructor._evaluate_acc_f1(self.model, val_data_loader, self.opt.use_gcn, device=self.opt.device)
            logger.info('>epoch: {}, val_acc: {:.4f}, val_f1: {:.4f}'.format(epoch, val_acc, val_f1))

            if val_acc > max_val_acc:
                max_val_acc = val_acc
                # saving_path = os.path.join(self.opt.outdir, "epoch_{}".format(epoch))
                saving_path = os.path.join(self.opt.outdir, "best_model")
                if not os.path.exists(saving_path):
                    os.makedirs(saving_path)
                # self.save_model(saving_path, self.model, self.opt)

                self.model.eval()
                # saving_path = os.path.join(self.opt.outdir, "epoch_{}_eval.txt".format(epoch))
                saving_path = os.path.join(self.opt.outdir, "best_prediction.txt")

                test_acc, test_f1 = self._evaluate_acc_f1(self.model, test_data_loader, self.opt.use_gcn, device=self.opt.device,
                                                          saving_path=saving_path)

                logger.info('>> epoch: {}, test_acc: {:.4f}, test_f1: {:.4f}'.format(epoch, test_acc, test_f1))

                # if self.task == 'MRPC_ori':
                # results["max_val_f1"] = max_val_acc
                # else:
                # results["max_val_acc"] = max_val_acc
                results["max_val_acc"] = max_val_acc
                results["test_acc"] = test_acc
                results["test_f1"] = test_f1

                results["{}_val_acc".format(epoch)] = val_acc
                results["{}_val_f1".format(epoch)] = val_f1
            else:
                early_steps += 1
                if early_steps >= 5:
                    break
            output_eval_file = os.path.join(self.opt.outdir, "eval_results.txt")
            with open(output_eval_file, "w") as writer:
                # writer.write(json.dumps(results, ensure_ascii=False))
                for k, v in results.items():
                    writer.write("{}={}\n".format(k, v))
        return path

    @staticmethod
    def _evaluate_acc_f1(model, data_loader, use_gcn, device, saving_path=None):
        print("------------------------------------------------------------")
        print(use_gcn)
        print("------------------------------------------------------------")
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        label_count = 2
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
                label_count = t_outputs.size(-1)
                n_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                n_total += len(t_outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

                if saving_path_f is not None:
                    for t_target, t_output, t_raw_text_a, t_raw_text_b in zip(t_targets.detach().cpu().numpy(),
                                                                              torch.argmax(t_outputs,
                                                                                           -1).detach().cpu().numpy(),
                                                                              t_raw_texts_a, t_raw_texts_b):
                        saving_path_f.write("{}\t{}\t{}\t{}\n".format(t_target, t_output, t_raw_text_a, t_raw_text_b))

        acc = round(n_correct / n_total, 4)
        # print(label_count)
        if label_count == 3:
            f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2],
                                  average='macro')
        else:
            f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), average='binary')
        f1 = round(f1, 4)
        return acc, f1

    def train(self):
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
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

    test_acc, test_f1 = Instructor._evaluate_acc_f1(model, test_data_loader, opt.use_gcn, device=opt.device)
    logger.info('>> test_acc: {:.4f}, test_f1: {:.4f}'.format(test_acc, test_f1))


def get_args():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='bert', type=str)
    parser.add_argument('--train_file', default='./parsing_data/SICK_ori/SICK_train.txt',
                        type=str)
    parser.add_argument('--val_file', default='./parsing_data/SICK_ori/SICK_trial.txt',
                        type=str)
    parser.add_argument('--test_file', default='./parsing_data/SICK_ori/SICK_test_annotated.txt',
                        type=str)  # SICK_ori/SICK_test_annotated.txt
    parser.add_argument('--learning_rate', default='2e-5', type=float)
    # parser.add_argument('--dropout', default=0.0, type=float)
    parser.add_argument('--bert_dropout', default=0.0, type=float)
    parser.add_argument('--l2reg', default=0.01, type=float)  # 0.01
    parser.add_argument('--num_epoch', default=15, type=int)  # 15
    parser.add_argument('--batch_size', default=32, type=int)  # 32
    parser.add_argument('--log_step', default=32, type=int)
    parser.add_argument('--max_seq_len', default=64, type=int)
    parser.add_argument('--label_dim', default=3, type=int)
    parser.add_argument('--device', default=None, type=str, help='e.g. cuda:0')
    parser.add_argument('--seed', default=123, type=int, help='set seed for reproducibility')  # 123
    parser.add_argument('--valset_ratio', default=0.0, type=float,
                        help='set ratio between 0 and 1 for validation support, only for MRPC task')
    parser.add_argument('--log', default='', type=str)
    parser.add_argument('--bert_model', default='./bert-base-uncased', type=str)
    parser.add_argument('--model_path', default='./bert-base-uncased', type=str)
    parser.add_argument('--incro', default='cls', type=str)
    parser.add_argument('--weight', default=-1, type=float, help='the weight of weighted average')
    parser.add_argument('--outdir', default='outputs/GCN-SE_Classification/',
                        type=str)  # './GCN-SE_Classification-new2/'
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--use_gcn", action='store_true', help="Whether to use gcn or not.")
    parser.add_argument('--task_name', default='SICK-E', type=str, help='training or evaluating task')
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
    if not os.path.exists(opt.log):
        os.makedirs(opt.log)
    logger.addHandler(logging.FileHandler(log_file))

    if opt.do_train:
        ins = Instructor(opt)
        ins.train()
    elif opt.do_eval:
        test(opt)


if __name__ == '__main__':
    main()

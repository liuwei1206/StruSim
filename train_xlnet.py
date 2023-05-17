# author = liuwei
# date = 2021-11-03

import logging
import json
import math
import os
import sys
import random
import time
import datetime

import argparse
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import RandomSampler, Sampler, SequentialSampler
from tqdm.auto import tqdm, trange

from transformers.configuration_bert import BertConfig
from transformers.tokenization_bert import BertTokenizer
from transformers.configuration_xlnet import XLNetConfig
from transformers.tokenization_xlnet import XLNetTokenizer
from transformers.optimization import AdamW
from torch.optim import Adam
from task_dataset import PretrainedDataset
from model import PretrainedModelForCohModeling

import warnings
warnings.filterwarnings('ignore')

# set logger, print to console
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

dt = datetime.datetime.now()
TIME_CHECKPOINT_DIR = "checkpoint_{}-{}-{}_{}:{}".format(dt.year, dt.month, dt.day, dt.hour, dt.minute)
PREFIX_CHECKPOINT_DIR = "checkpoint"

def get_argparse():
    """parameters for task and model"""
    parser = argparse.ArgumentParser()

    # for data
    parser.add_argument("--data_dir", default="data/dataset", type=str)
    parser.add_argument("--dataset", default="gcdc_clinton", type=str, help="toefl, gcdc")
    parser.add_argument("--fold_id", default=1, type=int, help="[1-5] for toefl, [1-10] for gcdc")
    parser.add_argument("--output_dir", default="data/result/pretrainedforcoh", type=str, help="path to save checkpoint")
    parser.add_argument("--label_list", default="low, medium, high", type=str, help="all type of labels")

    # for model
    # /pytorch_model.bin
    parser.add_argument("--pretrained_type", default="xlnet", type=str, help="bert, xlnet, ...")
    parser.add_argument("--freeze", default=False, action="store_true", help="whether to freeze the params of pretrained encoder")
    parser.add_argument("--hidden_dim", default=200, type=int, help="the dimension size of fc layer")
    parser.add_argument("--model_name_or_path", default="xlnet-base-cased", type=str, help="the pretrained bert path")

    parser.add_argument("--do_train", default=False, action="store_true", help="Whether to run training.")
    parser.add_argument("--do_dev", default=False, action="store_true", help="Whether to do evaluation")
    parser.add_argument("--do_test", default=False, action="store_true")
    parser.add_argument("--max_seq_length", default=512, type=int, help="the max length of input sequence")
    parser.add_argument("--train_batch_size", default=8, type=int, help="the training batch size")
    parser.add_argument("--eval_batch_size", default=24, type=int, help="the eval batch size")
    parser.add_argument("--num_train_epochs", default=2, type=int, help="training epoch, only work when max_step==-1")
    parser.add_argument("--learning_rate", default=1e-4, type=float, help="The initial learning rate for Adam")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="the weight of L2 normalization")
    parser.add_argument("--seed", default=106524, type=int, help="the seed used to initiate parameters")

    return parser

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_dataloader(dataset, args, mode='train'):
    if mode.upper() == 'TRAIN':
        sampler = RandomSampler(dataset)
        batch_size = args.train_batch_size
    else:
        sampler = SequentialSampler(dataset)
        batch_size = args.eval_batch_size

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        sampler=sampler
    )

    return data_loader

def train(model, args, train_dataset, dev_dataset, test_dataset):
    # 1.prepare data
    train_dataloader = get_dataloader(train_dataset, args, mode='train')
    num_train_epochs = args.num_train_epochs
    # 2.optimizer and model
    # optimizer = Adam(model.parameters(), lr=args.learning_rate)
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    for name, param in model.named_parameters():
        print(name, param.requires_grad)
    # 3.begin training
    global_step = 0
    epoch = 0
    tr_loss = 0.0
    logging_loss = 0.0
    best_dev_acc = 0.0
    best_dev_epoch = 0.0
    best_test_acc = 0.0
    best_test_epoch = 0.0
    model.zero_grad()
    train_iter = trange(1, num_train_epochs+1, desc="Epoch")
    for epoch in train_iter:
        epoch_iter = tqdm(train_dataloader, desc="Iteration")
        model.train()
        for step, batch in enumerate(epoch_iter):
            optimizer.zero_grad()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": None, "labels": batch[3]}
            outputs = model(**inputs)
            loss = outputs[0]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            tr_loss += loss.item()
            global_step += 1
            optimizer.step()
            print()
            if global_step % 40 == 0 and global_step != 0:
                print("Current Loss: ", loss.item(), "Average Loss: ", tr_loss / global_step)

        # evaluate
        train_acc = evaluate(model, args, train_dataset, epoch, description="train")
        dev_acc = evaluate(model, args, dev_dataset, epoch, description="Dev")
        test_acc = evaluate(model, args, test_dataset, epoch, description="Test")
        # print("Dev Acc: %.4f, Test Acc: %.4f" % (dev_acc, test_acc))
        print("Epoch: %d, Train Acc: %.4f, Dev Acc: %.4f, Test Acc: %.4f"%(epoch, train_acc, dev_acc, test_acc))

        if dev_acc > best_dev_acc:
            output_dir = os.path.join(args.output_dir, TIME_CHECKPOINT_DIR)
            output_dir = os.path.join(output_dir, f"{PREFIX_CHECKPOINT_DIR}_{epoch}")
            # os.makedirs(output_dir, exist_ok=True)
            # torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
            best_dev_acc = dev_acc
            best_dev_epoch = epoch

        if test_acc > best_test_acc:
            output_dir = os.path.join(args.output_dir, TIME_CHECKPOINT_DIR)
            output_dir = os.path.join(output_dir, f"{PREFIX_CHECKPOINT_DIR}_{epoch}")
            # os.makedirs(output_dir, exist_ok=True)
            # torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
            best_test_acc = test_acc
            best_test_epoch = epoch

    print("Best Dev Epoch: %d, Best Dev Acc: %.4f\n" % (best_dev_epoch, best_dev_acc))
    print("Best Test Epoch: %d, Best Test Acc: %.4f\n" % (best_test_epoch, best_test_acc))

def evaluate(model, args, dataset, epoch, description="dev", write_file=False, save_hidden_states=False, record_layer=1):
    """
    Args:
        description: which dataset you want to use for evaluation, train.json, dev.json or test.json
        write_file: if True, write the ground truth label and predict label into file
        save_hidden_states: save the hidden states for visualization
        record_layer: which hidden states to record, the first or second or last?
    """
    dataloader = get_dataloader(dataset, args, mode=description)
    batch_size = dataloader.batch_size
    all_input_ids = None
    all_label_ids = None
    all_predict_ids = None
    all_hidden_states = None

    # for batch in tqdm(dataloader, desc=description):
    model.eval()
    for batch in dataloader:
        batch = tuple(t.to(args.device) for t in batch)
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": None, "labels": batch[3], "flag": "Eval"}
        with torch.no_grad():
            outputs = model(**inputs)
            preds = outputs[0]
            hidden_states = outputs[1][record_layer]

        input_ids = batch[0].detach().cpu().numpy()
        label_ids = batch[3].detach().cpu().numpy()
        pred_ids = preds.detach().cpu().numpy()
        hidden_states = hidden_states.detach().cpu().numpy()
        if all_label_ids is None:
            all_input_ids = input_ids
            all_label_ids = label_ids
            all_predict_ids = pred_ids
            all_hidden_states = hidden_states
        else:
            all_input_ids = np.append(all_input_ids, input_ids, axis=0)
            all_label_ids = np.append(all_label_ids, label_ids, axis=0)
            all_predict_ids = np.append(all_predict_ids, pred_ids, axis=0)
            all_hidden_states = np.append(all_hidden_states, hidden_states, axis=0)
    print(all_label_ids[:20])
    print(all_predict_ids[:20])
    acc = int(np.sum(all_label_ids == all_predict_ids)) * 1.0 / len(all_label_ids)

    if write_file:
        output_dir = os.path.join(args.output_dir, TIME_CHECKPOINT_DIR)
        output_dir = os.path.join(output_dir, f"{PREFIX_CHECKPOINT_DIR}_{epoch}")
        os.makedirs(output_dir, exist_ok=True)
        pred_file = os.path.join(output_dir, "{}_res.txt".format(description))
        error_num = 1
        idx = 0
        with open(pred_file, 'w', encoding='utf-8') as f:
            f.write("%s\t%s\t%s\n"%("ID", "Gold", "Pred"))
            for label, pred in zip(all_label_ids, all_predict_ids):
                if label == pred:
                    f.write("%d\t%s\t%s\n"%(idx, label, pred))
                else:
                    f.write("%d\t%s\t%s\terror_num: %d\n"%(idx, label, pred, error_num))
                    error_num += 1
                idx += 1

    if save_hidden_states:
        output_dir = os.path.join(args.output_dir, TIME_CHECKPOINT_DIR)
        output_dir = os.path.join(output_dir, f"{PREFIX_CHECKPOINT_DIR}_{epoch}")
        save_dir = os.path(output_dir, "all_hidden_states")
        os.makedirs(save_dir, exist_ok=True)
        file = "{}-{}_hidden_state_in_pretrained_for_coh.npz".format(description, record_layer)
        file = os.path.join(save_dir, file)
        np.savez(file, hidden_states=all_hidden_states, labels=all_label_ids)

    return acc

def main():
    args = get_argparse().parse_args()
    args.no_cuda = not torch.cuda.is_available()
    if torch.cuda.is_available():
        args.n_gpu = 1
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        args.n_gpu = 0
    args.device = device
    logger.info("Training/evaluation parameters %s", args)
    set_seed(args.seed)

    # prepare data
    data_dir = os.path.join(args.data_dir, args.dataset)
    args.data_dir = os.path.join(data_dir, str(args.fold_id))
    output_dir = os.path.join(args.output_dir, args.dataset)
    output_dir = os.path.join(output_dir, str(args.fold_id))
    output_dir = os.path.join(output_dir, "xlnet+dnn")
    args.output_dir = output_dir
    os.makedirs(output_dir, exist_ok=True)
    train_file = os.path.join(args.data_dir, 'train.json')
    dev_file = os.path.join(args.data_dir, 'dev.json')
    test_file = os.path.join(args.data_dir, 'test.json')
    label_list = [l.strip().lower() for l in args.label_list.split(',')]
    args.label_list = label_list
    args.num_labels = len(label_list)

    # define model
    model_name_or_path = os.path.join("data/pretrained_models", args.model_name_or_path)
    args.config_name = os.path.join(model_name_or_path, 'config.json')
    model_path = os.path.join(model_name_or_path, "pytorch_model.bin")
    args.vocab_file = os.path.join(model_name_or_path, 'spiece.model')
    tokenizer = XLNetTokenizer.from_pretrained(args.vocab_file)
    config = XLNetConfig.from_pretrained(args.config_name)
    params = {}
    params['num_labels'] = len(label_list)
    params['hidden_dim'] = args.hidden_dim
    params['freeze'] = args.freeze
    params['model_path'] = model_path
    model = PretrainedModelForCohModeling(config=config, params=params)
    if not args.no_cuda:
        model = model.cuda()

    dataset_params = {
        'tokenizer': tokenizer,
        'max_seq_length': args.max_seq_length,
        'label_list': label_list,
        'data_dir': args.data_dir
    }

    if args.do_train:
        train_dataset = PretrainedDataset(train_file, params=dataset_params)
        dev_dataset = PretrainedDataset(dev_file, params=dataset_params)
        test_dataset = PretrainedDataset(test_file, params=dataset_params)
        train(model, args, train_dataset, dev_dataset, test_dataset)

    if args.do_dev or args.do_test:
        last_checkpoint_file = "data/result"
        last_checkpoint_file = os.path.join(last_checkpoint_file, 'pytorch_model.bin')
        epoch = 0
        model.load_state_dict(torch.load(last_checkpoint_file))
        model.from_pretrained(last_checkpoint_file, config=config, num_labels=len(label_list))

        if args.do_dev:
            dev_dataset = PretrainedDataset(dev_file, params=dataset_params)
            dev_acc = evaluate(model, args, dev_dataset, epoch=epoch, description='dev', write_file=True)
            print("Dev Acc: ", dev_acc)

        if args.do_test:
            test_dataset = PretrainedDataset(test_file, params=dataset_params)
            test_acc = evaluate(model, args, test_dataset, epoch=epoch, description='test', write_file=True)
            print("Test Acc: ", test_acc)

if __name__ == '__main__':
    main()

# author = liuwei
# date = 2021-11-03

import logging
import os
import sys
import json
import pickle
import math
import random
import time
import datetime
from tqdm import tqdm, trange

import argparse
import numpy as np
import scipy.sparse as sp
import networkx as nx
import pynauty
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler


from utils import corpus_guided_word2vec
from build_doc_subg_graph import k_nodes_graphs, graphs_to_cano_maps
from model import Graphlet_DNN
from task_dataset import GraphletDataset

# set logger, print to console
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

dt = datetime.datetime.now()
TIME_CHECKPOINT_DIR = "checkpoint_{}-{}-{}_{}:{}".format(dt.year, dt.month, dt.day, dt.hour, dt.minute)
PREFIX_CHECKPOINT_DIR = "checkpoint"

def get_argparse():
    """
    parameters for the task and model
    """
    parser = argparse.ArgumentParser()

    # for construct data
    parser.add_argument("--data_dir", default="data/dataset", type=str)
    parser.add_argument("--dataset", default="toefl", type=str, help="toefl, gcdc")
    parser.add_argument("--fold_id", default=1, type=int, help="[1-5] for toefl, [1-10] for gcdc")
    parser.add_argument("--output_dir", default="data/result/graphletdnn", type=str, help="path to save checkpoint")
    parser.add_argument("--embed_file", default="data/embedding/glove.840B.300d.txt", type=str)
    parser.add_argument("--vector_threshold", default=0.65, type=float, help="threshold when use word2vec to calculate similarity")
    parser.add_argument("--bleu_threshold", default=0.75, type=float, help="threshold when use n-gram char to calculate similarity")
    parser.add_argument("--k_node", default=5, type=int, help="k nodes subgraph features")
    parser.add_argument("--is_connected", default=False, action="store_true", help="only use weakly connected subgraph or not")
    parser.add_argument("--label_list", default="low, medium, high", type=str, help="all type of labels")
    parser.add_argument("--stanza_dir", default="data/stanza_resources", type=str, help="the download dir of stanza model")

    # for model
    parser.add_argument("--fc1_hidden_dim", default=128, type=int, help="hidden dimension of the first fc layer")
    parser.add_argument("--fc2_hidden_dim", default=48, type=int, help="hidden dimension of the second fc layer")
    parser.add_argument("--dropout", default=0.5, type=float, help="dropout value")

    # for training
    parser.add_argument("--do_train", default=False, action="store_true")
    parser.add_argument("--do_dev", default=False, action="store_true")
    parser.add_argument("--do_test", default=False, action="store_true")
    parser.add_argument("--num_train_epochs", default=10, type=int, help="training epoch")
    parser.add_argument("--train_batch_size", default=8, type=int, help="the training batch size")
    parser.add_argument("--eval_batch_size", default=8, type=int, help="the eval batch size")
    parser.add_argument("--learning_rate", default=0.02, type=float, help="learning rate")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight for L2 loss on embedding matrix.")  # 5e-4
    parser.add_argument("--seed", default=106524, type=int, help="random seed")

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
    """
    Train the model
    Args:
        model: gcn model
        args:
        inputs:
        model_path:
    """
    # 1.data
    train_dataloader = get_dataloader(train_dataset, args, mode='train')
    num_train_epochs = args.num_train_epochs

    # 2.optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # 3.begin training
    global_step = 0
    epoch = 0
    tr_loss = 0.0
    logging_loss = 0.0
    model.zero_grad()
    best_test_acc = 0.0
    best_test_epoch = 0.0
    best_dev_acc = 0.0
    best_dev_epoch = 0.0
    model.zero_grad()
    train_iter = trange(0, num_train_epochs, desc="Epoch")
    for epoch in train_iter:
        model.train()
        epoch_iter = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iter):
            optimizer.zero_grad()

            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"inputs": batch[0], "labels": batch[1], "flag": "Train"}
            outputs = model(**inputs)
            loss = outputs[0]

            loss.backward()
            tr_loss += loss.item()
            global_step += 1
            optimizer.step()
            if global_step % 50 == 0:
                print("Current Loss: ", loss.item(), "Average Loss: ", tr_loss / global_step)

        ## evaluation every epoch
        train_acc = evaluate(model, args, train_dataset, epoch, description="Train")
        dev_acc = evaluate(model, args, dev_dataset, epoch, description="Dev")
        test_acc = evaluate(model, args, test_dataset, epoch, description="Test")
        print("Epoch: %d, Train Acc: %.4f, Dev Acc: %.4f, Test Acc: %.4f" % (epoch, train_acc, dev_acc, test_acc))

        if dev_acc > best_dev_acc:
            output_dir = os.path.join(args.output_dir, TIME_CHECKPOINT_DIR)
            output_dir = os.path.join(output_dir, f"{PREFIX_CHECKPOINT_DIR}_{epoch}")
            os.makedirs(output_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
            best_dev_acc = dev_acc
            best_dev_epoch = epoch

        if test_acc > best_test_acc:
            output_dir = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}_{epoch}")
            os.makedirs(output_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
            best_test_acc = test_acc
            best_test_epoch = epoch

    print("Best Dev Epoch: %d, Best Dev Acc: %.4f\n"%(best_dev_epoch, best_dev_acc))
    print("Best Test Epoch: %d, Best Test Acc: %.4f\n" % (best_test_epoch, best_test_acc))

def evaluate(model, args, dataset, epoch, description="dev", write_file=False, save_hidden_states=False, record_layer=3):
    dataloader = get_dataloader(dataset, args, mode=description)
    batch_size = dataloader.batch_size
    all_label_ids = None
    all_predict_ids = None
    all_hidden_states = None

    epoch_iter = tqdm(dataloader, desc="Iteration")
    model.eval()
    for step, batch in enumerate(epoch_iter):
        batch = tuple(t.to(args.device) for t in batch)
        inputs = {"inputs": batch[0], "labels": batch[1], "flag": "Eval"}
        with torch.no_grad():
            outputs = model(**inputs)
            preds = outputs[0]
            hidden_states = outputs[1][record_layer]

        label_ids = batch[1].detach().cpu().numpy()
        pred_ids = preds.detach().cpu().numpy()
        hidden_states = hidden_states.detach().cpu().numpy()
        if all_label_ids is None:
            all_label_ids = label_ids
            all_predict_ids = pred_ids
            all_hidden_states = hidden_states
        else:
            all_label_ids = np.append(all_label_ids, label_ids, axis=0)
            all_predict_ids = np.append(all_predict_ids, pred_ids, axis=0)
            all_hidden_states = np.append(all_hidden_states, hidden_states, axis=0)

    acc = int(np.sum(all_label_ids == all_predict_ids)) * 1.0 / len(all_label_ids)

    if write_file:
        output_dir = os.path.join(args.output_dir, TIME_CHECKPOINT_DIR)
        output_dir = os.path.join(output_dir, f"{PREFIX_CHECKPOINT_DIR}_{epoch}")
        os.makedirs(output_dir, exist_ok=True)
        pred_file = os.path.join(output_dir, "{}_res.txt".format(description))
        error_num = 1
        idx = 1
        with open(pred_file, 'w', encoding='utf-8') as f:
            f.write("%s\t%s\n" % ("Gold", "Pred"))
            for label, pred in zip(all_label_ids, all_predict_ids):
                if label == pred:
                    f.write("%d\t%s\t%s\n" % (idx, label + 1, pred + 1))
                else:
                    f.write("%d\t%s\t%s\t%d\n" % (idx, label + 1, pred + 1, error_num))
                    error_num += 1
                idx += 1

    if save_hidden_states:
        output_dir = os.path.join(args.output_dir, TIME_CHECKPOINT_DIR)
        output_dir = os.path.join(output_dir, f"{PREFIX_CHECKPOINT_DIR}_{epoch}")
        save_dir = os.path(output_dir, "all_hidden_states")
        os.makedirs(save_dir, exist_ok=True)
        file = "{}-{}_hidden_state_in_graphlet_dnn.npz".format(description, record_layer)
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

    ## 1. prepare features, adjacency matrix, label for train, dev, test
    # mask for train, dev, test
    data_dir = os.path.join(args.data_dir, args.dataset)
    data_dir = os.path.join(data_dir, str(args.fold_id))
    args.data_dir = data_dir
    output_dir = os.path.join(args.output_dir, args.dataset)
    output_dir = os.path.join(output_dir, str(args.fold_id))
    output_dir = os.path.join(output_dir, "graphlet_dnn")
    args.output_dir = output_dir
    os.makedirs(output_dir, exist_ok=True)
    train_file = os.path.join(data_dir, 'train.json')
    dev_file = os.path.join(data_dir, 'dev.json')
    test_file = os.path.join(data_dir, 'test.json')
    label_list = [l.strip().lower() for l in args.label_list.split(',')]
    args.label_list = label_list
    args.num_labels = len(label_list)
    all_k_nodes_graphs = k_nodes_graphs(args.k_node, args.is_connected, data_dir)
    k_graphlet_cano_map = graphs_to_cano_maps(all_k_nodes_graphs)
    word_list, word2vec = corpus_guided_word2vec(data_dir, args.embed_file, args.stanza_dir)

    ## 2.model
    node_num = len(k_graphlet_cano_map) + 1 # plus 1 because there is a default node
    params = {}
    params['input_dim'] = node_num
    params['fc1_hidden_dim'] = args.fc1_hidden_dim
    params['fc2_hidden_dim'] = args.fc2_hidden_dim
    params['dropout'] = args.dropout
    params['num_labels'] = args.num_labels
    model = Graphlet_DNN(params)
    if not args.no_cuda:
        model = model.cuda()

    dataset_params = {}
    dataset_params['word2vec'] = word2vec
    dataset_params['stanza_dir'] = args.stanza_dir
    dataset_params['vector_threshold'] = args.vector_threshold
    dataset_params['bleu_threshold'] = args.bleu_threshold
    dataset_params['k_graphlet_cano_map'] = k_graphlet_cano_map
    dataset_params['k_node'] = args.k_node
    dataset_params['label_list'] = label_list

    if args.do_train:
        train_dataset = GraphletDataset(train_file, dataset_params)
        dev_dataset = GraphletDataset(dev_file, dataset_params)
        test_dataset = GraphletDataset(test_file, dataset_params)
        train(model, args, train_dataset, dev_dataset, test_dataset)

    if args.do_dev or args.do_test:
        last_checkpoint_file = "data/result"
        last_checkpoint_file = os.path.join(last_checkpoint_file, 'pytorch_model.bin')
        epoch = 0
        model.load_state_dict(torch.load(last_checkpoint_file))
        if args.do_dev:
            dev_dataset = GraphletDataset(dev_file, dataset_params)
            acc = evaluate(model, args, dev_dataset, 0, description="Dev")
            print("Dev Acc: %.4f\n"%(acc))

        if args.do_test:
            test_dataset = GraphletDataset(test_file, dataset_params)
            acc = evaluate(model, args, test_dataset, 0, description="Dev")
            print("Test Acc: %.4f\n" % (acc))

if __name__ == "__main__":
    main()

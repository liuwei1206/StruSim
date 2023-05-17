# author = liuwei
# date = 2022-07-25

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
from torch.utils.data import DataLoader

from transformers.configuration_bert import BertConfig
from transformers.tokenization_bert import BertTokenizer
from transformers.modeling_bert import BertModel

from transformers.configuration_xlnet import XLNetConfig
from transformers.tokenization_xlnet import XLNetTokenizer
from transformers.modeling_xlnet import XLNetModel

from task_dataset import PretrainedDataset
from torch.utils.data.sampler import SequentialSampler
from build_doc_subg_graph import corpus_to_heterogeneous_graph_inductive
from utils import normalize_adj, get_input_from_adjacency_matrix_inductive, cal_acc_with_mask
from model import Custom_GCN

# set logger, print to console
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

# for save model
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
    parser.add_argument("--output_dir", default="data/result", type=str, help="path to save checkpoint")
    parser.add_argument("--embed_file", default="data/embedding/glove.840B.300d.txt", type=str)
    parser.add_argument("--vector_threshold", default=0.65, type=float,
                        help="threshold when use word2vec to calculate similarity")
    parser.add_argument("--bleu_threshold", default=0.75, type=float,
                        help="threshold when use n-gram char to calculate similarity")
    parser.add_argument("--k_node", default=5, type=int, help="k nodes subgraph features")
    parser.add_argument("--windom_size", default=5, type=int,
                        help="maximum distance between sentences to form a subgraph")
    parser.add_argument("--label_list", default="low, medium, high", type=str, help="all type of labels")
    parser.add_argument("--stanza_dir", default="data/stanza_resources", type=str,
                        help="the download dir of stanza model")

    # for recall bert
    parser.add_argument("--model_name_or_path", default="xlnet-base-cased", type=str,
                        help="the pretrained bert path")
    parser.add_argument("--max_seq_length", default=1024, type=int,
                        help="the max length of input sequence, 1024 toefl, 512 gcdc")
    parser.add_argument("--pretrained_batch_size", default=8, type=int, help="batch size for calling bert")

    # for model
    parser.add_argument("--hidden_dim", default=200, type=int, help="hidden dimension of gcn layer")
    parser.add_argument("--num_gc_layer", default=2, type=int, help="number of gc layer")
    parser.add_argument("--input_dropout", default=0, type=float, help="dropout value for input features")
    parser.add_argument("--dropout", default=0.5, type=float, help="dropout value")

    # for training
    parser.add_argument("--do_train", default=False, action="store_true")
    parser.add_argument("--do_dev", default=False, action="store_true")
    parser.add_argument("--do_test", default=False, action="store_true")
    parser.add_argument("--num_train_epochs", default=800, type=int, help="training epoch")
    parser.add_argument("--learning_rate", default=0.02, type=float, help="learning rate")
    parser.add_argument("--weight_decay", default=1e-8, type=float,
                        help="Weight for L2 loss on embedding matrix.")  # 5e-4
    parser.add_argument("--seed", default=106524, type=int, help="random seed")

    return parser


def get_dataloader(dataset, batch_size):
    """sequence load texts to extract pretrained vector"""
    sampler = SequentialSampler(dataset)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        sampler=sampler
    )
    return data_loader


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(model, args, inputs, all_ids):
    """
    Train the model
    Args:
        model: gcn model
        args:
        inputs:
        model_path:
    """
    # 1.data
    train_adj_matrix, dev_adj_matrix_group, test_adj_matrix_group = inputs[0], inputs[1], inputs[2]

    train_adj_matrix, dev_adj_matrix_group, test_adj_matrix_group, features, train_labels, dev_labels, test_labels, train_mask, subgraph_mask = inputs
    train_adj_matrix = train_adj_matrix.to(args.device)
    dev_adj_matrix_group = [t.to(args.device) for t in dev_adj_matrix_group]
    test_adj_matrix_group = [t.to(args.device) for t in test_adj_matrix_group]
    features = features.to(args.device)
    train_labels = train_labels.to(args.device)
    dev_labels = dev_labels.to(args.device)
    test_labels = test_labels.to(args.device)
    train_mask = train_mask.to(args.device)
    subgraph_mask = subgraph_mask.to(args.device)
    train_ids, dev_ids, test_ids = all_ids[0], all_ids[1], all_ids[2]
    train_size, dev_size, test_size = len(train_ids), len(dev_ids), len(test_ids)
    subg_num = int(train_adj_matrix.size(0)) - train_size
    # 2.optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # 3.begin training
    epoch = 0
    epoch_trained = 0
    num_train_epochs = args.num_train_epochs
    tr_loss = 0.0
    logging_loss = 0.0
    best_dev_acc = 0.0
    best_dev_epoch = 0
    best_test_acc = 0.0
    best_test_epoch = 0
    train_iterator = trange(epoch_trained, int(num_train_epochs), desc="Epoch")
    train_indexes = torch.arange(0, subg_num+train_size).long().to(features.device)
    train_features = torch.index_select(features, 0, train_indexes)
    for epoch in train_iterator:
        model.train()
        optimizer.zero_grad()

        outputs = model(
            features=train_features,
            adjacency_matrix=train_adj_matrix,
            mask=train_mask,
            labels=train_labels,
            flag='Train'
        )
        loss = outputs[0]
        # loss += args.weight_decay * model.l2_loss()
        preds = outputs[1]

        loss.backward()
        logging_loss = loss.item()
        if "toefl" in args.dataset.lower():
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        tr_loss += logging_loss
        optimizer.step()

        train_acc, train_f1 = evaluate(model, args, inputs, all_ids, epoch, description="train", evaluate_during_train=True)
        dev_acc, dev_f1 = evaluate(model, args, inputs, all_ids, epoch, description='dev', write_file=True)
        test_acc, test_f1 = evaluate(model, args, inputs, all_ids, epoch, description='test', write_file=True)

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

        output_dir = os.path.join(args.output_dir, TIME_CHECKPOINT_DIR)
        output_dir = os.path.join(output_dir, f"{PREFIX_CHECKPOINT_DIR}_{epoch}")
        # os.makedirs(output_dir, exist_ok=True)
        # torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))

        if epoch > 0 and epoch % 1 == 0:
            logger.info("Epoch: %d, Train Acc: %.4f, Dev Acc: %.4f, Test Acc: %.4f, Train F1: %.4f, Dev F1: %.4f, Test F1: %.4f, Loss: %.4f, Avg_loss: %.4f" % (
            epoch, train_acc, dev_acc, test_acc, train_f1, dev_f1, test_f1, logging_loss, tr_loss / (epoch + 1.0)))

    print("Best Dev Epoch: %d, Best Dev Acc: %.4f\n" % (best_dev_epoch, best_dev_acc))
    print("Best Test Epoch: %d, Best Test Acc: %.4f\n" % (best_test_epoch, best_test_acc))


def evaluate(model, args, inputs, all_ids, epoch, description="dev", evaluate_during_train=True, write_file=False):
    """
        Args:
            description: which dataset you want to use for evaluation, train.json, dev.json or test.json
            write_file: if True, write the ground truth label and predict label into file
            save_hidden_states: save the hidden states for visualization
            record_layer: which hidden states to record, the first or second or last?
    """
    # 1.data
    train_adj_matrix, dev_adj_matrix_group, test_adj_matrix_group, features, train_labels, dev_labels, test_labels, train_mask, subgraph_mask = inputs
    train_adj_matrix = train_adj_matrix.to(args.device)
    dev_adj_matrix_group = [t.to(args.device) for t in dev_adj_matrix_group]
    test_adj_matrix_group = [t.to(args.device) for t in test_adj_matrix_group]
    features = features.to(args.device)
    train_labels = train_labels.to(args.device)
    dev_labels = dev_labels.to(args.device)
    test_labels = test_labels.to(args.device)
    train_mask = train_mask.to(args.device)
    subgraph_mask = subgraph_mask.to(args.device)
    train_ids, dev_ids, test_ids = all_ids[0], all_ids[1], all_ids[2]
    train_size, dev_size, test_size = len(train_ids), len(dev_ids), len(test_ids)
    subg_num = int(train_adj_matrix.size(0)) - train_size

    model.eval()

    if description.lower() == "train":
        train_indexes = torch.arange(0, subg_num + train_size).long().to(args.device)
        train_features = torch.index_select(features, 0, train_indexes)
        with torch.no_grad():
            outputs = model(
                features=train_features,
                adjacency_matrix=train_adj_matrix,
                mask=train_mask,
                labels=train_labels,
                flag='Eval'
            )

        preds = outputs[0]
        acc, f1 = cal_acc_with_mask(train_labels, preds, train_mask)
    elif description.lower() == "dev":
        # we have to evaluate one by one
        dev_preds = []
        for idx in range(dev_size):
            ith_dev_adj_matrix = dev_adj_matrix_group[idx]
            dev_id = dev_ids[idx]
            ith_dev_indexes = torch.tensor(list(range(subg_num+train_size)) + [dev_id]).long().to(args.device)
            dev_features = torch.index_select(features, 0, ith_dev_indexes)
            with torch.no_grad():
                outputs = model(
                    features=dev_features,
                    adjacency_matrix=ith_dev_adj_matrix,
                    flag='Eval'
                )
            preds = outputs[0]
            dev_preds.append(preds[-1])

        dev_preds = torch.tensor(dev_preds)
        acc, f1 = cal_acc_with_mask(dev_labels, dev_preds)
        labels = dev_labels
        preds = dev_preds
    elif description.lower() == "test":
        # we have to evaluate one by one
        test_preds = []
        for idx in range(test_size):
            ith_test_adj_matrix = test_adj_matrix_group[idx]
            test_id = test_ids[idx]
            ith_test_indexes = torch.tensor(list(range(subg_num + train_size)) + [test_id]).long().to(args.device)
            test_features = torch.index_select(features, 0, ith_test_indexes)
            with torch.no_grad():
                outputs = model(
                    features=test_features,
                    adjacency_matrix=ith_test_adj_matrix,
                    flag='Eval'
                )
            preds = outputs[0]
            test_preds.append(preds[-1])

        test_preds = torch.tensor(test_preds)
        acc, f1 = cal_acc_with_mask(test_labels, test_preds)

        labels = test_labels
        preds = test_preds

    if (not evaluate_during_train) and write_file:
        output_dir = os.path.join(args.output_dir, TIME_CHECKPOINT_DIR)
        output_dir = os.path.join(output_dir, f"{PREFIX_CHECKPOINT_DIR}_{epoch}")
        os.makedirs(output_dir, exist_ok=True)
        pred_file = os.path.join(output_dir, "{}_res.txt".format(description))
        with open(pred_file, 'w', encoding='utf-8') as f:
            f.write("%s\t%s\t%s\n" % ('ID', 'Gold', 'Pred'))
            count = 1
            for id in len(range(labels)):
                if labels[id] == preds[id]:
                    f.write(
                        "%d\t%s\t%s\n" % (int(id), args.label_list[int(labels[id])], args.label_list[int(preds[id])]))
                else:
                    f.write("%d\t%s\t%s\terror_num: %d\n" % (
                    int(id), args.label_list[int(labels[id])], args.label_list[int(preds[id])], count))
                    count += 1

    return acc, f1


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

    ## 1. prepare adjacency matrix
    # mask for train, dev, test
    data_dir = os.path.join(args.data_dir, args.dataset)
    data_dir = os.path.join(data_dir, str(args.fold_id))
    args.data_dir = data_dir
    output_dir = os.path.join(args.output_dir, args.dataset)
    output_dir = os.path.join(output_dir, str(args.fold_id))
    output_dir = os.path.join(output_dir, "custom_gcn_xlnet")
    args.output_dir = output_dir
    os.makedirs(output_dir, exist_ok=True)
    label_list = [l.strip().lower() for l in args.label_list.split(',')]
    args.label_list = label_list
    args.num_labels = len(label_list)
    graph_matrix, _, all_ids, all_labels, all_texts, _ = corpus_to_heterogeneous_graph_inductive(
        data_dir=data_dir,
        embedding_file=args.embed_file,
        k=args.k_node,
        windom_size=args.windom_size,
        stanza_dir=args.stanza_dir,
        vector_threshold=args.vector_threshold,
        bleu_threshold=args.bleu_threshold
    )

    ## 2. prepare features, train_labels, dev_labels, test_labels, train_mask, dev_mask, test_mask based on adj_matrix
    data_params = {}
    data_params['data_dir'] = args.data_dir
    data_params['max_seq_length'] = args.max_seq_length
    data_params['label_list'] = label_list
    model_name_or_path = os.path.join("data/pretrained_models", args.model_name_or_path)
    config_name = os.path.join(model_name_or_path, 'config.json')
    model_path = os.path.join(model_name_or_path, "pytorch_model.bin")
    vocab_file = os.path.join(model_name_or_path, 'spiece.model')
    tokenizer = XLNetTokenizer.from_pretrained(vocab_file)
    config = XLNetConfig.from_pretrained(config_name)
    encoder = XLNetModel.from_pretrained(model_path, config=config)
    data_params['tokenizer'] = tokenizer
    corpus_dataset = PretrainedDataset(all_texts, data_params)
    dataloader = get_dataloader(corpus_dataset, args.pretrained_batch_size)
    encoder = encoder.to(args.device)

    inputs = get_input_from_adjacency_matrix_inductive(
        adjacency_matrix=graph_matrix,
        all_ids=all_ids,
        all_labels=all_labels,
        label_list=label_list,
        encoder=encoder,
        dataloader=dataloader,
        data_dir=args.data_dir
    )

    ## 3.model
    node_num, feature_dim = inputs[3].size()
    print("node num: %d, feature dim: %d" % (node_num, feature_dim))
    params = {}
    params['input_dim'] = feature_dim
    params['hidden_dim'] = args.hidden_dim
    params['dropout'] = args.dropout
    if "toefl" in args.dataset.lower():
        args.input_dropout = 0.5
    params['input_dropout'] = args.input_dropout
    params['featureless'] = args.featureless
    params['num_labels'] = args.num_labels
    params['num_gc_layer'] = args.num_gc_layer

    model = Custom_GCN(params)
    if not args.no_cuda:
        model = model.cuda()

    if args.do_train:
        train(model, args, inputs, all_ids)

    if args.do_dev or args.do_test:
        ## toefl1
        last_checkpoint_file = "data/result/toefl_p5/1/custom_gcn_xlnet/checkpoint_2022-5-16_17:24/checkpoint_474"
        last_checkpoint_file = os.path.join(last_checkpoint_file, 'pytorch_model.bin')
        epoch = 474

        model.load_state_dict(torch.load(last_checkpoint_file))
        if args.do_dev:
            acc = evaluate(model, args, inputs, epoch, description="dev", evaluate_during_train=False, write_file=True)
            print("Dev Acc: %.4f\n" % (acc))

        if args.do_test:
            acc = evaluate(model, args, inputs, epoch, description="test", evaluate_during_train=False, write_file=True)
            print("Test Acc: %.4f\n" % (acc))


if __name__ == "__main__":
    main()

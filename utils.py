# author = liuwei
# date=2021-10-15

import os
import json
import pickle
from tqdm import tqdm

import networkx as nx
import numpy as np
import torch
import stanza
import scipy.sparse as sp
from torch_sparse import SparseTensor
from itertools import combinations
import matplotlib.pyplot as plt
from sklearn import manifold
from sklearn.metrics import accuracy_score, f1_score


# from bert_serving.client import BertClient

def corpus_guided_word2vec(corpus_dir, embedding_file, stanza_dir="/hits/fast/nlp/liuwi/stanza_resources"):
    """
    Args:
        corpus_dir: train.json, dev.json, test.json
        embedding_file:
    """
    task_name = corpus_dir.split('/')[-2]
    corpus_word2vec_name = "word2vec_for_{}.pkl".format(task_name)
    save_dir = os.path.dirname(corpus_dir)  # different fold share the same vocab
    corpus_word2vec_file = os.path.join(save_dir, corpus_word2vec_name)
    if os.path.exists(corpus_word2vec_file):
        with open(corpus_word2vec_file, 'rb') as f:
            results = pickle.load(f)
            word_list = results[0]
            word2vec = results[1]
    else:
        stanza_nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,lemma', model_dir=stanza_dir)
        files = os.listdir(corpus_dir)
        files = [os.path.join(corpus_dir, file) for file in files if ".json" in file]
        words_in_corpus = set()
        for file in files:
            with open(file, 'r', encoding="utf-8") as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if line:
                        sample = json.loads(line)
                        text = sample['text']
                        doc = stanza_nlp(text)
                        for sent in doc.sentences:
                            for word in sent.words:
                                words_in_corpus.add(word.text)
                                words_in_corpus.add(word.lemma)
        words_in_corpus = list(words_in_corpus)

        word_list = []
        word2vec = {}
        with open(embedding_file, 'r', encoding='utf-8') as f:
            idx = 0
            load_word_num = 0
            for line in f:
                line = line.strip()
                if line:
                    item = line.split()
                    if len(item) != 301:
                        continue
                    word = item[0]
                    vector = item[1:]
                    if word.strip() in words_in_corpus:
                        word_list.append(word)
                        assert len(vector) == 300, (len(vector), 300)
                        word2vec[word] = np.array(vector, dtype=np.float)
                        load_word_num += 1

                        if len(words_in_corpus) == load_word_num:
                            break
                idx += 1
                if idx % 20600 == 0:
                    print("%d%%" % (idx / 20600))

        with open(corpus_word2vec_file, 'wb') as f:
            pickle.dump([word_list, word2vec], f)

    return word_list, word2vec

def normalize_adj(adj_matrix, doc_num=1):
    """
    normalize a sparse adjacency matrix

    function based on: normalize(A) = D^-0.5 A D^0.5,
        D is the degree matrix of A
    Args:
        adj_matrix: the input adjacency matrix
    Returns:

    """
    # if min_max_norm:
    #     min_val = np.amin(adj_matrix, axis=-1, keepdims=True) # [n+k]
    #     max_val = np.amax(adj_matrix, axis=-1, keepdims=True) # [n+k]
    #     adj_matrix = (adj_matrix - min_val + 1e-6) / (max_val - min_val + 1e-6)
    """
    subg_num = adj_matrix.shape[0] - doc_num
    doc_doc = np.eye(doc_num, dtype=float) * 100
    subg_subg = np.zeros((subg_num, subg_num), dtype=float)
    subg_subg = np.eye(subg_num, dtype=float) # * 1e-6

    subg_doc = np.zeros((subg_num, doc_num), dtype=float)
    doc_subg = np.zeros((doc_num, subg_num), dtype=float)
    doc_rows = np.concatenate([doc_doc, doc_subg], axis=1)
    subg_rows = np.concatenate([subg_doc, subg_subg], axis=1) 
    self_connect = np.concatenate([doc_rows, subg_rows], axis=0)
    adj_matrix = adj_matrix + self_connect
    """
    # print("in graph normalization +++++++++++ ")
    adj_matrix = adj_matrix + np.eye(adj_matrix.shape[0], dtype=np.float)  # self-connect
    adj_matrix = sp.coo_matrix(adj_matrix)
    rowsum = np.array(adj_matrix.sum(1))  # D
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()  # D^-0.5
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)  # D^-0.5

    normalized_adj_matrix = adj_matrix.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()  # D^-0.5AD^0.5

    return normalized_adj_matrix.todense()

def prepared_adj_matrix(graph_matrix, all_ids):
    # prepare normalized matrix for training and evaluation, ensure inductive learning
    total_node_num = graph_matrix.shape[0]
    train_size, dev_size, test_size = len(all_ids[0]), len(all_ids[1]), len(all_ids[2])
    subg_node_num = total_node_num - train_size - dev_size - test_size
    subg_train_nodes = list(range(subg_node_num + train_size))
    train_adj_matrix = graph_matrix[np.ix_(subg_train_nodes, subg_train_nodes)]
    train_adj_matrix = normalize_adj(train_adj_matrix)

    dev_adj_matrix_group = []
    for dev_id in all_ids[1]:
        subg_train_ith_dev_nodes = list(range(subg_node_num + train_size)) + [dev_id]
        ith_dev_adj_matrix = graph_matrix[np.ix_(subg_train_ith_dev_nodes, subg_train_ith_dev_nodes)]
        ith_dev_adj_matrix = normalize_adj(ith_dev_adj_matrix)
        dev_adj_matrix_group.append(ith_dev_adj_matrix)

    test_adj_matrix_group = []
    for test_id in all_ids[2]:
        subg_train_ith_test_nodes = list(range(subg_node_num + train_size)) + [test_id]
        ith_test_adj_matrix = graph_matrix[np.ix_(subg_train_ith_test_nodes, subg_train_ith_test_nodes)]
        ith_test_adj_matrix = normalize_adj(ith_test_adj_matrix)
        test_adj_matrix_group.append(ith_test_adj_matrix)

    return train_adj_matrix, dev_adj_matrix_group, test_adj_matrix_group


def get_input_from_adjacency_matrix_inductive(
    adjacency_matrix,
    all_ids,
    all_labels,
    label_list,
    encoder=None,
    dataloader=None,
    data_dir="data/dataset"
):
    """
    Args:
        adjacency_matrix: [n+k, n+k], n is doc number, k is subgraph feature number
        all_ids: [[], [], []], a list of train_ids, dev_ids, test_ids, total number equal to n
        all_labels: [n], label of each doc
        label_list: all label list
        pretrained_type: which type of pretrained model, xlnet, bert robert ?
        encoder: encoder corresponding to each pretrained type
        dataloader: load dateset for encoder
        data_dir: where to save datas
    Returns:
        adjacency_matrix:
        features: [n+k, n+k]
        train_labels: [n+k]
        dev_labels: [n+k]
        test_labels: [n+k]
        train_mask: [n+k]
        dev_mask: [n+k]
        test_mask: [n+k]
    """
    total_size = adjacency_matrix.shape[0]
    features = texts_to_pretrained_vectors(
        encoder=encoder,
        dataloader=dataloader,
        node_num=total_size,
        data_dir=data_dir,
        do_inductive=True
    )

    train_adj_mat, dev_adj_mat_group, test_adj_mat_group = prepared_adj_matrix(adjacency_matrix, all_ids)

    train_ids, dev_ids, test_ids = all_ids[0], all_ids[1], all_ids[2]
    train_size, dev_size, test_size = len(train_ids), len(dev_ids), len(test_ids)
    subg_num = total_size - train_size - dev_size - test_size

    train_mask = np.zeros(subg_num + train_size, dtype=np.int32)
    train_mask[train_ids] = 1

    all_labels = [label_list.index(l.lower()) for l in all_labels]
    train_labels = np.zeros(subg_num + train_size)
    train_labels[-train_size:] = all_labels[:train_size]
    dev_labels = all_labels[train_size:(train_size + dev_size)]
    test_labels = all_labels[-test_size:]
    dev_labels = np.array(dev_labels)
    test_labels = np.array(test_labels)

    train_adj_mat = torch.tensor(train_adj_mat).float()
    dev_adj_mat_group = [torch.tensor(adj_mat).float() for adj_mat in dev_adj_mat_group]
    test_adj_mat_group = [torch.tensor(adj_mat).float() for adj_mat in test_adj_mat_group]
    train_labels = torch.tensor(train_labels).long()
    dev_labels = torch.tensor(dev_labels).long()
    test_labels = torch.tensor(test_labels).long()
    train_mask = torch.tensor(train_mask)
    subgraph_mask = 1 - train_mask

    return (
        train_adj_mat, dev_adj_mat_group, test_adj_mat_group, features,
        train_labels, dev_labels, test_labels, train_mask, subgraph_mask
    )

def cal_acc_with_mask(labels, preds, mask=None):
    assert labels.size(0) == preds.size(0), (labels.size(0), preds.size(0))
    if mask is not None:
        assert labels.size(0) == mask.size(0), (labels.size(0), mask.size(0))
        mask = mask.bool()
        labels = torch.masked_select(labels, mask)
        preds = torch.masked_select(preds, mask)

    labels = labels.detach().cpu().numpy()
    preds = preds.detach().cpu().numpy()
    acc = accuracy_score(y_true=labels, y_pred=preds)
    f1 = f1_score(y_true=labels, y_pred=preds, average="macro")

    return acc, f1

def texts_to_pretrained_vectors(encoder, dataloader, node_num, data_dir, do_inductive=False):
    """
    encode texts into pretrained vectors, which are used as feature of document nodes in graph.
    Args:
        encoder: BertModel, XLNetModel, ...
        dataloader: sequence loading dataset
        node_num: total number of nodes in graph
        data_dir: save the processed vector
    Returns:

    """
    encoder.eval()
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    dataset = data_dir.split("/")[-2]
    # file_name = "{}_vectors_for_corpus_{}.vec".format(pretrained_type, dataset)
    if do_inductive:
        file_name = "xlnet_induced_vectors_for_corpus_{}_node_num_{}.vec".format(dataset, node_num)
    else:
        file_name = "xlnet_vectors_for_corpus_{}_node_num_{}.vec".format(dataset, node_num)
    data_dir = os.path.join(data_dir, "corpus_vector")
    os.makedirs(data_dir, exist_ok=True)
    file_name = os.path.join(data_dir, file_name)
    print("Feature file name: ", file_name)
    file_name1 = "good_dataset"
    if os.path.exists(file_name):
        vectors = torch.load(file_name)
    elif os.path.exists(file_name1):
        with open(file_name1, "rb") as f:
            results = pickle.load(f)
            vectors = results[0]
            vectors = torch.tensor(vectors)
    else:
        all_vectors = []
        epoch_iter = tqdm(dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iter):
            batch = tuple(t.to(device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
            with torch.no_grad():
                outputs = encoder(**inputs)
            last_hidden_states = outputs[0]
            attention_mask = batch[1]  # [batch, seq_len]
            batch_length = torch.sum(attention_mask, dim=-1)  # [batch]
            batch_sum = last_hidden_states * attention_mask.unsqueeze(2)  # [batch, seq_len, dim]
            batch_sum = torch.sum(batch_sum, dim=1)  # [batch, dim]
            pooling_outputs = batch_sum / batch_length.unsqueeze(1)  # [batch, dim]
            all_vectors.append(pooling_outputs.detach().cpu())

        all_vectors = torch.cat(all_vectors, dim=0)
        all_vectors = all_vectors.to(device)
        padding_size = node_num - len(dataloader.dataset)
        dim = all_vectors.size(1)

        padding_vectors = torch.zeros((padding_size, dim)).float()
        padding_vectors = padding_vectors.to(device)
        if do_inductive:
            vectors = torch.cat((padding_vectors, all_vectors), dim=0)
        else:
            vectors = torch.cat((all_vectors, padding_vectors), dim=0)

        assert vectors.size(0) == node_num, (vectors.size(0), node_num)
        assert vectors.size(1) == dim, (vectors.size(1), dim)

        torch.save(vectors, file_name)

    return vectors


def visualize_doc_vector(doc_vector_files, learning_rate=200, perplexity=30):
    """
    Visualize the learnt document representation
    Args:
        doc_vector_file: a npz file, the first item is hidden_states, the second item is labels
        learning_rate: param for t-SNE, default value is 200
        perplexity: param for t-SNE, default value is 30
    """
    all_hidden_states = None
    all_labels = None
    jpg_name = None
    if isinstance(doc_vector_files, str):  # a single file
        data_dir = os.path.dirname(doc_vector_files)
        prefix = doc_vector_files.split("/")[-1].split(".")[0]
        jpg_name = os.path.join(data_dir, "{}-per_{}.jpg".format(prefix, perplexity))
        with np.load(doc_vector_files) as dataset:
            all_hidden_states = dataset["hidden_states"]
            all_labels = dataset["labels"]
    elif isinstance(doc_vector_files, list):
        data_dir = os.path.dirname(doc_vector_files[0])
        prefix = doc_vector_files[0].split("/")[-1].split(".")[0].split("-")[1]
        jpg_name = os.path.join(data_dir, "{}-per_{}.jpg".format(prefix, perplexity))
        for file in doc_vector_files:
            with np.load(file) as dataset:
                if all_hidden_states is None:
                    all_hidden_states = dataset["hidden_states"]
                    all_labels = dataset["labels"]
                else:
                    all_hidden_states = np.append(all_hidden_states, dataset["hidden_states"])
                    all_labels = np.append(all_labels, dataset["labels"])

    tsne = manifold.TSNE(n_components=2, learning_rate=learning_rate, perplexity=perplexity, random_state=106524)
    low_dim_X = tsne.fit_transform(X=all_hidden_states)
    unique_labels = np.unique(all_labels)

    colors = ['r', 'k', 'b']
    markers = ['o', 'v', 's']
    X_groups = [low_dim_X[all_labels == label] for label in unique_labels]
    for idx, idx_X in enumerate(X_groups):
        plt.scatter(idx_X[:, 0], idx_X[:, 1], c=colors[idx], marker=markers[idx], label=unique_labels[idx])

    plt.tight_layout()
    plt.savefig(jpg_name)
    # plt.show()
    plt.clf()


def custom_combinations(node_ids, k=3, windom_size=3):
    total_num = len(node_ids)
    start_pos = 0
    end_pos = total_num - k + 1
    all_k_nodes_set = []
    for idx in range(start_pos, end_pos, windom_size - k + 1):
        windom_nodes = node_ids[idx:idx + windom_size]
        windom_k_nodes_set = list(combinations(windom_nodes, k))
        all_k_nodes_set.extend(windom_k_nodes_set)
    # print(all_k_nodes_set)
    return all_k_nodes_set

    # all_nodes = []
    # for idx in range(total_num-k+1):
    #     tmp_nodes = node_ids[idx:idx+k]
    #     # print(tmp_nodes)
    #     all_nodes.append(tmp_nodes)
    # return all_nodes


if __name__ == "__main__":
    doc_file = ""
    visualize_doc_vector(doc_file, perplexity=20)


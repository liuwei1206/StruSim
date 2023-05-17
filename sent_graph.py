# author = liuwei
# date = 2021-10-10

import os
import json
import itertools
import nltk
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm, trange

import stanza
import networkx as nx
import matplotlib.pyplot as plt
import pickle

import numpy as np
import torch
from tqdm import tqdm, trange
from utils import corpus_guided_word2vec

rng = np.random.RandomState(seed=1)
# stanza.download('en')

def calculate_sentence_similar_vector(sent_A, sent_B, word2vec, eps=1e-6):
    """
    calculate the similarity of sentence based on word similarity
    Args:
        sent_A: nouns in A
        sent_B: nouns in B
    Returns:
        max_score:
        pos_A: the nouns position in A
        pos_B: the nouns position in B
    """
    if len(sent_A) == 0 or len(sent_B) == 0:
        return 0.0, (0, 0)
    # 1.convert into embeddings
    sent_A_vectors = []
    sent_B_vectors = []
    for token in sent_A:
        if token in word2vec:
            sent_A_vectors.append(word2vec[token])
        else:
            vec = rng.uniform(low=-0.2, high=+0.2, size=(300,))
            sent_A_vectors.append(vec)
            word2vec[token] = vec  # add to the word2vec

    for token in sent_B:
        if token in word2vec:
            sent_B_vectors.append(word2vec[token])
        else:
            vec = rng.uniform(low=-0.2, high=+0.2, size=(300,))
            sent_B_vectors.append(vec)
            word2vec[token] = vec

    # 2.calculate cos_similarity
    sent_A_vectors = torch.tensor(sent_A_vectors)  # [N, D]
    sent_B_vectors = torch.tensor(sent_B_vectors)  # [M, D]
    if len(sent_A_vectors.size()) == 1:
        sent_A_vectors.unsqueeze(0)
    if len(sent_B_vectors.size()) == 1:
        sent_B_vectors.unsqueeze(0)

    # cos(A, B) = A * B / (|A| * |B|)
    # [N, M]
    numerator = torch.matmul(sent_A_vectors, torch.transpose(sent_B_vectors, 1, 0))

    A_norm = torch.norm(sent_A_vectors, p=2, dim=-1)  # [N]
    B_norm = torch.norm(sent_B_vectors, p=2, dim=-1)  # [M]
    A_norm = A_norm.unsqueeze(1)  # [N, 1]
    B_norm = B_norm.unsqueeze(0)  # [1, M]
    denominator = torch.mul(A_norm, B_norm)  # [N, M]

    # obtain max val and corresponding position
    cos_similarity = numerator / (denominator + eps)  # [N, M]
    row_vals, y_poss = torch.max(cos_similarity, dim=-1)
    max_val, x_pos = torch.max(row_vals, dim=0)
    position = (int(x_pos), int(y_poss[int(x_pos)]))

    return max_val, position


def calculate_word_bleu(word1, word2, weights=[0.1, 0.2, 0.3, 0.4]):
    """
    使用BLEU来计算两个词语的相似度
    Args:
        word1:
        word2:
        weights: the weight of 1-gram score, 2-gram score, 3-gram, 4-gram
                sum(weights)=1
    """
    sent1 = [ch for ch in word1]
    sent2 = [ch for ch in word2]
    # first param is reference, a list of sentences,
    # second param is candidate sentence
    score = sentence_bleu([sent1], sent2, weights=weights)
    return score


def calculate_sentence_similar_ngram(sent_A, sent_B):
    """
    calculate similarity of sent_A and sent_B based on word bleu
    Args:
        sent_A: nouns in sent_A
        sent_B: nouns in sent_B
    """
    len_A = len(sent_A)
    len_B = len(sent_B)
    if len_A == 0 or len_B == 0:
        return 0.0, (0, 0)
    scores = [[] for _ in range(len_A)]
    for idx in range(len_A):
        for idy in range(len_B):
            word1 = sent_A[idx]
            word2 = sent_B[idy]
            score = calculate_word_bleu(word1, word2)
            scores[idx].append(score)

    scores = torch.tensor(scores)
    # print(scores.size())
    row_vals, y_poss = torch.max(scores, dim=-1)
    max_val, x_pos = torch.max(row_vals, dim=0)
    position = (int(x_pos), int(y_poss[int(x_pos)]))

    return max_val, position


def draw_graph(graph, id, label=None, data_dir=None):
    """
    draw the picture of graph
    """
    if isinstance(graph, nx.DiGraph):
        nx.draw_circular(graph, with_labels=True, arrows=True)
    else:
        graph = nx.from_numpy_matrix(graph, create_using=nx.DiGraph)
        nx.draw_circular(graph, with_labels=True, arrows=True)
    if label is not None:
        file_name = os.path.join(data_dir, "id_{}_label_{}.jpg".format(id, label))
    else:
        file_name = os.path.join(data_dir, "id_{}.jpg".format(id))
    plt.savefig(file_name)
    # plt.show()
    plt.clf()


def draw_graphs(all_graphs, all_ids, all_labels, data_dir, max_id=20):
    """
    draw the graphs
    Args:
        all_graphs:
        all_ids: [train_ids, dev_ids, test_ids]
        all_labels:
        data_dir: where to save the data
    """
    os.makedirs(data_dir, exist_ok=True)

    ids = list(itertools.chain.from_iterable(all_ids))
    for graph, id, label in zip(all_graphs, ids, all_labels):
        # print(label)
        if id < max_id:
            draw_graph(graph, id, label, data_dir)
        else:
            break


def draw_specific_graph(text_id, all_graphs, all_ids, all_labels, data_dir):
    """
        draw the graphs
        Args:
            all_graphs:
            all_ids: [train_ids, dev_ids, test_ids]
            all_labels:
            data_dir: where to save the data
        """
    data_dir = os.path.join(data_dir, "text_graphs")
    os.makedirs(data_dir, exist_ok=True)
    ids = list(itertools.chain.from_iterable(all_ids))
    draw_graph(all_graphs[text_id], ids[text_id], all_labels[text_id], data_dir)


def convert_doc_to_sent_graph(doc, stanza_nlp, word2vec, vector_threshold, bleu_threshold):
    """
    convert a document into a sentence graph
    Args:
        doc: made up of many sentences
        stanza_nlp: to obtain nouns
        word2vec: to calculate entity similar
    Returns:
        graph: the sent graph of the document
    """
    # 1.obtain sentences and lemma
    nlp_doc = stanza_nlp(doc)
    sentences = []
    sentence_entities = []
    for sentence in nlp_doc.sentences:
        if len(sentence.words) >= 6:
            sentences.append(sentence.text)
            sentence_entities.append([])
            for word in sentence.words:
                word_text = word.text
                word_lemma = word.lemma
                word_pos = word.pos

                if word_pos.lower() in ["noun", "propn"]:
                    sentence_entities[-1].append(word_lemma)
        # print(sentence_entities)

    assert len(sentences) == len(sentence_entities), (len(sentences), len(sentence_entities))

    # 2. calculate cos_similar to obtain connection
    node_num = len(sentences)
    graph = [[] for _ in range(node_num)]
    graph = np.zeros((node_num, node_num))
    for idx in range(node_num):  # start sentence
        # print(sentence_entities[idx])
        for idy in range(idx + 1, node_num):  # target sentence
            sent_A = sentence_entities[idx]
            sent_B = sentence_entities[idy]
            max_val, position = calculate_sentence_similar_vector(sent_A, sent_B, word2vec)
            if max_val >= vector_threshold:
                # print("vector score")
                graph[idx][idy] = 1
                # print(max_val, sentence_entities[idx][position[0]], sentence_entities[idy][position[1]])
            else:
                max_val, position = calculate_sentence_similar_ngram(sent_A, sent_B)
                if max_val >= bleu_threshold:
                    # print("bleu score")
                    graph[idx][idy] = 1
                    # print(max_val, sentence_entities[idx][position[0]], sentence_entities[idy][position[1]])

    return graph


def file_to_sent_graphs(
    data_file, word2vec, vector_threshold=0.65, bleu_threshold=0.75,
    stanza_dir="/hits/fast/nlp/liuwi/stanza_resources", need_draw_graph=False
):
    """
    convert a data file into sent graphs
    Args:
        data_file:
        word2vec:
        vector_threshold:
        bleu_threshold:
        stanza_dir:
        need_draw_graph:
    """
    task_name = data_file.split('/')[-2]
    mode = data_file.split('/')[-1].split('.')[0]
    corpus_dir = os.path.dirname(data_file)
    file_graph_name = "{}_sent_graphs_for_{}_vec_{}_bleu_{}.pkl".format(
        mode, task_name, int(vector_threshold * 100), int(bleu_threshold * 100)
    )
    save_dir = os.path.join(corpus_dir, "corpus_graph")
    os.makedirs(save_dir, exist_ok=True)
    graph_file = os.path.join(save_dir, file_graph_name)
    if os.path.exists(graph_file):
        print("Loading sentence graphs from %s" % (graph_file))
        with open(graph_file, 'rb') as f:
            results = pickle.load(f)
            all_graphs = results[0]
            all_ids = results[1]
            all_labels = results[2]
            all_texts = results[3]
    else:
        print("Building sentence graphs from scratch......")
        stanza_nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,lemma', model_dir=stanza_dir)

        all_graphs = []
        all_labels = []
        all_ids = [[]]  # [[train_ids]] or [[dev_ids]], or [[test_ids]]
        all_texts = []
        count = 0
        # train
        with open(data_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line:
                    sample = json.loads(line)
                    text = sample['text']
                    label = sample['score']
                    graph = convert_doc_to_sent_graph(
                        doc=text,
                        stanza_nlp=stanza_nlp,
                        word2vec=word2vec,
                        vector_threshold=vector_threshold,
                        bleu_threshold=bleu_threshold
                    )
                    # print(graph.shape)
                    graph = nx.from_numpy_matrix(graph, create_using=nx.DiGraph)
                    all_graphs.append(graph)
                    all_labels.append(label)
                    all_ids[-1].append(count)
                    all_texts.append(text)
                    count += 1

        with open(graph_file, 'wb') as f:
            pickle.dump([all_graphs, all_ids, all_labels, all_texts], f)

    if need_draw_graph:
        graph_dir = "plot_graphs_with_{}_{}_vec_{}_bleu_{}".format(
            task_name, mode, int(vector_threshold * 100), int(bleu_threshold * 100)
        )
        graph_dir = os.path.join(corpus_dir, graph_dir)
        draw_graphs(all_graphs, all_ids, all_labels, data_dir=graph_dir, max_id=20)

    return all_graphs, all_ids, all_labels, all_texts


def corpus_to_sent_graphs(
    corpus_dir, word2vec, vector_threshold=0.65, bleu_threshold=0.75,
    stanza_dir="/hits/fast/nlp/liuwi/stanza_resources", need_draw_graph=False
):
    """
    Args:
        corpus_dir: train.json, dev.json, test.json
        vector_threshold:
        bleu_threshold:
    Returns:

    """
    task_name = corpus_dir.split('/')[-2]
    corpus_graph_name = "train-dev-test_sent_graphs_for_{}_vec_{}_bleu_{}.pkl".format(
        task_name, int(vector_threshold * 100), int(bleu_threshold * 100)
    )
    save_dir = os.path.join(corpus_dir, "corpus_graph")
    os.makedirs(save_dir, exist_ok=True)
    corpus_graph_file = os.path.join(save_dir, corpus_graph_name)
    if False and os.path.exists(corpus_graph_file):
        with open(corpus_graph_file, 'rb') as f:
            results = pickle.load(f)
            all_graphs = results[0]
            all_ids = results[1]
            all_labels = results[2]
            all_texts = results[3]
    else:
        stanza_nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,lemma', model_dir=stanza_dir)

        all_texts = []
        all_graphs = []
        all_labels = []
        all_ids = []  # [[train_ids], [dev_ids], [test_ids]]
        count = 0
        # train
        for idx, file_name in enumerate(['train.json', 'dev.json', 'test.json']):
            file = os.path.join(corpus_dir, file_name)
            all_ids.append([])
            with open(file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if line:
                        sample = json.loads(line)
                        text = sample['text']
                        label = sample['score']
                        # print()
                        # print("#######################")
                        graph = convert_doc_to_sent_graph(
                            doc=text,
                            stanza_nlp=stanza_nlp,
                            word2vec=word2vec,
                            vector_threshold=vector_threshold,
                            bleu_threshold=bleu_threshold
                        )
                        # print(graph.shape)
                        graph = nx.from_numpy_matrix(graph, create_using=nx.DiGraph)
                        all_texts.append(text)
                        all_graphs.append(graph)
                        all_labels.append(label)
                        all_ids[idx].append(count)
                        count += 1

        with open(corpus_graph_file, 'wb') as f:
            pickle.dump([all_graphs, all_ids, all_labels, all_texts], f)

    if need_draw_graph:
        graph_dir = "plot_graphs_with_{}_vec_{}_bleu_{}".format(
            task_name, int(vector_threshold * 100), int(bleu_threshold * 100)
        )
        graph_dir = os.path.join(save_dir, graph_dir)
        draw_graphs(all_graphs, all_ids, all_labels, data_dir=graph_dir, max_id=20)

    return all_graphs, all_ids, all_labels, all_texts


if __name__ == "__main__":
    corpus_dir = "data/dataset/test/1"
    embedding_file = "data/embedding/glove.840B.300d.txt"
    word_list, word2vec = corpus_guided_word2vec(
        corpus_dir, embedding_file, stanza_dir="/hits/basement/nlp/liuwi/stanza_resources"
    )
    all_graphs, all_ids, all_labels, all_texts = corpus_to_sent_graphs(
        corpus_dir, word2vec, bleu_threshold=0.75, vector_threshold=0.65,
        stanza_dir="/hits/basement/nlp/liuwi/stanza_resources"
    )
    ids = list(itertools.chain.from_iterable(all_ids))
    graph_dir = os.path.join(corpus_dir, "doc_graph")
    os.makedirs(graph_dir, exist_ok=True)
    for graph, id, label in zip(all_graphs, ids, all_labels):
        draw_graph(graph, id, label, graph_dir)

    # word1 = 'spouse'
    # word2 = 'married'
    # vec1 = word2vec.get()

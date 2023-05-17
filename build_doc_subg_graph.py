# author = liuwei
# date = 2021-10-13

"""
convert graph into graphlet(subgraph features), mainly refer to the paper
“Efficient graphlet kernels for large graph comparison”

and then build the doc-subgraph heterogeneous graph
"""

import json
import os
import random
import math
from itertools import chain, combinations
import copy, time, math, pickle

import numpy as np
import networkx as nx
import pynauty

from utils import corpus_guided_word2vec, custom_combinations
from sent_graph import corpus_to_sent_graphs, draw_graph

seed = 106524


def graph_to_adjacency_matrix(graph):
    """
    get the adjacency matrix of a given graph

    We do not use the below implementation:
        adjacency_matrix = nx.adjacency_matrix(graph)
        adjacency_matrix = adjacency_matrix.to_dense()
    because of below example print a wrong adjacency matrix:
        g = nx.DiGraph()
        g.add_edge(1, 4, weight=1)
        g.add_edge(3, 2, weight=1)
        g.add_edge(3, 0, weight=1)
        g.add_edge(2, 0, weight=1)

        for edge in g2.edges():
	        print(edge)
	    # (1, 4)
        # (3, 2)
        # (3, 0)
        # (2, 0)

        adj_matrix = nx.adjacency_matrix(g2)
        print(adj_matrix)
        # (0, 1)	1
        # (2, 3)	1
        # (2, 4)	1
        # (3, 4)	1

        the result is different from its definition

    Args:
        graph: a given graph, is an object of nx.DiGraph
    Return:
        adjacency_matrix:
    """
    node_num = graph.number_of_nodes()
    adjacency_matrix = np.zeros((node_num, node_num))
    for edge in graph.edges():
        from_node = edge[0]
        to_node = edge[1]
        adjacency_matrix[from_node][to_node] = 1
    return adjacency_matrix


def k_nodes_graphs(k, dir_path=None):
    """
    given a node number k, we find out all possible k nodes graphs,
    those graphs will be graphlet, i.e. k nodes subgraphs features

    Args:
        k: the number of nodes
        dir_path: directory to save the generated graphs
    Returns:
        graphs: all possible graphes, each graph is an object of nx.DiGraph
    """
    file_name = "all_{}_nodes_graphs.pkl".format(k)
    if dir_path is not None:
        file_path = os.path.join(os.path.dirname(dir_path), file_name)
    else:
        file_path = None
    print(file_path)
    if file_path is not None and os.path.exists(file_path):
        print("loading from existing file.....")
        with open(file_path, 'rb') as f:
            graphs = pickle.load(f)
    else:
        print("init subgraphs....")
        graphs = []
        # 1.prepare foward full connected graph
        adjacency_matrix = np.zeros((k, k))
        # we only consider forward directional graph without self-loop, since entity-graph can not look back
        for idx in range(k):
            for idy in range(idx + 1, k):
                adjacency_matrix[idx][idy] = 1
        forward_full_connected_graph = nx.DiGraph(adjacency_matrix)

        # 2. we iterate all possible graphs with k nodes by deleting certain edge
        # total edge number in full connected graph, upper triangle nodes of the matrix
        total_edge_num = int(k * (k - 1) / 2)
        all_edges = forward_full_connected_graph.edges()
        # iterate all possible edge number, from 0 to total_edge_num
        for edge_num in range(total_edge_num + 1):
            # subset is all possible combination of edge num edges of all_edges
            for subset in combinations(all_edges, edge_num):
                tmp_graph = forward_full_connected_graph.copy()
                for i, j in subset:
                    tmp_graph.remove_edge(i, j)
                graphs.append(tmp_graph)

        if file_path is not None:
            with open(file_path, 'wb') as f:
                pickle.dump(graphs, f)
    print("Total number of {} subgraphs is {}: ".format(k, len(graphs)))
    return graphs


def canonical_map_of_graph(graph):
    """
    get the canonical map of a given graph.

    canonical map is a kind of one-to-one map, which convert
    a graph into a unique string. If two graphs have the same canonical
    map value, then they are isomorphic

    Args:
        graph: input graph, a nx.Digraph object
    Return:
        graph_key: a unique string of the graph
    """
    node_num = graph.number_of_nodes()
    if node_num > 0:
        adj_matrix_dict = {idx: [] for idx in range(node_num)}
        for edge in graph.edges():
            from_node = edge[0]
            to_node = edge[1]
            adj_matrix_dict[from_node].append(to_node)

        # this implement doesn't take nodes' order into account, it produces
        # the identical canonoical map for following graphs
        # 0-->1 2, 0 1-->2, 0-->2 1
        # nauty_graph = pynauty.Graph(
        #     number_of_vertices=len(g.nodes()),
        #     directed=True,
        #     adjacency_dict = adj_matrix_dict
        # )

        # this implement takes nodes' order into account
        nauty_graph = pynauty.Graph(
            number_of_vertices=node_num,
            directed=True,
            adjacency_dict=adj_matrix_dict,
            vertex_coloring=[set([t]) for t in range(node_num)]
        )

        graph_key = pynauty.certificate(nauty_graph)
    else:
        graph_key = ""

    return graph_key


def graphs_to_cano_maps(graphs, data_dir=None):
    """
    convert graphs into cano_map, which provide a effecient way for compare if two
    graphs are isomorphic.

    Args:
        graphs: input graphs

    Return:
        cano_maps:
    """
    cano_maps = {}
    data_dir = os.path.join(data_dir, "plot_subgraphs")
    os.makedirs(data_dir, exist_ok=True)
    count = 0
    for graph in graphs:
        graph_key = canonical_map_of_graph(graph)

        if graph_key not in cano_maps:
            values = {"idx": count, "graph": graph}
            cano_maps[graph_key] = values
            draw_graph(graph, id=count, data_dir=data_dir)
            count += 1

    return cano_maps


def graph_to_subgraph_count_sample(graph, k, k_graphlet_cano_map, windom_size):
    """
    convert a graph into a subgraph vector by sampling, each value
    corresponding to the frequency of that subgraph.

    The sample process is inspired by the paper "Efficient graphlet
    kernels for large graph comparison"

    Args:
        graph: the graph need to be processed
        k: node number of subgraph
        k_graphlet_cano_map: a map, key is the canonical map value of
                            a subgraph
    Return:
        count_map: a dict, the key is map value of each subgraph,
                    value is the count
    """
    np.random.seed(seed)
    adjacency_matrix = graph_to_adjacency_matrix(graph)
    node_num = graph.number_of_nodes()
    # count_map = {node id: absolute count, ...}
    count_map = {}
    k_sample_sizes = {3: 4061, 4: 8497, 5: 21251}

    assert k in [3, 4, 5], ("the size is: ", k)
    sample_size = k_sample_sizes[k]

    if node_num < k:
        # we set a default value
        count_map[len(k_graphlet_cano_map)] = sample_size
    else:
        sample_set = []
        count = 0
        while count < sample_size:
            k_nodes = random.sample(range(node_num), k)  # sample k nodes
            sorted_k_nodes = np.sort(k_nodes).tolist()
            # wrong implement
            sample_set.append(sorted_k_nodes)
            count += 1

        for sample in sample_set:
            # subgraph made up of sample nodes and corresponding edges in graph
            subgraph = adjacency_matrix[np.ix_(sample, sample)]
            pattern = nx.DiGraph(subgraph)
            tmp_cano_map = canonical_map_of_graph(pattern)
            if tmp_cano_map in k_graphlet_cano_map:
                pattern_id = k_graphlet_cano_map[canonical_map_of_graph(pattern)]['idx']
                count_map[pattern_id] = count_map.get(pattern_id, 0) + 1.0

    return count_map


def graph_to_subgraph_count_combination(graph, k, k_graphlet_cano_map, windom_size):
    """
    convert a graph into a subgraph vector by combinations. we
    enumerate all possible  subgraph in the graph by combinations.

    the return value is a dict, key is the subgraph id, value is
    the frequency of that subgraph in the graph

    Args:
        graph: a input graph
        k: node number
        k_graphlet_cano_map: a map, key is the canonical map value of
                            a subgraph, value if the information of
                            the subgraph

    Return:
        count_map: a dict. key is the id of subgraph, value is the frequency
    """
    np.random.seed(seed)
    adjacency_matrix = graph_to_adjacency_matrix(graph)
    node_num = graph.number_of_nodes()
    count_map = {}

    if node_num < k:
        count_map[len(k_graphlet_cano_map)] = 1
    else:
        # all_k_nodes_set = combinations(list(graph.nodes()), k)
        all_k_nodes_set = custom_combinations(list(graph.nodes()), k, windom_size)
        for node_set in all_k_nodes_set:
            subgraph = adjacency_matrix[np.ix_(node_set, node_set)]
            pattern = nx.DiGraph(subgraph)
            tmp_cano_map = canonical_map_of_graph(pattern)
            if tmp_cano_map in k_graphlet_cano_map:
                pattern_id = k_graphlet_cano_map[canonical_map_of_graph(pattern)]['idx']
                count_map[pattern_id] = count_map.get(pattern_id, 0) + 1.0

    return count_map


def graphs_to_subgraph_count(graphs, k, k_graphlet_cano_map, windom_size, use_sample=False):
    """
    convert graphs_to_graphlet_count_vector

    Args:
        graphs: input graphs
        k:
        k_graphlet_cano_map:
        use_sample: based on sample or combination

    Return:
        graph_subgraph_matrix:

    """

    graph_subgraph_matrix = []
    for graph in graphs:
        if use_sample:
            count_map = graph_to_subgraph_count_sample(graph, k, k_graphlet_cano_map, windom_size)
        else:
            count_map = graph_to_subgraph_count_combination(graph, k, k_graphlet_cano_map, windom_size)

        count_map = sorted(count_map.items(), key=lambda item: int(item[0]))
        vector = [0 for _ in range(len(k_graphlet_cano_map) + 1)]  # add a default value
        for key, val in count_map:
            vector[int(key)] = val
        graph_subgraph_matrix.append(vector)

    return graph_subgraph_matrix


def count_matrix_to_heterogeneous_graph_inductive(graph_subgraph_matrix, all_ids):
    """
        similar to count_matrix_to_heterogeneous_graph, but we make sure that edge values between subgraphs,
        and idf values are computed on training dataset only. In this way, the method can achieve inductive
    """
    matrix = np.array(graph_subgraph_matrix, dtype=np.float32)
    n = matrix.shape[0]
    k = matrix.shape[1]  # include a default value
    train_ids, dev_ids, test_ids = all_ids[0], all_ids[1], all_ids[2]
    train_size, dev_size, test_size = len(train_ids), len(dev_ids), len(test_ids)
    print(train_size, dev_size, test_size)
    # 1. for subg-subg edge, we calculate co-occurance from train corpus, not test
    train_matrix = matrix[:train_size, :]  # [train_size, k]
    binary_matrix = (train_matrix > 0).astype(np.float32)
    print(binary_matrix.shape)
    doc_has_subg_num = np.sum(binary_matrix, axis=0)
    doc_has_subg_num = np.where(doc_has_subg_num > 0, doc_has_subg_num, 1e-6)
    all_p_i = doc_has_subg_num / float(train_size)
    all_p_ij = np.zeros((k, k), dtype=np.float32)  # [k, k]
    for idx in range(k):
        for idy in range(idx + 1, k):
            idx_col = binary_matrix[..., idx]
            idy_col = binary_matrix[..., idy]
            idx_idy = float(np.sum(idx_col * idy_col))
            all_p_ij[idx][idy] = idx_idy / float(train_size)
            all_p_ij[idy][idx] = all_p_ij[idx][idy]
    # all_p_ij = np.where(all_p_ij > 0, all_p_ij, 1e-6)
    subg_subg_graph = np.zeros((k, k), dtype=np.float32)
    """
    for idx in range(k):
        for idy in range(k):
            pmi = math.log(all_p_ij[idx][idy] / (all_p_i[idx] * all_p_i[idy] + 1e-6) + 1e-6)
            if pmi > 0:
                subg_subg_graph[idx][idy] = pmi
    """

    # 2. for doc-doc graph, no direct connection
    doc_doc_graph = np.zeros((n, n), dtype=np.float32)  # [n, n]

    # 3. for doc-subg graph, idf from the training corpus
    idf_i = np.log(float(train_size) / doc_has_subg_num)  # [k]
    tf_ij = matrix / np.sum(matrix, axis=1, keepdims=True)  # [n, k],
    tf_idf_ij = tf_ij * idf_i  # [n, k]
    transpose_tf_idf_ij = tf_idf_ij.transpose((1, 0))  # [k, n]

    # 4.combine all small grpah to the heterogeneous graph
    # first subgraph nodes then doc nodes, total nodes number is n + k
    subg_subgdoc = np.concatenate([subg_subg_graph, transpose_tf_idf_ij], axis=1)  # [k, k+n]
    doc_subgdoc = np.concatenate([tf_idf_ij, doc_doc_graph], axis=1)  # [n, k+n]
    doc_subg_heter_graph = np.concatenate([subg_subgdoc, doc_subgdoc], axis=0)  # [k+n, k+n]

    return doc_subg_heter_graph


def corpus_to_heterogeneous_graph_inductive(
    data_dir, embedding_file, k=3, windom_size=3, stanza_dir=None,
    vector_threshold=0.65, bleu_threshold=0.75
):
    """
    build a doc-subgraph heterogeneous graph based on the input data dir
    Args:
        data_dir:
        embedding_file: embedding file
        k: subgraph nodes
    """
    task_name = data_dir.split('/')[-2]
    file_name = "{}-doc_subg_graph-tf_idf-k_{}-w_{}-vec_{}-bleu_{}.pkl".format(
        task_name, k, windom_size,
        int(vector_threshold * 100),
        int(bleu_threshold * 100)
    )
    model_data_dir = os.path.join(data_dir, "corpus_graph")
    os.makedirs(model_data_dir, exist_ok=True)
    graph_file = os.path.join(model_data_dir, file_name)
    print(graph_file)
    if os.path.exists(graph_file):
        with open(graph_file, 'rb') as f:
            doc_subg_heter_graph, all_graphs, all_ids, all_labels, all_texts, all_k_nodes_graphs = pickle.load(f)
    else:
        # 1. obtain sent graphs
        word_list, word2vec = corpus_guided_word2vec(data_dir, embedding_file, stanza_dir=stanza_dir)
        all_graphs, all_ids, all_labels, all_texts = corpus_to_sent_graphs(
            data_dir, word2vec,
            vector_threshold=vector_threshold,
            bleu_threshold=bleu_threshold,
            stanza_dir=stanza_dir
        )

        # 2. obtain all k-nodes graphs
        all_k_nodes_graphs = k_nodes_graphs(k, model_data_dir)
        k_graphlet_cano_map = graphs_to_cano_maps(all_k_nodes_graphs, model_data_dir)

        # 3. doc-subgraph
        doc_subgraph_matrix = graphs_to_subgraph_count(
            graphs=all_graphs,
            k=k,
            k_graphlet_cano_map=k_graphlet_cano_map,
            windom_size=windom_size
        )

        # 4. doc_subg_heter_graph
        doc_subg_heter_graph = count_matrix_to_heterogeneous_graph_inductive(
            graph_subgraph_matrix=doc_subgraph_matrix,
            all_ids=all_ids,
        )

        subg_num = len(doc_subgraph_matrix[0])  # offset ids
        for idx in range(len(all_ids)):
            for idy in range(len(all_ids[idx])):
                all_ids[idx][idy] += subg_num

        # print(all_ids)
        with open(graph_file, 'wb') as f:
            pickle.dump([doc_subg_heter_graph, all_graphs, all_ids, all_labels, all_texts, all_k_nodes_graphs], f)

    # for idx in range(100):
    #     draw_graph(all_graphs[idx], idx+1, all_labels[idx], data_dir)
    return doc_subg_heter_graph, all_graphs, all_ids, all_labels, all_texts, all_k_nodes_graphs


if __name__ == "__main__":
    corpus_dir = "data/dataset/toefl"
    embedding_file = "data/embedding/glove.840B.300d.txt"

    graph_matrix = corpus_to_heterogeneous_graph(corpus_dir, embedding_file)
    print(graph_matrix)

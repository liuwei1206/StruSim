# author = liuwei
# date = 2021-11-02

import os
import json
import random
import pickle

import numpy as np
import torch
from torch_sparse import SparseTensor
from torch.utils.data import Dataset

from sent_graph import file_to_sent_graphs, nx
from build_doc_subg_graph import graphs_to_subgraph_count

# from torch_geometric.nn.conv.gcn_conv import gcn_norm

random.seed(106524)


class GraphletDataset(Dataset):
    """"Dataset for graphlet features + DNN model"""

    def __init__(self, file_name, params):
        """
        Args:
             file_name: the input file dataset
             params: the params
        """
        self.word2vec = params['word2vec']
        self.stanza_dir = params['stanza_dir']
        self.vector_threshold = params['vector_threshold']
        self.bleu_threshold = params['bleu_threshold']
        self.k_graphlet_cano_map = params['k_graphlet_cano_map']
        self.k_node = params['k_node']
        self.label_list = params['label_list']
        mode = file_name.split("/")[-1].split(".")[0]
        data_dir = os.path.dirname(file_name)
        dataset = data_dir.split("/")[-2]
        self.data_dir = os.path.join(data_dir, "graphlet_dnn")
        os.makedirs(self.data_dir, exist_ok=True)
        self.graphlet_matrix_file = os.path.join(
            self.data_dir,
            "{}_graphlet_matrix_for_{}-k_{}-vec_{}-bleu_{}.pkl".format(
                mode, dataset, self.k_node,
                self.vector_threshold * 100,
                self.bleu_threshold * 100
            )
        )
        self.file_name = file_name

        self.init_dataset()

    def init_dataset(self):
        if os.path.exists(self.graphlet_matrix_file):
            print("Loading graphlet matrix from %s" % (self.graphlet_matrix_file))
            with open(self.graphlet_matrix_file, 'rb') as f:
                result = pickle.load(f)
                features = result[0]
                label_ids = result[1]
        else:
            print("Building graphlet matrix from scratch......")
            all_graphs, all_ids, all_labels, _ = file_to_sent_graphs(
                data_file=self.file_name,
                word2vec=self.word2vec,
                vector_threshold=self.vector_threshold,
                bleu_threshold=self.bleu_threshold,
                stanza_dir=self.stanza_dir
            )

            doc_subgraph_matrix = graphs_to_subgraph_count(
                graphs=all_graphs,
                k=self.k_node,
                k_graphlet_cano_map=self.k_graphlet_cano_map
            )

            # norminazal
            doc_subgraph_matrix = np.array(doc_subgraph_matrix)
            sum_val = np.sum(doc_subgraph_matrix, axis=-1, keepdims=True)
            features = doc_subgraph_matrix / (sum_val + 1e-8)
            label_ids = [self.label_list.index(label) for label in all_labels]

            with open(self.graphlet_matrix_file, "wb") as f:
                pickle.dump([features, label_ids], f)

        self.features = features
        self.label_ids = label_ids
        self.total_size = len(features)

    def __len__(self):
        return self.total_size

    def __getitem__(self, index):
        return (
            torch.tensor(self.features[index]).float(),
            torch.tensor(self.label_ids[index])
        )


class PretrainedDataset(Dataset):
    """Dataset for Pretrained models"""

    def __init__(self, file_name_or_texts, params):
        """
        Args:
            file_name_or_texts: input a file name or text list
            params: tokenizer, max_seq_length, label_list
        """
        # params
        self.tokenizer = params['tokenizer']
        self.max_seq_length = params['max_seq_length']
        self.label_list = params['label_list']
        self.data_dir = params["data_dir"]
        self.dataset = self.data_dir.split("/")[-2]

        # text and label
        if isinstance(file_name_or_texts, str):
            mode = file_name_or_texts.split("/")[-1].split(".")[0]
            all_raw_texts = []
            all_raw_labels = []
            with open(file_name_or_texts, 'r', encoding="utf-8") as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if line:
                        sample = json.loads(line)
                        text = sample['text']
                        label = sample['score']
                        # print(label, text)
                        all_raw_texts.append(text)
                        all_raw_labels.append(label)
        elif isinstance(file_name_or_texts, list):
            mode = "train-dev-test"
            all_raw_texts = file_name_or_texts
            all_raw_labels = [self.label_list[0]] * len(all_raw_texts)

        # path to save the precessed data
        file_name = "{}_{}_for_xlnet_inputs.pkl".format(mode, self.dataset)
        save_dir = os.path.join(self.data_dir, "pretrained_inputs")
        os.makedirs(save_dir, exist_ok=True)
        self.np_file = os.path.join(save_dir, file_name)

        self._init_dataset(all_raw_texts, all_raw_labels)

    def _init_dataset(self, all_raw_texts, all_raw_labels):
        """This is a private function to init the dataset"""
        if os.path.exists(self.np_file):
            with np.load(self.np_file) as dataset:
                self.input_ids = dataset["input_ids"]
                self.attention_mask = dataset["attention_mask"]
                self.segment_ids = dataset["segment_ids"]
                self.labels = dataset["labels"]
        else:
            all_input_ids = []
            all_segment_ids = []
            all_attention_mask = []
            all_labels = []

            for text, label in zip(all_raw_texts, all_raw_labels):
                text_tokens = self.tokenizer.tokenize(text)
                if len(text_tokens) > self.max_seq_length - 2:
                    text_tokens = text_tokens[:self.max_seq_length - 2]

                # for xlnet, each text represented as tokens <sep> <cls>
                text_tokens.append(self.tokenizer.sep_token)
                text_tokens.append(self.tokenizer.cls_token)
                token_ids = self.tokenizer.convert_tokens_to_ids(text_tokens)
                label_id = self.label_list.index(label)

                # note, different tokenizer posses different pad_token_id
                input_ids = np.ones(self.max_seq_length, dtype=np.int) * int(self.tokenizer.pad_token_id)
                segment_ids = np.ones(self.max_seq_length, dtype=np.int)
                attention_mask = np.zeros(self.max_seq_length, dtype=np.int)

                # for xlnet, padding as added at the begining of text
                input_ids[-len(token_ids):] = token_ids
                segment_ids[-len(token_ids):] = 0
                attention_mask[-len(token_ids):] = 1

                all_input_ids.append(input_ids)
                all_segment_ids.append(segment_ids)
                all_attention_mask.append(attention_mask)
                all_labels.append(label_id)

            assert len(all_input_ids) == len(all_segment_ids), (len(all_input_ids), len(all_segment_ids))
            assert len(all_input_ids) == len(all_attention_mask), (len(all_input_ids), len(all_attention_mask))
            assert len(all_input_ids) == len(all_labels), (len(all_input_ids), len(all_labels))

            all_input_ids = np.array(all_input_ids)
            all_segment_ids = np.array(all_segment_ids)
            all_attention_mask = np.array(all_attention_mask)
            all_labels = np.array(all_labels)
            """
            np.savez(
                self.np_file,
                input_ids=all_input_ids,
                attention_mask=all_attention_mask,
                segment_ids=all_segment_ids,
                labels=all_labels
            )
            """
            self.input_ids = all_input_ids
            self.segment_ids = all_segment_ids
            self.attention_mask = all_attention_mask
            self.labels = all_labels

        self.total_size = self.input_ids.shape[0]

    def __len__(self):
        return self.total_size

    def __getitem__(self, index):
        return (
            torch.tensor(self.input_ids[index]),
            torch.tensor(self.attention_mask[index]),
            torch.tensor(self.segment_ids[index]),
            torch.tensor(self.labels[index])
        )

# author = liuwei
# date = 2021-10-17

import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907

    one layer is equal to AXW,
        A: adjacency matrix
        X: features of nodes, if featureless, X = I
        W: the weight of convolution
    """
    def __init__(self, input_dim, output_dim, featureless=False):
        """
        Args:
            input_dim: dimension of input
            output_dim: dimension of output
            num_features_nonzero: number of nonzero features
            dropout:
            is_sparse_inputs:
            bias:
            activation:
            featureless:
        """
        super(GraphConvolution, self).__init__()

        self.featureless = featureless
        self.weight = nn.Parameter(torch.randn(input_dim, output_dim))

    def forward(self, inputs):
        """
        Args:
            features: X
            adjacency_matrix: normalized
        """
        features, adjacency_matrix = inputs
        # convolve, AXW
        # featureless means the feature is empty, just one-hot matrix,
        if not self.featureless:  # if it has features x
            xw = torch.mm(features, self.weight)
        else:
            xw = self.weight

        output = torch.mm(adjacency_matrix, xw) # AXW

        return output, adjacency_matrix

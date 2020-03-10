import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.nn import functional as F

class GraphConvolution(Module):
    '''
    '''

    def __init__(self, nnodes, in_features, out_features):
        '''
        nnodes <int>            number of nodes in the graph
        in_features <int>       number of features in each node from previous layer
        out_features <int>      number of features in each node at current layer
        k <int>                 number of edge types
        gamma <list> [0,1]      weighted importance of information passing forward [1-gamma going backward] - index matches edge type
        '''
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.nnodes = nnodes

        self.W = Parameter(torch.FloatTensor(in_features, out_features))
        self.B = Parameter(torch.FloatTensor(out_features, 1))

        self.reset_parameters()

    def reset_parameters(self):
        '''
        initialize parameters
        '''
        stdv = 1. / math.sqrt(self.out_features)

        self.W.data.uniform_(-stdv, stdv)
        self.B.data.uniform_(-stdv,stdv)

    def forward(self, X, A, C=None):
        '''
        X   <torch array>   (#nodes, #features)              rows (i) - node ; cols {j} - features
        A   <torch array>   (#nodes, #nodes, #edge_types)    non-symetric adjacency matrix ---- should not have self edges
        C   <torch array>   (#nodes, #nodes)                 coupling; learned edge weights
        '''
        Z = torch.matmul(X, self.W)
        Z = Z[...,] + self.B.T
        A = A*C
        Z = torch.bmm(A, Z)
        return Z


    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

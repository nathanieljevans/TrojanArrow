import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
import numpy as np

class TORTOISE_GCN(nn.Module):
    def __init__(self, nnodes, nfeat, GO_mat):
        super(TORTOISE_GCN, self).__init__()

        self.nnodes = nnodes
        self.nfeat = nfeat
                                  # nnodes,     in_features,    out_features,
        self.gc1 = GraphConvolution(nnodes,     nfeat,          25)
        self.gc2 = GraphConvolution(nnodes,     25,             10)
        self.gc3 = GraphConvolution(nnodes,     10,              1)

        # lets learn edge weights, call it coupling=C
        self.C = nn.parameter.Parameter(torch.ones(nnodes, nnodes, dtype=torch.float32)).requires_grad_(True)

        # Don't train this matrix 
        self.GO = GO_mat.detach().clone().type(torch.float).requires_grad_(False)

        # output FFNN
        self.out1 = nn.parameter.Parameter(torch.FloatTensor(self.GO.size()[1], 5))
        self.out1.data.uniform_(-0.1, 0.1)
        self.out1_bias = nn.parameter.Parameter(torch.FloatTensor(1, 5))
        self.out1_bias.data.uniform_(-0.1, 0.1)

        self.out2 = nn.parameter.Parameter(torch.FloatTensor(5, 1))
        self.out2.data.uniform_(-0.1, 0.1)

    def get_node_activations(self, x, adj):
        '''
        same as forward, without the final summing step.
        '''
        x = F.leaky_relu( self.gc1(x, adj, self.C) )
        x = F.leaky_relu( self.gc2(x, adj, self.C) )
        x = F.leaky_relu( self.gc3(x, adj, self.C) )
        x =               self.gc4(x, adj, self.C)

        x = x.squeeze(2)
        return x

    def forward(self, x, adj):
        x = F.leaky_relu( self.gc1(x, adj, self.C) )
        x = F.leaky_relu( self.gc2(x, adj, self.C) )
        x =               self.gc3(x, adj, self.C)

        x = x.squeeze(2)

        #! GO pathway mapping 
        x = x @ self.GO  
        x = F.leaky_relu ( x )        

        x = torch.matmul(x, self.out1)
        x = x[...,] + self.out1_bias
        x = F.leaky_relu( x )

        x = torch.matmul(x, self.out2)
        o = x.squeeze(1)
        
        #! sum method 
        #o = torch.sum(x, dim=1)

        #! concatenation method
        #x = x.squeeze(2)   # must be only one output feature per node
        #o = torch.mm(x, self.outW)
        #o = o + self.outB.expand_as(o)
        #o = o.squeeze(1)

        #! classify nodes as -1, 0, +1 method
        #x = F.softmax( x, dim=1 )
        #o = torch.argmax(x, 2) - 1                              # shift to -1,0,+1
        #o = o * torch.diag(self.coupling).expand_as(o)          # scale each node by self coupling
        #o = torch.sum(o, 1).view(-1,1).type(torch.float32)
        #o = F.relu( torch.mm(o, self.W_out1) )
        #o = torch.mm(o, self.W_out2)
        #o = o.view(-1)

        return o

    def unfreeze_coupling(self, unfreeze, verbose=True):
        '''
        To aid model convergence, it may be useful to only train coupling after the aggregation function has been learned.
        Example: freeze coupling parameter for 10 epochs, then unfreeze and train coupling & aggregator.

        freeze <bool> whether to freeze the coupling matrix. If False, will unfreeze (allow training).
        '''
        if verbose: print('Coupling Matrix (C) is frozen:', not unfreeze)
        self.C = self.C.requires_grad_(unfreeze)

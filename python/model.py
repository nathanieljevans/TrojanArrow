import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
import numpy as np

class TORTOISE_GCN(nn.Module):
    def __init__(self, nnodes, nfeat):
        super(TORTOISE_GCN, self).__init__()

        self.nnodes = nnodes
        self.nfeat = nfeat
                                  # nnodes,     in_features,    out_features,
        self.gc1 = GraphConvolution(nnodes,     nfeat,          50)
        self.gc2 = GraphConvolution(nnodes,     50,             50)
        self.gc3 = GraphConvolution(nnodes,     50,              10)
        self.gc4 = GraphConvolution(nnodes,     10,               1)

        # lets learn edge weights, call it coupling=C
        self.C = nn.parameter.Parameter(torch.ones(nnodes, nnodes, dtype=torch.float32)).requires_grad_(True)

        #self.outW = nn.parameter.Parameter(torch.FloatTensor(nnodes,1))
        #self.outB = nn.parameter.Parameter(torch.FloatTensor(1))
        #self.outW.data.uniform_(- nnodes**-0.5, nnodes**-0.5)
        #self.outB.data.uniform_(-1,1)

        # scale the sum of outputs to match gene depenency range ----
        #self.W_out1 = nn.parameter.Parameter(torch.FloatTensor(1,10))
        #self.W_out2 = nn.parameter.Parameter(torch.FloatTensor(10,1))
        #self.W_out1.data.uniform_(-1, 1)
        #self.W_out2.data.uniform_(-1, 1)

    def get_node_activations(self, x, adj):
        '''
        same as forward, without the final summing step.
        '''
        x = F.leaky_relu( self.gc1(x, adj, self.C) )
        x = F.leaky_relu( self.gc2(x, adj, self.C) )
        x = F.leaky_relu( self.gc3(x, adj, self.C) )
        x =               self.gc4(x, adj, self.C)

        # sum all last layer features
        x = x.squeeze(2)
        return x

    def forward(self, x, adj):
        x = F.leaky_relu( self.gc1(x, adj, self.C) )
        x = F.leaky_relu( self.gc2(x, adj, self.C) )
        x = F.leaky_relu( self.gc3(x, adj, self.C) )
        x =               self.gc4(x, adj, self.C)

        # sum all last layer features
        x = x.squeeze(2)
        o = torch.sum(x, dim=1)

        # concatenate all features
        #x = x.squeeze(2)   # must be only one output feature per node
        #o = torch.mm(x, self.outW)
        #o = o + self.outB.expand_as(o)
        #o = o.squeeze(1)

        # classify nodes as -1, 0, +1
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

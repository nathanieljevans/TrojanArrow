import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
import numpy as np
import config
import pickle as pkl

class TORTOISE_GCN(nn.Module):
    def __init__(self, nnodes, nfeat, GO_mat):
        super(TORTOISE_GCN, self).__init__()

        self.nnodes = nnodes
        self.nfeat = nfeat
                                  # nnodes,     in_features,    out_features,
        self.gc1 = GraphConvolution(nnodes,     nfeat,          50)
        self.gc2 = GraphConvolution(nnodes,     50,             50)
        self.gc3 = GraphConvolution(nnodes,     50,             10)
        self.gc4 = GraphConvolution(nnodes,     10,             1)

        # lets learn edge weights, call it coupling=C
        self.C = nn.parameter.Parameter(torch.ones(nnodes, nnodes, dtype=torch.float32)).requires_grad_(True)

        # Don't train this matrix 
        self.GO = GO_mat.detach().clone().type(torch.float).requires_grad_(False)

        # output FFNN
        pathway_latent_layer_size = 5
        stdv = 1. / pathway_latent_layer_size**0.5
        self.out1 = nn.parameter.Parameter(torch.FloatTensor(self.GO.size()[1], pathway_latent_layer_size))
        self.out1.data.uniform_(-stdv, stdv)
        self.out1_bias = nn.parameter.Parameter(torch.FloatTensor(1, 5))
        self.out1_bias.data.uniform_(-stdv, stdv)

        self.out2 = nn.parameter.Parameter(torch.FloatTensor(5, 1))
        self.out2.data.uniform_(-1, 1)

    def forward(self, x, adj, return_activations=False):

        D = torch.sum(adj, axis=1)
        Dinv = torch.pow(D,-0.5)
        Dinv[Dinv==float("Inf")] = 0
        Dinv = torch.diag_embed(Dinv)
        Anorm = torch.bmm(Dinv, adj)
        Anorm = torch.bmm(Anorm, Dinv)

        x =                config.ACTIVATION( self.gc1(x, Anorm, self.C) )
        x =                config.ACTIVATION( self.gc2(x, Anorm, self.C) )
        x =                config.ACTIVATION( self.gc3(x, Anorm, self.C) )
        gene_activations = config.ACTIVATION( self.gc4(x, Anorm, self.C) ) #? Should we be using a activation here? Forces pathway values to be only positive 

        x = gene_activations.squeeze(2)

        #! GO pathway mapping 
        x = x @ self.GO  
        pathway_activations = config.ACTIVATION ( x )        

        x = torch.matmul(pathway_activations, self.out1)
        x = x[...,] + self.out1_bias
        x = config.ACTIVATION( x )

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

        if return_activations: 
            return o, gene_activations, pathway_activations
        else: 
            return o

    def unfreeze_coupling(self, unfreeze, verbose=True):
        '''
        To aid model convergence, it may be useful to only train coupling after the aggregation function has been learned.
        Example: freeze coupling parameter for 10 epochs, then unfreeze and train coupling & aggregator.

        freeze <bool> whether to freeze the coupling matrix. If False, will unfreeze (allow training).
        '''
        if verbose: print('Coupling Matrix (C) is frozen:', not unfreeze)
        self.C = self.C.requires_grad_(unfreeze)

    def load_my_state_dict(self, state_dict_path, params=None):
        '''
        '''
        print('using pretrained weights...')

        with open(state_dict_path, 'rb') as f: 
            state_dict = pkl.load(f)

        if params is None: 
            params = state_dict.keys()

        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in params: 
                continue
            print(f'\t{name}: ', end='')
            if name not in own_state:
                 continue
                 print('Not Found', end='')
            else: 
                param = param.data
                print('Updated', end='')
            own_state[name].copy_(param)
            print()

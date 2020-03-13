'''

'''

import torch
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from torch.utils import data
import pickle as pkl
import imageio
import config
import time 


class Dependency_Dataset(data.Dataset):
    '''
    Characterizes a Dependency dataset
    '''

    def __init__(self, list_IDs, labels):
        '''
        Initialization
        '''
        self.label_dict = labels
        self.list_IDs = list_IDs

        self.ADJ = torch.load(config.ADJ_PATH)

        with open(config.GENEORDER_PATH, 'rb') as f:
            self.gene_order = pkl.load(f)


    def __len__(self):
        '''
        Denotes the total number of samples
        '''
        return len(self.list_IDs)

    def __getitem__(self, index):
        '''
        Generates one sample of data. Note that the adjacency matrix returned is normalized by the degree matrix
        '''

        # Select sample ID
        ID = self.list_IDs[index]
        target = self.label_dict[ID]['target']

        # load expression data for given cell line
        EXPR = torch.reshape(torch.load(f'{config.EXPR_PATH}{self.label_dict[ID]["ccle_line"]}.pt'), (self.ADJ.size(0), 1)).clone().detach().type(torch.float32)
        EXPR.requires_grad = False

        # remove connections to/from target gene - operate on target
        A = self.ADJ.clone().detach().type(torch.float32)
        A.requires_grad = False
        
        # FROM any gene TO target gene
        A[:, self.gene_order[target]] = 0 #! Use 1e-16 to avoid a singular matrix - not necessary since we're not inverting - just normalize
        # TO any gene FROM target gene
        A[self.gene_order[target], :] = 0

        y = self.label_dict[ID]['response']
        
        return A, EXPR, y


# plot and show learning process
# some material taken from: https://medium.com/@benjamin.phillips22/simple-regression-with-neural-networks-in-pytorch-313f06910379
class Training_Progress_Plotter:
    def __init__(self, figsize = (12,10), ylim = (-2.5,2.5)):
        '''
        '''
        #self.fig, self.axes = plt.subplots(1,2, figsize = figsize, sharey = True)
        self.images = []
        self.ylim = ylim
        self.figsize = figsize

    def update(self,tr_ys, tr_yhats, tst_ys, tst_yhats, epoch, tr_loss, tst_loss):
        '''
        Record the training progress at each epoch.
        '''
        self.fig, self.axes = plt.subplots(1,2, figsize = self.figsize, sharey = True)

        ######### TRAIN #########
        alpha_ = 200./len(tr_ys) if len(tr_ys) > 200. else 1.

        tr_df = pd.DataFrame({'y':tr_ys, 'yhat':tr_yhats})
        tr_df.sort_values(by='y', inplace=True)

        self.axes[0].plot(tr_df.values[:,0], 'ro', label='true', alpha=alpha_)
        self.axes[0].plot(tr_df.values[:,1], 'bo', label='predicted', alpha=alpha_)

        self.axes[0].set_title('Model output [Training Set]', fontsize=15)
        self.axes[0].set_xlabel('Sorted observations', fontsize=24)
        self.axes[0].set_ylabel('Gene Dependency Response', fontsize=24)

        self.axes[0].text(100, 30, 'Epoch = %d' % epoch, fontdict={'size': 24, 'color':  'red'})
        self.axes[0].text(100, 50, 'Loss = %.4f' % tr_loss, fontdict={'size': 24, 'color':  'red'})

        self.axes[0].set_ylim(bottom=self.ylim[0], top=self.ylim[1])

        ######### TEST #########
        alpha_ = 200./len(tst_ys) if len(tst_ys) > 200. else 1.

        self.axes[1].cla()
        tst_df = pd.DataFrame({'y':tst_ys, 'yhat':tst_yhats})
        tst_df.sort_values(by='y', inplace=True)

        self.axes[1].plot(tst_df.values[:,0], 'ro', label='true', alpha=alpha_)
        self.axes[1].plot(tst_df.values[:,1], 'bo', label='predicted', alpha=alpha_)
        plt.legend(loc='upper right')

        self.axes[1].set_title('Model output [Test Set]', fontsize=15)
        self.axes[1].set_xlabel('Sorted observations', fontsize=24)

        self.axes[1].text(1, 1, 'Epoch = %d' % epoch, fontdict={'size': 24, 'color':  'red'})
        self.axes[1].text(1, 2, 'Loss = %.4f' % tst_loss, fontdict={'size': 24, 'color':  'red'})

        self.axes[1].set_ylim(bottom=self.ylim[0], top=self.ylim[1])

        # (https://ndres.me/post/matplotlib-animated-gifs-easily/)
        self.fig.canvas.draw()                                                   # draw the canvas, cache the renderer
        image = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype='uint8')     #  
        image = image.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))   #  

        self.images.append(image)                                                #
        plt.clf()                                                                #
        plt.close('all')                                                         #

    def save_gif(self, path):
        '''
        '''
        imageio.mimsave(path, self.images, fps=3)

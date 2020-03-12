
'''
This is a fantastic resource, we'll start by implenting a super simple version of this.
https://towardsdatascience.com/how-to-do-deep-learning-on-graphs-with-graph-convolutional-networks-7d2250723780

'''

#! Wow, important! 
#? questioning 
# TODO: something 
# 

import time
import pickle as pkl
import numpy as np
from matplotlib import pyplot as plt
#from pympler import asizeof

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp

import model
import utils
import config



def plot_weight_changes(new, old):
    '''
    '''
    for param in new:
        param_delta = new[param] - old[param]
        try:
            f = plt.figure()
            plt.imshow(param_delta)
            plt.savefig(f'{config.OUTPUT_PATH}param_deltas/0{epoch}_{param}.png')
            plt.close(f)
        except:
            print('plotting weight delta failed')
            print('param', param)
            print(param_delta)
            print(param_delta.shape)



def train(model, scheduler, optimizer, loss_function, plotter, training_generator, validation_generator, epoch, device):
    '''

    '''
    _loss = 0 ; i = 0 ; tic = time.time() ; tr_ys = [] ; tr_yhats = []
    model.train()
    for A, X, y in training_generator:
        A, X, y = A.to(device), X.to(device), y.to(device)
        i += X.size(0)
        optimizer.zero_grad()
        output = model(X, A)
        tr_ys += y.detach().cpu().numpy().ravel().tolist()
        tr_yhats += output.detach().cpu().numpy().ravel().tolist()
        loss = loss_function(output, y)
        _loss += loss.detach().cpu().numpy()
        loss.backward()
        optimizer.step()
        print(f'epoch \t{epoch} --- progress: {i/len(training_generator.dataset)*100:.2f}% [{i}] --- batch loss: {loss:.4f} \t\t\t', end='\r')

    model.eval()
    _vloss = 0 ; j = 0 ; tst_ys = [] ; tst_yhats = []
    for A, X, y in validation_generator:
        j += X.size(0)
        output = model(X, A)
        tst_ys += y.detach().cpu().numpy().tolist()
        tst_yhats += output.detach().cpu().numpy().tolist()
        loss = loss_function(output, y)
        _vloss += loss.detach().cpu().numpy()

    scheduler.step(_vloss)

    try:
        plotter.update(tr_ys, tr_yhats, tst_ys, tst_yhats, epoch, _loss, _vloss)
        plotter.save_gif(f'{config.OUTPUT_PATH}cancer107_training.gif')
    except:
        print('plotting failed. continuing...')

    t = time.time() - tic
    print(f'epoch {epoch} avg train log-loss: {np.log10(_loss/len(training_generator.dataset)):.3f} \t [avg val log-loss: {np.log10(_vloss/len(validation_generator.dataset)):.3f}] --- time elapsed: {t:.1f}s \t\t\t')


if __name__ == '__main__':

    print('starting...')
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)

    print('loading data generators ... ')
    with open(f'{config.DATA_PATH}partition_dict.pkl', 'rb') as f:
        partition = pkl.load(f)

    with open(f'{config.DATA_PATH}label_dict.pkl', 'rb') as f:
        labels = pkl.load(f)
  
    # Generators
    training_set = utils.Dependency_Dataset(partition['train'], labels)
    training_generator = torch.utils.data.DataLoader(training_set, **config.params)

    validation_set = utils.Dependency_Dataset(partition['val'], labels)
    validation_generator = torch.utils.data.DataLoader(validation_set, **config.params)

    nnodes = training_set.ADJ.size()[0]
    print('number of nodes in graph:', nnodes)

    #! CUDA info

    cuda_available = torch.cuda.is_available()
    print('cuda is available:', cuda_available)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device:', device)

    print('loading GO-term matrix...')
    GO_matrix = torch.load(config.GO_MATRIX_PATH)
    GO_matrix = GO_matrix.to(device)

    # Model and optimizer
    model = model.TORTOISE_GCN(nnodes=nnodes, nfeat=1, GO_mat=GO_matrix)
    model.to(device)

    model.unfreeze_coupling(True)
    optimizer = optim.Adam(model.parameters(), lr=config.LR, weight_decay=config.L2)
    loss_function = torch.nn.SmoothL1Loss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-5)

    plotter = utils.Training_Progress_Plotter()

    old_state_dict = {}
    for key in model.state_dict():
        old_state_dict[key] = model.state_dict()[key].clone()

    print('beginning training... use multiprocessing:', config.MULTIPROCESSING)
    for epoch in range(config.EPOCHS):
        
        train(model, scheduler, optimizer, loss_function, plotter, training_generator, validation_generator, epoch, device=device)

        new_state_dict = {}
        for key in model.state_dict():
            new_state_dict[key] = model.state_dict()[key].clone()

        plot_weight_changes(new_state_dict, old_state_dict)
        old_state_dict = new_state_dict

    print('training complete.')

    with open(f'{config.OUTPUT_PATH}/model127.pkl', 'wb') as f:
        pkl.dump(model, f)

    print('model saved.')

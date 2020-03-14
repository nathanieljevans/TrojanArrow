'''

'''
from mish import Mish
import torch.nn.functional as F
from radam import RAdam
from ranger import Ranger

#! ---------------------
#! PATHS 
#! ---------------------

DATA_PATH = '../data/'
ADJ_PATH = '../data/MAPK-adjacency_matrix.pt'
GENEORDER_PATH = '../data/MAPK-gene_order.pkl'
EXPR_PATH = '../data/expr/'
OUTPUT_PATH = '../output/'
GO_MATRIX_PATH = '../data/MAPK&overlap_pathway_matrix.pt'

#! ---------------------
#! MODEL ARCHITECTURE 
#! ---------------------

ACTIVATION = Mish()    # F.leaky_relu

#! ---------------------
#! PRETRAINING - WEIGHT TRANSFER 
#! ---------------------

USE_PRETRAINED_WEIGHTS = True
STATE_DICT_PATH = '../pretrained_weights/model_state_dict-EPOCH_18.pkl'
#GCN_WEIGHTS_PATH = '../pretrained_weights/cancer127_state_dict.pkl'

#! ---------------------
#! TRAINING PARAMS 
#! ---------------------




USE_SEED = False 
SEED = 0

## Learning Weight Scheduler 
PATIENCE = 10 
DECAY_FACTOR = 0.5

# Optimizer 
OPTIMIZER = Ranger    # [RAdam, Adam]
LR = 1e-3
L2 = 0

# Number Epochs 
EPOCHS = 100

# Train on Subset 
SHORT_EPOCHS = True      #! use this for hyperparameter optimization
TRAIN_EPOCH_SIZE = 100000
VAL_EPOCH_SIZE =   10000
# Data Generator 
params = {
          'batch_size'   :    100,
          'shuffle'      :    True,
          'num_workers'  :    6
          }

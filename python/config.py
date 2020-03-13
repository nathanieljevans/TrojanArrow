'''

'''

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
#! TRAINING PARAMS 
#! ---------------------

SEED = 0

## Learning Weight Scheduler 
PATIENCE = 2 
DECAY_FACTOR = 0.5

# Optimizer 
LR = 1e-2
L2 = 1e-6

# Number Epochs 
EPOCHS = 25

# Train on Subset 
SHORT_EPOCHS = True      #! use this for hyperparameter optimization
EPOCH_SIZE = 100000

# Data Generator 
params = {
          'batch_size'   :    100,
          'shuffle'      :    True,
          'num_workers'  :    12
          }

'''

'''

#! ---------------------
#! PATHS 
#! ---------------------

DATA_PATH = '../data/'
ADJ_PATH = '../data/adjacency_matrix.pt'
GENEORDER_PATH = '../data/gene_order.pkl'
EXPR_PATH = '../data/expr/'
OUTPUT_PATH = '../output/'

#! ---------------------
#! TRAINING PARAMS 
#! ---------------------

SEED = 0

LR = 5e-2
L2 = 1e-6
EPOCHS = 100

MULTIPROCESSING = False
WORKERS = 2

params = {'batch_size': 2,
          'shuffle': True,
          'num_workers': 1}

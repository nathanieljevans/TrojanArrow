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

LR = 5e-2
L2 = 1e-6
EPOCHS = 100

MULTIPROCESSING = False
WORKERS = 2

params = {'batch_size': 500,
          'shuffle': True,
          'num_workers': 8}

#caminhos
DATA_DIR = 'dados/'
TRAIN_FILE = DATA_DIR + 'mitbih_train.csv'
TEST_FILE = DATA_DIR + 'mitbih_test.csv'
OUTPUT_DIR = 'outputs/'
BEST_PARAMS_FILE = OUTPUT_DIR + 'best_hyperparameters.json' 

#processamento de dados
VALIDATION_SIZE = 0.2
RANDOM_STATE = 42

#otimização (random search)
N_ITER_SEARCH = 15
CV_FOLDS = 3

#redes neurais
NN_EPOCHS = 50
NN_BATCH_SIZE = 128
NN_INPUT_SHAPE = (187,)
NUM_CLASSES = 5
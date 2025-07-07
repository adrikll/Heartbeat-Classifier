# --- Caminhos ---
DATA_DIR = 'dados/'
TRAIN_FILE = DATA_DIR + 'mitbih_train.csv'
TEST_FILE = DATA_DIR + 'mitbih_test.csv'
OUTPUT_DIR = 'outputs/'

# --- Parâmetros de Processamento de Dados ---
VALIDATION_SIZE = 0.2  # 20% do treino para validação
RANDOM_STATE = 42      # Para reprodutibilidade

# --- Parâmetros de Otimização (Random Search) ---
# Número de combinações de hiperparâmetros a serem testadas
N_ITER_SEARCH = 15
# Número de folds para validação cruzada dentro do search
CV_FOLDS = 3

# --- Parâmetros dos Modelos de Rede Neural ---
NN_INPUT_SHAPE = (187, 1)
NUM_CLASSES = 5
NN_EPOCHS = 20
NN_BATCH_SIZE = 128
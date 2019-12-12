#general
SINGLE_SENT_TRAIN_FILE = '../glue_data/SST-2/train.tsv'
SINGLE_SENT_VAL_FILE = '../glue_data/SST-2/dev.tsv'
SINGLE_SENT_TEST_FILE = '../glue_data/SST-2/test.tsv'
BERT_MODEL = 'uncased_L-12_H-768_A-12'
BERT_PRETRAINED_PYTORCH = 'bert-base-uncased'
BERT_PRETRAINED_DIR = '/Users/nahidalam/Documents/AI/NLP/BERT/uncased_L-12_H-768_A-12'

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#BERT Globals
LAYERS = [-1,-2,-3,-4]
NUM_TPU_CORES = 8
MAX_SEQ_LENGTH = 128
BERT_CONFIG = BERT_PRETRAINED_DIR + '/bert_config.json'
CHKPT_DIR = BERT_PRETRAINED_DIR + '/bert_model.ckpt'
VOCAB_FILE = BERT_PRETRAINED_DIR + '/vocab.txt'
INIT_CHECKPOINT = BERT_PRETRAINED_DIR + '/bert_model.ckpt'
BATCH_SIZE = 128
# In the original paper, the authors used a length of 512.
MAX_LEN = 128

#hyperparameters

hyper_param = {
    'lr': [2e-5, 3e-5, 4e-5, 5e-5]
}

# Embedding
WORD2VEC_PRETRAINED = 'GoogleNews-vectors-negative300.bin'

# System Constants

SINGLE_SENTENCE_TASK = 'single sentence classification'
SENTENCE_PAIR_TASK = 'sentence pair detection'

# student model input parameters 

student_input_size = 300
student_hidden_size = 300
student_num_layers = 1
student_num_classes = 2
student_classification_hidden_size=400
student_dropout_rate=0.15

import json
import logging
import pandas as pd
from util.prepare_BERT_input import *
from util.load_model import *
from util.features import *
from configs.constant import *

logging.basicConfig()
logger = logging.getLogger(__name__)

'''
Dataset used for single sentence classification SST-2
Raw version of the data not available
'''

def run():
    # make sure you have the data in a directory
    # prepare input - this is specific to dataset SST-2
    train_df = pd.read_csv(SINGLE_SENT_TRAIN_FILE, delimiter='\t', header=None, names=['sentence', 'label'])
    train_sentences = train_df.sentence.values
    train_dataloader = prepare_BERT_input(train_sentences[1:])

    val_df = pd.read_csv(SINGLE_SENT_VAL_FILE, delimiter='\t', header=None, names=['sentence', 'label'])
    val_sentences = val_df.sentence.values
    validation_dataloader = prepare_BERT_input(val_sentences[1:])

    test_df = pd.read_csv(SINGLE_SENT_TEST_FILE, delimiter='\t', header=None, names=['sentence', 'label'])
    test_sentences = test_df.sentence.values
    test_dataloader = data.DataLoader(test_sentences[1:])

    # initialize teacher object
    teacher_single_sentence = teacher(BERT_PRETRAINED_PYTORCH, SINGLE_SENTENCE_TASK, train_dataloader, validation_dataloader, test_dataloader)
    logits_teacher = teacher_single_sentence.run_teacher()

'''
fine tune BERT for single sentence classification
Neural Network in pytorch https://blog.paperspace.com/pytorch-101-building-neural-networks/
Softmax in pytorch https://pytorch.org/tutorials/beginner/nlp/deep_learning_tutorial.html
'''
'''
We fine-tune four models using the
Adam optimizer with learning rates {2, 3, 4, 5} ×10−5,
picking the best model on the validation set.


Fine-tune philosophy::
https://colab.research.google.com/drive/1ywsvwO6thOVOrfagjjfuxEf6xVRxbUNO#scrollTo=fwQ7JcuJQZ0o

For this task, we first want to modify the pre-trained BERT model to give
outputs for classification, and then we want to continue training the model on
our dataset until that the entire model, end-to-end, is well-suited for our
task. Thankfully, the huggingface pytorch implementation includes a set of
interfaces designed for a variety of NLP tasks. Though these interfaces are all
built on top of a trained BERT model, each has different top layers and output
types designed to accomodate their specific NLP task.

We'll load BertForSequenceClassification. This is the normal BERT model with an
added single linear layer on top for classification that we will use as a
sentence classifier. As we feed input data, the entire pre-trained BERT model
and the additional untrained classification layer is trained on our specific task.
'''

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertConfig
from transformers import AdamW, BertForSequenceClassification
from tqdm import tqdm, trange
import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt
% matplotlib inline
from configs.constant import *

from util.prepare_BERT_input import *
from util.embedding import *

class teacher():
    def __init__(self, model_path, model_task, train_dataloader, validation_dataloader, test_dataloader):
        self.best_val_acc = -1
        self.best_model = None
        self.model_path = model_path  # path of the pretrained BERT
        self.model_task = model_task  # what type of task is it? single sentence classification or senternce pair
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.test_dataloader = test_dataloader

    def load_model(self):
        if self.model_task == 'SINGLE_SENTENCE_TASK':
            model = BertForSequenceClassification.from_pretrained(self.model_path, num_labels=2)
        return model

    def create_optimizer(self, model, lr, no_decay = ['bias', 'gamma', 'beta']):
        '''
        TODO: use 4 different lr to find the best model in validation set
        '''
        param_optimizer = list(model.named_parameters())
        optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
         ]
        optimizer = AdamW(optimizer_grouped_parameters, lr, correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
        return optimizer

    def create_scheduler(self, optimizer, num_warmup_steps, num_total_steps):
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=num_warmup_steps, t_total=num_total_steps)  # PyTorch scheduler
        return scheduler

    # Function to calculate the accuracy of our predictions vs labels
    def compute_accuracy(self, preds, labels):
        pred_flat = preds
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    def model_train(self, train_dataloader, validation_dataloader, lr):
        train_loss = 0
        validation_accuracy = 0
        #model = self.load_model()
        model = BertForSequenceClassification.from_pretrained(self.model_path, num_labels=2)
        model.to(device)
        # Store our loss and accuracy for plotting
        train_loss_set = []
        # Number of training epochs (authors recommend between 2 and 4)
        epochs = 1 # we can try 4 instead
        # trange is a tqdm wrapper around the normal python range
        for _ in trange(epochs, desc="Epoch"):
            # Set our model to training mode (as opposed to evaluation mode)
            model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            # Train the data for one epoch
            for step, batch in enumerate(train_dataloader):
                # Add batch to GPU
                batch = tuple(t.to(device) for t in batch)
                # Unpack the inputs from our dataloader
                b_input_ids, b_input_mask, b_labels = batch
                optimizer = self.create_optimizer(model, lr, no_decay = ['bias', 'gamma', 'beta'])
                # Clear out the gradients (by default they accumulate)
                optimizer.zero_grad()
                # Forward pass
                outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
                loss = outputs[0]
                train_loss_set.append(loss)
                # Backward pass
                loss.backward()
                # Update parameters and take a step using the computed gradient
                optimizer.step()


                # Update tracking variables
                tr_loss += loss
                nb_tr_examples += b_input_ids.size(0)
                nb_tr_steps += 1
            train_loss = tr_loss/nb_tr_steps
            print("Train loss: {}".format(train_loss))
            # Put model in evaluation mode to evaluate loss on the validation set
            model.eval()
            # Tracking variables
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0
            # Evaluate data for one epoch
            for batch in validation_dataloader:
                # Add batch to GPU
                batch = tuple(t.to(device) for t in batch)
                # Unpack the inputs from our dataloader
                b_input_ids, b_input_mask, b_labels = batch
                # Telling the model not to compute or store gradients, saving memory and speeding up validation
                with torch.no_grad():
                    # Forward pass, calculate logit predictions
                    logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

                logits = np.asarray(logits)
                label_ids = b_labels.to('cpu').numpy()
                tmp_eval_accuracy = self.compute_accuracy(logits, label_ids)
                eval_accuracy += tmp_eval_accuracy
                nb_eval_steps += 1
            validation_accuracy = eval_accuracy/nb_eval_steps
            if validation_accuracy > self.best_val_acc:
                self.best_val_acc = validation_accuracy
                self.best_model = model
            print("Validation Accuracy: {}".format(validation_accuracy))

        #return model, validation_accuracy

    def run_teacher(self):
        # call model_train with 4 different lr
        learning_rate = hyper_param['lr']
        model_list = []
        val_acc_list = []
        for lr in learning_rate:
            self.model_train(self.train_dataloader, self.validation_dataloader, lr)
        # self.best_model has the model
        model = self.best_model

        #put it in eval mode
        # we can't create test_dataloader the  same way as other dataloader. So how to  find logits
        
        model.eval()
        for batch in self.test_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            # Telling the model not to compute or store gradients, saving memory and speeding up validation
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

            logits_teacher = np.asarray(logits)
        return logits_teacher

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch import optim
#import torchtext
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertConfig
from transformers import AdamW, BertForSequenceClassification, WarmupLinearSchedule
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from nltk import word_tokenize
from gensim.models import KeyedVectors, Word2Vec
import pandas as pd
import numpy as np
import operator
import logging
import gc
import os
import io



class studentModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, classification_hidden_size, dropout_rate):
        super(customModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bilstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        #self.fcl = nn.Linear(hidden_size*2, num_classes)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size*2, classification_hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(classification_hidden_size, num_classes)
        )

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, hidden = self.bilstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        #concat hidden state of forward and backword
        fw_hidden = out[-1, :, :self.hidden_size]
        bk_hidden = out[0, :, :self.hidden_size]
        features = torch.cat((fw_hidden, bk_hidden), dim = 1)
        logits = self.classifier(features)
        return logits, hidden

    def save_studentModel(model, output_dir):
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        torch.save(model.state_dict(), os.path.join(output_dir, "weights.pth"))


class StrudentTrainer():
    def __init__(self, model, device):
        self.model = model
        self.device = device
        assert self.loss_option in ["cross_entropy", "mse"]
        if self.loss_option == "cross_entropy":
            self.loss_function = nn.CrossEntropyLoss(reduction="sum")
        elif self.loss_option == "mse":
            self.loss_function = nn.MSELoss(reduction="sum")

    #TODO
    def get_distill_loss(self, logits_out, b_labels, batch_size):
        loss = self.loss_function(
            logits_out,
            b_labels
        ) / batch_size
        return loss

    def train(self, train_dataloader, validation_dataloader, lr):
        train_loss = 0
        validation_accuracy = 0
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
                student_logits = outputs[0]
                loss = self.get_distill_loss (student_logits, b_labels)
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
            return model

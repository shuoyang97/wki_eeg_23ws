# !pip install -r requirements.txt

import csv
import matplotlib.pyplot as plt
import numpy as np
import os
from wettbewerb import load_references, get_3montages, get_6montages
import mne
from scipy import signal as sig
from scipy.fft import fft
import ruptures as rpt
import json
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from torch.autograd import Variable 

class wki_model(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(wki_model, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        
        self.fc_1_1 = nn.Linear(60, hidden_size)
        self.lstm1 = nn.LSTM(self.input_size , self.hidden_size, self.num_layers)
        self.fc_1_2 =  nn.Linear(hidden_size, 20) #fully connected 1   
        self.fc_1_3 = nn.Linear(20, num_classes) #fully connected last layer
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.25)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):

        out = self.fc_1_1(x.unsqueeze(0))

        # out为tensor，尺寸为torch.Size([batch, 50])，将其转化尺寸为torch.Size([50, batch, 1])
        out = torch.transpose(out.unsqueeze(2), 0, 1)

        # Propagate input through LSTM
        _, (h_n, _) = self.lstm1(out) #lstm with input, hidden, and internal state

        # out = self.relu(out)
        out = self.fc_1_2(h_n[-1, :, :]) #first Dense
        out = self.relu(out) #relu
        out = self.dropout(out)
        out = self.fc_1_3(out) #Final Output
        out = self.sigmoid(out)

        return out


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from scaler import StandardScalerLSTM

SEQ_LEN = 8
BATCH_SIZE = 16
FEATURE_SIZE = 9 
# ====================== Model ======================== 
class PricePredictor(nn.Module):
    def __init__(self,scaler,batch_size, input_size=FEATURE_SIZE, hidden_layer_size=150, time_segment=5, output_size=1, device='cpu'):
        super().__init__()
        self.scaler = scaler
        self.hidden_layer_size = hidden_layer_size
        self.time_segment_length = time_segment
        self.lstm = nn.LSTM(input_size, hidden_layer_size,batch_first=True) # N x L x input_size(12)
        self.linear = nn.Linear(hidden_layer_size, output_size) # N x L x output_size(1)
        self.ReLU = nn.ReLU()
        # self.sigmoid = nn.Sigmoid()
        # cell double is zeros vector of shape ((1, batch_size, hidden_layer_size), (1, batch_size, hidden_layer_size))
        self.cell_double =   (torch.zeros(1,batch_size,hidden_layer_size,requires_grad=False).to(device),
                              torch.zeros(1,batch_size,hidden_layer_size,requires_grad=False).to(device))
        self.remember = False 

    def forward(self, input_seq):
        '''
            input : N x L x input_size
            ouput : N x output_size
        '''
        input_seq = self.scaler(input_seq)

        if self.remember:
            output,self.cell_double = self.lstm(input_seq,self.cell_double) # N x L x hidden_layer_size
        else:
            output,_ = self.lstm(input_seq) # N x L x hidden_layer_size

        # non-linear activation function
        output = self.ReLU(output)
        # output = self.sigmoid(output)
        self.cell_double = (self.cell_double[0].detach(),self.cell_double[1].detach())
        predictions = self.linear(output[:,-1,:].squeeze()) # N x output_size
        predictions = self.scaler.inverse_transform(predictions)
        return predictions
    
    def hidden_predict(self, input_seq):
        if self.cell_double[0].shape[1] != 1:
            # self.cell_double = (torch.zeros(1,1,self.hidden_layer_size,requires_grad=False).to(device), torch.zeros(1,1,self.hidden_layer_size,requires_grad=False).to(device))
             self.cell_double = (torch.zeros(1,1,self.hidden_layer_size,requires_grad=False), torch.zeros(1,1,self.hidden_layer_size,requires_grad=False))
        output, self.cell_double = self.lstm(input_seq, self.cell_double)
        output = self.ReLU(output)
        self.cell_double = (self.cell_double[0].detach(),self.cell_double[1].detach())
        predictions = self.linear(output[:,-1,:].squeeze()) # N x output_size
        predictions = self.scaler.inverse_transform(predictions)
        return predictions, self.cell_double[0]


# ====================== Training ========================

class Traning():
    def __init__(self, model, loss_function, optimizer, train_loader, test_loader, epochs=80,device='cpu'):
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.test_losses = []
        self.device = device
        self.patience = 3


    def early_stopping(self):
        return self.patience < len(self.test_losses) and all([self.test_losses[-i-1] > self.test_losses[-i-2] for i in range(self.patience)])

    def train(self):
        for i in range(self.epochs):
            for seq, targets in self.train_loader:
                if seq.shape[0]!=BATCH_SIZE: # if the last batch is not full
                    continue
                seq, targets = seq.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                y_pred = self.model(seq)
                single_loss = self.loss_function(y_pred, targets)
                single_loss.backward()
                self.optimizer.step()

            if(i%5==0):
                # print the test loss
                with torch.no_grad():
                    self.model.remember = True  # remember the last state of the cell
                    test_loss_ = 0
                    for seq_, targets_ in self.test_loader:
                        if(seq_.shape[0]!=BATCH_SIZE): # if the last batch is not full
                            continue
                        seq_, targets_ = seq_.to(self.device), targets_.to(self.device)
                        y_pred_test = self.model(seq_)
                        test_loss_ += self.loss_function(y_pred_test, targets_)
                        # print(y_pred_test.shape)
                    print(f'Test loss: {test_loss_.item():10.8f}')
                    self.test_losses.append(test_loss_.item())
                    if self.early_stopping():
                        break 
                    self.model.remember = False # forget the last state of the cell for further training
        

    def test(self):
        self.model.remember = True
        actual_prices = []
        predictions = []
        with torch.no_grad():
            test_loss = 0
            for seq, targets in self.test_loader:
                seq, targets = seq.to(self.device), targets.to(self.device)
                if seq.shape[0]!=BATCH_SIZE: # if the last batch is not full
                    continue
                y_pred = self.model(seq)
                targets = targets.cpu().numpy()
                y_pred = y_pred.cpu().numpy()
                actual_prices.extend(targets.flatten())
                predictions.extend(y_pred.flatten())

        actual_prices = np.array(actual_prices)
        predictions = np.array(predictions)
        x = np.arange(len(actual_prices))

        plt.plot(x,actual_prices, label="actual")
        plt.plot(x,predictions, label="predictions")
        plt.legend()
        plt.show()

    def save(self,name):
        torch.save(self.model.state_dict(), f"files/{name}.pth")
    
    def load(self,path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
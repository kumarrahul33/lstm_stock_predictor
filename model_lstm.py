
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
# ====================== Model ======================== 
class PricePredictor(nn.Module):
    def __init__(self,scaler,batch_size, input_size,train_remember=False, hidden_layer_size=150, output_size=1, device='cpu'):
        super().__init__()
        self.scaler = scaler
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size,batch_first=True) # N x L x input_size(12)
        self.linear = nn.Linear(hidden_layer_size, output_size) # N x L x output_size(1)
        self.ReLU = nn.ReLU()
        # self.sigmoid = nn.Sigmoid()
        # cell double is zeros vector of shape ((1, batch_size, hidden_layer_size), (1, batch_size, hidden_layer_size))
        self.cell_double =   (torch.zeros(1,batch_size,hidden_layer_size,requires_grad=False).to(device),
                              torch.zeros(1,batch_size,hidden_layer_size,requires_grad=False).to(device))
        self.cell_double_test =   (torch.zeros(1,1,hidden_layer_size,requires_grad=False).to(device),
                              torch.zeros(1,1,hidden_layer_size,requires_grad=False).to(device))
        self.remember = False 
        self.train_remember = train_remember 
        self.eval_mode = False
        # self.log = {
        #     "input_size" : input_size,
        #     "hidden_layer_size" : hidden_layer_size,
        #     "batch_size" : batch_size,
        # }

    def forward(self, input_seq):
        '''
            input : N x L x input_size
            ouput : N x output_size
        '''
        input_seq = self.scaler(input_seq)

        if self.remember:
            # output,self.cell_double = self.lstm(input_seq,self.cell_double) # N x L x hidden_layer_size
            if self.eval_mode:
                output,self.cell_double_test = self.lstm(input_seq,self.cell_double_test)
                self.cell_double_test = (self.cell_double_test[0].detach(),self.cell_double_test[1].detach())
            else:
                output,self.cell_double = self.lstm(input_seq,self.cell_double)
                self.cell_double = (self.cell_double[0].detach(),self.cell_double[1].detach())
        else:
            output,_ = self.lstm(input_seq) # N x L x hidden_layer_size


        # non-linear activation function
        output = self.ReLU(output)
        # output = self.sigmoid(output)
        predictions = self.linear(output[:,-1,:].squeeze()) # N x output_size
        predictions = self.scaler.inverse_transform(predictions)
        return predictions
    
    def hidden_predict(self, input_seq):
        if not hasattr(self, 'pred_self_state'):
            self.pred_self_state = (torch.zeros(1,1,self.hidden_layer_size,requires_grad=False), torch.zeros(1,1,self.hidden_layer_size,requires_grad=False))
        output, self.pred_self_state = self.lstm(input_seq, self.pred_self_state)
        output = self.ReLU(output)
        self.pred_self_state = (self.pred_self_state[0].detach(),self.pred_self_state[1].detach())
        predictions = self.linear(output[:,-1,:].squeeze()) # N x output_size
        predictions = self.scaler.inverse_transform(predictions)
        return predictions, self.pred_self_state[0]


# ====================== Training ========================

class Traning():
    def __init__(self, model, loss_function, optimizer, train_loader, test_loader,test_plot_loader,name, epochs=80,device='cpu'):
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.test_losses = []
        self.device = device
        self.patience = 5 
        self.test_plot_loader = test_plot_loader
        self.save_name = name 


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
                y_pred = y_pred.squeeze()
                targets = targets.squeeze()
                single_loss = self.loss_function(y_pred, targets)
                single_loss.backward()
                self.optimizer.step()

            if(i%5==0):
                # print the test loss
                # validation
                with torch.no_grad():
                    self.model.remember = True  # remember the last state of the cell
                    val_loss_ = 0
                    for seq_, targets_ in self.test_loader:
                        if(seq_.shape[0]!=BATCH_SIZE): # if the last batch is not full
                            continue
                        seq_, targets_ = seq_.to(self.device), targets_.to(self.device)
                        y_pred_test = self.model(seq_)
                        y_pred_test = y_pred_test.squeeze()
                        targets_ = targets_.squeeze()
                        val_loss_ += self.loss_function(y_pred_test, targets_)
                        # print(y_pred_test.shape)
                    print(f'Validation loss: {val_loss_.item():10.8f}')
                    self.test_losses.append(val_loss_.item())
                    if self.early_stopping():
                        break 
                    self.model.remember = self.model.train_remember# forget the last state of the cell for further training
        

    def test(self):
        self.model.eval_mode = True
        self.model.remember = True
        actual_prices = []
        predictions = []
        with torch.no_grad():
            test_loss = 0
            i = 0
            for seq, targets in self.test_plot_loader:
                # if(seq.shape[0]!=BATCH_SIZE): # if the last batch is not full
                #     continue
                # print("seq.shape: ", seq.shape)
                i += seq.shape[0]
                seq, targets = seq.to(self.device), targets.to(self.device)
                y_pred = self.model(seq)
                y_pred = y_pred.squeeze()
                targets = targets.squeeze()
                targets = targets.cpu().numpy()
                y_pred = y_pred.cpu().numpy()
                actual_prices.extend(targets.flatten())
                predictions.extend(y_pred.flatten())

            print("total_days_passed: ", i)
        actual_prices = np.array(actual_prices)
        predictions = np.array(predictions)
        x = np.arange(len(actual_prices))
        # clear previous plots
        plt.clf()
        plt.plot(x[2:],actual_prices[2:], label="actual")
        plt.plot(x[2:],predictions[2:], label="predictions")
        plt.legend()
        # save the plot
        #  get the seq of the first batch
        # seq_,_ = next(iter(self.train_loader))
        # plt.savefig(f"results/{self.save_name}.png")
        plt.show()

    def save_losses(self):
        # save the test losses
        seq_,_ = next(iter(self.train_loader))
        df = pd.DataFrame(self.test_losses)
        df.to_csv(f"results/"+self.save_name+"_losses.csv",index=False)


    def save(self,name):
        torch.save(self.model.state_dict(), f"files/{name}.pth")
    
    def load(self,path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
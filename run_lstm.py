# DATA Segment
# ================================imports and constants===============================
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from scaler import StandardScalerLSTM
from model_lstm import PricePredictor, Traning

SEQ_LEN = 8
BATCH_SIZE = 16
FEATURE_SIZE = 9 

# Check if CUDA (GPU support) is available
device = 'cpu'
if torch.cuda.is_available():
    print("CUDA is available! You can use the GPU.")
    device = 'cuda'
else:
    print("CUDA is not available. You'll be using CPU.")
# ================================data processing===============================

data_df = pd.read_csv("data/final_dataset.csv")

# date drop
# data_df = data_df.drop(['Date'], axis=1)
# data_df = data_df.iloc[1900:,:]
data = np.nan_to_num(np.array(data_df, dtype=np.float32))

# Scaler
minmax_scaler = StandardScalerLSTM(batch_size=BATCH_SIZE, sequence_length=SEQ_LEN, device=device)
minmax_scaler.fit(data)

def create_sequence(data,seq_len):
    xs = []
    ys = []
    for i in range(len(data)-seq_len-1):
        x = data[i:(i+seq_len),:]
        # print(x.shape)
        y = data[i+seq_len,0]
        xs.append(x)
        ys.append(y)
    return np.array(xs),np.array(ys)

inputs , targets = create_sequence(data,SEQ_LEN)
inputs=torch.from_numpy(inputs);targets=torch.from_numpy(targets)


# split the input data into train and test data
train_size = int(0.9 * len(inputs))
test_size = len(inputs) - train_size
train_inputs, test_inputs = inputs[:train_size], inputs[train_size:]
train_targets, test_targets = targets[:train_size], targets[train_size:]
print(train_inputs.shape, test_inputs.shape)


# ====================== Data Loader ========================
train_dataset = TensorDataset(train_inputs, train_targets)
test_dataset = TensorDataset(test_inputs, test_targets)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = PricePredictor(minmax_scaler,BATCH_SIZE,device=device).to(device)
loss_function = nn.MSELoss()
# loss_function = nn.L1Loss()
# optimizer = torch.optim.Adagrad(model.parameters(), lr=0.0001)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
epochs = 80

    
if __name__ == "__main__":
    # ask for the model name in key args
    name = input("Enter the model name: ") 
    trainer = Traning(model, loss_function, optimizer, train_loader, test_loader, epochs=epochs,device=device)
    trainer.train()
    trainer.test()
    trainer.save(name)


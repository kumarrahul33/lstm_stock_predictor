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
FEATURE_SIZE = 8
DATASET_PATH = "data/final_dataset_ind_mod.csv"

# Check if CUDA (GPU support) is available
# device = 'cpu'
# if torch.cuda.is_available():
#     print("CUDA is available! You can use the GPU.")
#     device = 'cuda'
# else:
#     print("CUDA is not available. You'll be using CPU.")
# ================================data processing===============================

def get_minmax(device):
    data_df = pd.read_csv(DATASET_PATH)
    data = np.nan_to_num(np.array(data_df, dtype=np.float32))

    minmax_scaler = StandardScalerLSTM(batch_size=BATCH_SIZE, sequence_length=SEQ_LEN, device=device)
    minmax_scaler.fit(data)
    return minmax_scaler

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

# def create_plo(data):
#     seq_len = 1
#     xs = []
#     ys = []
#     for i in range(len(data)-seq_len-1):
#         x = data[i:(i+seq_len),:]
#         # print(x.shape)
#         y = data[i+seq_len,0]
#         xs.append(x)
#         ys.append(y)
#     return np.array(xs),np.array(ys)



def make_loaders(batch_size=BATCH_SIZE,seq_len=SEQ_LEN,splits=[.7,.15]):
    data_df = pd.read_csv(DATASET_PATH)
    data = np.nan_to_num(np.array(data_df, dtype=np.float32))

    inputs , targets = create_sequence(data,seq_len=seq_len)
    inputs=torch.from_numpy(inputs)
    targets=torch.from_numpy(targets)

    train_size = int(splits[0] * len(inputs))
    val_size =  int(train_size + splits[1] * len(inputs))
    train_inputs, val_inputs, test_inputs= inputs[:train_size], inputs[train_size:val_size], inputs[val_size:]
    train_targets, val_targets, test_targets = targets[:train_size], targets[train_size:val_size], targets[val_size:]

    # ====================== Data Loader ========================
    train_dataset = TensorDataset(train_inputs, train_targets)
    test_dataset = TensorDataset(test_inputs, test_targets)
    val_dataset = TensorDataset(val_inputs, val_targets)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

    
if __name__ == "__main__":
    # ask for the model name in key args
    if(torch.cuda.is_available()):
        device = 'cuda'
    else:
        device = 'cpu'
    minmax_scaler = get_minmax(device)

    loss_function = nn.MSELoss()

    # train_loader,test_loader = make_loaders()
    # Experiment : 1
    # for seq_len in [1]:
    #     for tr in [False]:
    #         model = PricePredictor(minmax_scaler,BATCH_SIZE,train_remember=tr,device=device,input_size=FEATURE_SIZE).to(device)
    #         train_loader,val_loader,_= make_loaders(batch_size=BATCH_SIZE,seq_len=8)
    #         _,_,test_plot_loader= make_loaders(batch_size=1,seq_len=1)
    #         optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    #         epochs = 300

    #         name = "model_"
    #         trainer = Traning(model, loss_function, optimizer, train_loader, val_loader,test_plot_loader,name=name, epochs=epochs,device=device)
    #         trainer.train()
    #         trainer.test()
    #         trainer.save_losses()


    model = PricePredictor(minmax_scaler,BATCH_SIZE,train_remember=False,device=device,input_size=FEATURE_SIZE).to(device)
    train_loader,val_loader,_= make_loaders(batch_size=BATCH_SIZE,seq_len=SEQ_LEN)
    _,_,test_plot_loader= make_loaders(batch_size=1,seq_len=8)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=0.001)
    epochs = 80 

    name = "model_IND_wo"
    trainer = Traning(model, loss_function, optimizer, train_loader, val_loader,test_plot_loader,name=name, epochs=epochs,device=device)
    trainer.train()
    trainer.test()
    trainer.save_losses()
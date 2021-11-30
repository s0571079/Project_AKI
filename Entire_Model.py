import torch
import torch.nn as nn
import torch.optim as optim
import random
import pickle
import CustomMultiInput_LSTM as lstm
import numpy
import matplotlib.pyplot as plt
from os import listdir
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


"""
Describes the entire network architecture
For visualisation see './Grafiken/EntireNetwork_Architecture.png'

Steps which happen here:

READ INPUT (DataSet class)
- Read the previously generated pickle files (by Generate_Samples.py)
- Divide them into chunks of size T = 22
- Convert them from DataFrames to arrays/lists/dicts

DEFINITION OF BASIC NETWORK ARCHITECTURE
- Definition of the four main layers CustomLSTM -> LSTM -> ReLu -> Linear
- Definition of some pre-settings like hidden_size & batch_size
- Definition of the forward steps between the layers

FEED AND EVALUATE
- Feed the pickle data into the network
- Calculate error
- Plot results

"""
class Net(nn.Module):

    def __init__(self, seq_size, hidden_size):
        super(Net, self).__init__()
        self.seq_size = seq_size

        # CustomLSTM -> LSTM -> ReLu -> Linear
        self.MI_LSTM_layer = lstm.CustomMultiInputLSTM(5, hidden_size)
        self.StandardLSTM_layer = nn.LSTM(hidden_size, hidden_size)
        self.relu_layer = nn.ReLU()
        self.lin_layer = nn.Linear(hidden_size, 1)

    def forward(self, Y, x1, x2, x3, x4, x5, x6, x7):
        # Executed when input is passed into the neural network
        output, hidden = self.MI_LSTM_layer(Y, x1, x2, x3, x4, x5, x6, x7)
        # output shape = 1, 64 // hidden = 1, 22, 64

        # Note that the hidden sequence is passed here
        hidden = torch.transpose(hidden, 0, 1)
        output, hidden = self.StandardLSTM_layer(hidden)
        # output shape = 1, 22, 64

        output = self.relu_layer(hidden[0])
        # output shape = 1, 22, 64

        output = self.lin_layer(output)
        # output shape = 1, 22, 1

        # I need to get the output shape as one single value
        return output


class StockDataSet(Dataset):

    def __init__(self):
        # make a list containing the path to all your pkl files
        self.paths = listdir('./Pickle/')
        random.shuffle(self.paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        with open('./Pickle/' + self.paths[idx], 'rb') as f:
            item = pickle.load(f)
        # [ [Liste 22x5 (Open,Close,Volume, Min, Max], (Talib-Klasse1, 22x5), (Talib-Klasse2, 22x 3) ... ]
        Y = item[["Open", "High", "Low", "Close", "Volume"]].to_numpy()
        y = item.loc[item.index[-1], "Close"]

        print(f.name)
        Y = Y[:-1]

        # Overlap Studies
        x_1 = item[["upperBB", "middleBB", "lowerBB", "midpoint", "wma"]].to_numpy()
        # Momentum Indicators
        x_2 = item[["mom", "mfi", "bop"]].to_numpy()
        # Volume Indicators
        x_3 = item[["adline", "adosc", "obv"]].to_numpy()
        # Volability Indicators
        x_4 = item[["atr", "natr", "tr"]].to_numpy()
        # Price Transform
        x_5 = item[["avgprice", "typprice", "wclprice"]].to_numpy()
        # Pattern Recognition
        x_6 = item[["whitesoldiers", "starsinsouth", "twocrows"]].to_numpy()
        # Statistic Functions
        x_7 = item[["linearreg", "stddev", "tsf"]].to_numpy()

        return Y, y, x_1, x_2, x_3, x_4, x_5, x_6, x_7


# The time window is 22 days
T = 22
numberOfNodesPerLayer = 64
batch_size = 1
loss_plot_values = []

dataset = StockDataSet()
loader = DataLoader(dataset=dataset, batch_size=batch_size)

net = Net(T, numberOfNodesPerLayer)

criterion = nn.L1Loss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(10):

    running_loss = 0

    # T = 22
    for Y, y, x1, x2, x3, x4, x5, x6, x7 in loader:
        optimizer.zero_grad()
        # Shapes:  Y = (1, 5, 22); x1 = (1, 3 ,22) (bei 3 Kennzahlen pro Klasse); x2 = ...
        #print("Y: " + str(Y))
        #print("x1: " + str(x1) + "/ x2:" + str(x2) + "/ x3:" + str(x3) + "/ x4:" + str(x4) + "/ x5:" + str(x5) + "/ x6:" + str(x6) + "/ x7:" + str(x7))
        #print("y: " + str(y))
        outputs = net(Y.float(), x1.float(), x2.float(), x3.float(), x4.float(), x5.float(), x6.float(), x7.float())
        output2 = torch.squeeze(outputs.float())
        y2 = torch.squeeze(y.float())
        print("Netzwerkoutput: " + str(output2) + " / Label: " + str(y2))
        loss = criterion(output2, y2)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        #print(loss.item())
        loss_plot_values.append(running_loss / len(loader))

    print('Epoch loss: ' + str(running_loss / len(loader)))


#Plot results
plt.plot(loss_plot_values)
plt.ylabel('loss')
plt.xlabel('')
plt.show()
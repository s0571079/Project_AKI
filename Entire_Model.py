import torch
import torch.nn as nn
import torch.optim as optim
import random
import pickle
import CustomMultiInput_LSTM as lstm
import numpy
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
        # output = 1, 64 // hidden = 1, 22, 64
        output, hidden = self.StandardLSTM_layer(hidden)
        # Note that the hidden sequence is passed here
        output = self.relu_layer(output)
        output = self.lin_layer(output)
        return output


class StockDataSet(Dataset):

    def __init__(self):
        # make a list containing the path to all your pkl files
        self.paths = listdir('c:/data/htw/KI_Project/Samples/')
        random.shuffle(self.paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        with open('c:/data/htw/KI_Project/Samples/' + self.paths[idx], 'rb') as f:
            item = pickle.load(f)
        # Für jede einzelne Aktie, jeden einzelnen Zeitpunkt werden die Daten hier ausgelesen
        # [ [Liste 22x5 (Open,Close,Volume, Min, Max], (Talib-Klasse1), (Talib-Klasse2) ... ]
        # Y = 22x5 (Open,Close,Volume, Min, Max) df
        # y = labels -> Wert, der vorhergesagt werden soll (y+1) -> also ein float Wert
        # x_1 = Klasse 1: 22 x 3 (bei 3 Talib Kennzahlen)
        # x_2 = Klasse 2: 22 x 3 (bei 3 Talib Kennzahlen)
        # ...
        # x_7 = Klasse 8: 22 x 3 (bei 3 Talib Kennzahlen)
        # Y = item['Y'].to_numpy()
        # y = item['y']
        # X_p = item['X_p'].to_numpy()
        # X_n = item['X_n'].to_numpy()

        # No dataframes here, only dicts and lists
        numpyArrayNumbers = random.sample(range(100, 300), 22)
        numpyArrayNumbersKennzahlen = random.sample(range(3, 30), 22)

        yData = numpy.array(
            [numpyArrayNumbers, numpyArrayNumbers, numpyArrayNumbers, numpyArrayNumbers, numpyArrayNumbers])
        y = 0.234
        xData = numpy.array([numpyArrayNumbersKennzahlen, numpyArrayNumbersKennzahlen, numpyArrayNumbersKennzahlen])

        return yData, y, xData, xData, xData, xData, xData, xData, xData


# The time window is 22 days
T = 22
numberOfNodesPerLayer = 64
batch_size = 1

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
        outputs = net(Y.float(), x1.float(), x2.float(), x3.float(), x4.float(), x5.float(), x6.float(), x7.float())
        loss = criterion(torch.squeeze(outputs), y.float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        print(loss.item())

    print('Epoch loss: ' + str(running_loss / len(loader)))


# TODO: Plot results

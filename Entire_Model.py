import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
from os import listdir
import random
import pickle
import CustomMultiInput_LSTM as lstm
import pandas
import numpy

# Entire network architecture
class Net(nn.Module):

    def __init__(self, seq_size, hidden_size):
        super(Net, self).__init__()
        self.seq_size = seq_size

        # ? Sequence Size?

        # Network architecture is described here; see network_architecture.png
        # CustomLSTM -> LSTM -> ReLu -> Linear
        # QUESTION: Why 1 here?; Why same hidden size each here? Hidden size hier 64? Kann ich hidden_size beliebig variieren?
        self.MI_LSTM_layer = lstm.CustomMultiInputLSTM(5, hidden_size)
        self.StandardLSTM_layer = nn.LSTM(hidden_size, hidden_size)
        self.relu_layer = nn.ReLU()
        self.lin_layer = nn.Linear(hidden_size, 1)

    def forward(self, Y, x1, x2, x3, x4, x5, x6, x7, x8):
        # Executed when input is passed into the neural network
        # Put the data through the layers -> CustomLSTM -> LSTM -> ReLu -> Linear
        # Pass sequence as parameter for second layer
        # output = 1, 64, hidden = 1 22, 64
        output, hidden = self.MI_LSTM_layer(Y, x1, x2, x3, x4, x5, x6, x7, x8)
        output, hidden = self.StandardLSTM_layer(hidden)
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
        # x_8 = Klasse 8: 22 x 3 (bei 3 Talib Kennzahlen)
        #Y = item['Y'].to_numpy()
        #y = item['y']
        #X_p = item['X_p'].to_numpy()
        #X_n = item['X_n'].to_numpy()


        # No dataframes here, only dicts and lists
        numpyArrayNumbers = random.sample(range(100, 300), 22)
        numpyArrayNumbersKennzahlen = random.sample(range(3, 30), 22)

        yData = numpy.array([numpyArrayNumbers,numpyArrayNumbers,numpyArrayNumbers,numpyArrayNumbers,numpyArrayNumbers])
        y = 0.234
        xData = numpy.array([numpyArrayNumbersKennzahlen, numpyArrayNumbersKennzahlen, numpyArrayNumbersKennzahlen])

        return yData, y, xData, xData, xData, xData, xData, xData, xData, xData


# QUESTION: Sqc Size hier 22?
T = 22
q = 64

# Read the pickle files here
# QUESTION: What to do with batch size here?
batch_size = 1 # gibt an, wieviele Daten jeweils reingeladen werden zum trainieren (Speicher)
dataset = StockDataSet()
loader = DataLoader(dataset=dataset, batch_size=batch_size)

# QUESTION q ist hidden size? kann man variieren?
net = Net(T, q)

criterion = nn.L1Loss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
item = [({'Kennzahl1': [random.sample(range(1, 30), 22)], 'Kennzahl2': [random.sample(range(1, 30), 22)], 'Kennzahl3': [random.sample(range(1, 30), 22)]})]

with open('c:/data/htw/KI_Project/Samples/test2.pkl', 'wb') as f:
    pickle.dump(item, f, pickle.HIGHEST_PROTOCOL)

for epoch in range(10):

    running_loss = 0

    # Loop through every record
    for Y, y, x1, x2, x3, x4, x5, x6, x7, x8 in loader:
        optimizer.zero_grad()
        # Y = 1, 5, 22
        # x1 = 1, 3 ,22
        # x2 = gleich
        # y = 1 Wert (bei batch size 1); wird zu liste der Größe Anzahl batch size
        print(Y)
        # Input ins Netzwerk -> (Liste 22x5 (Open,Close,Volume, Min, Max), (Talib-Klasse1), (Talib-Klasse2) ... (Talib-Klasse8)]
        outputs = net(Y.float(), x1.float(), x2.float(), x3.float(), x4.float(), x5.float(), x6.float(), x7.float(), x8.float())
        # Vergleich mit labels
        loss = criterion(torch.squeeze(outputs), y.float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        print(loss.item())

    print('Epoch loss: ' + str(running_loss / len(loader)))

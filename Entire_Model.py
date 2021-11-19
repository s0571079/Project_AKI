import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
from os import listdir
import random
import pickle
import CustomMultiInput_LSTM as milstm

# = Grobgerüst Gesamtmodell (mit den 6 inputs links) -> y +1 output
class Net(nn.Module):

    def __init__(self, seq_size, hidden_size):
        super(Net, self).__init__()
        self.seq_size = seq_size

        # ? Sequence Size?

        # Netzwerkarchitektur
        self.MI_LSTM_layer = milstm.CustomMultiInput_LSTM(1, hidden_size)
        self.Y_layer = nn.LSTM(hidden_size, hidden_size)
        self.relu_layer = nn.ReLU()
        self.lin_layer = nn.Linear(hidden_size, 1)

    def forward(self, y, x1, x2, x3, x4, x5, x6, x7, x8, x9): # wird ausgeführt sobald input reinkommt;

        # Put the data through the layers -> CustomLSTM -> LSTM -> ReLu -> Linear
        Y_tilde, Y_tilde_sqc_hidden = self.MI_LSTM_layer(y, x1, x2, x3, x4, x5, x6, x7, x8, x9)
        Y_tilde = self.Y_layer(Y_tilde_sqc_hidden, Y_tilde)
        Y_tilde = self.relu_layer(Y_tilde)
        output = self.lin_layer(Y_tilde)
        return output


class CustomDataset(Dataset):

    def __init__(self):
        # make a list containing the path to all your pkl files
        self.paths = listdir('c:/data/htw/KI_Project/Samples/')
        random.shuffle(self.paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        with open('c:/data/htw/2KI_Project/Samples/' + self.paths[idx], 'rb') as f:
            item = pickle.load(f)
        # Für jede einzelne Aktie, jeden einzelnen Zeitpunkt werden die Daten hier ausgelesen
        Y = item['Y'].to_numpy()
        y = item['y']
        X_p = item['X_p'].to_numpy()
        X_n = item['X_n'].to_numpy()
        return Y, y, X_p, X_n


T = 20
batch_size = 512
q = 64

# Read the pickle files here
dataset = CustomDataset()
loader = DataLoader(dataset=dataset, batch_size=batch_size)

net = Net(T, q)

criterion = nn.L1Loss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(10):

    running_loss = 0

    # Loop through every record
    for Y, labels, X_p, X_n in loader:
        optimizer.zero_grad()
        # Input ins Netzwerk -> Hier Anzahl Parameterin Netzwerk = (Liste 22x(5+10) (Open,Close,Volume...) ) + (TalibWert_1) + (TalibWert_2) ...?
        # Anzahl Parameter?
        outputs = net(Y.float(), X_p.float(), X_n.float())
        # Vergleich mit labels
        loss = criterion(torch.squeeze(outputs), labels.float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        print(loss.item())

    print('Epoch loss: ' + str(running_loss / len(loader)))

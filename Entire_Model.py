import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
from os import listdir
import random
import pickle
import custom_lstm as lstm
import MultiInput_LSTM as milstm

# = Grobgerüst Gesamtmodell (mit den 6 inputs links) -> y +1 output
class Net(nn.Module):

    def __init__(self, seq_size, hidden_size):
        super(Net, self).__init__()
        self.seq_size = seq_size
        # Das ist das ganz links oben für y Input
        self.Y_layer = lstm.CustomLSTM(1, hidden_size)

        # Definition der 2 LSTM Layer + Relu + Linear

        # Modulelist = Liste an P layers
        self.X_p_layers = nn.ModuleList()
        self.X_n_layers = nn.ModuleList()
        # Je nachdem wieviel Korrelationen (Firmen im Bsp waren es 15) haben -> füge ich layer hinzu
        for i in range(self.seq_size):
            self.X_p_layers.append(lstm.CustomLSTM(1, hidden_size))
            self.X_n_layers.append(lstm.CustomLSTM(1, hidden_size))

        # Weitere Layer, die müssen selbst gebaut werden
        self.MI_LSTM_layer = milstm.MultiInputLSTM(hidden_size, hidden_size)
        self.Attention_layer = milstm.Attention(hidden_size, hidden_size)

        self.lin_layer = nn.Linear(hidden_size, 1)

    def forward(self, Y, X_n, X_p): # wird ausgeführt sobald input reinkommt; (Input aus net Methode glaube ich) -> also hier anpassen
        # Logik Gesamtnetzwerkarchitektur
        # Y wird durch den y layer durchcgeführt (links oben in Bild)
        # Y_tilde_hidden = gesamte Sequenz die nach oben gegeben wird; wenn nächster Layer Feed Forward wäre, dann bräuchten wir nur Y_tilde
        # Sequenzielle Netzwerke brauchen als output immer eine Sequenz

        # Hier wird zunächst der Input ins erste Multi-Input LSTM gegeben
        # Danach dessen Input (Sequence) wird Input für das nächste Custom LSTM

        Y_tilde, Y_tilde_hidden = self.Y_layer(Y.unsqueeze(2))

        X_p_list = list()
        X_n_list = list()
        # Für jede der z .B. 15 Korrelations Firmen, durch LSTM und output in Liste
        for i in range(self.seq_size):
            # Erste korrelierte Aktie durch den Layer
            X_p_out, X_p_hidden = self.X_p_layers[i](X_p[:,:,i:i+1])
            # In die Liste -> x~1p (auf Bild) in Liste
            X_p_list.append(X_p_hidden)
            X_n_out, X_n_hidden = self.X_n_layers[i](X_n[:,:,i:i+1])
            X_n_list.append(X_n_hidden)

        # Stack meine Liste (baue Tensor) -> auf Tensor dann Durchschnitt anwenden
        X_p_tensor = torch.stack(X_p_list)
        P_tilde = torch.mean(X_p_tensor, 0)
        X_n_tensor = torch.stack(X_n_list)
        N_tilde = torch.mean(X_n_tensor, 0)

        # Index haben wir weggelassen
        Y_tilde_prime_out, Y_tilde_prime_hidden = self.MI_LSTM_layer(Y_tilde_hidden, P_tilde, N_tilde)

        # Sequenz durch attention layer
        y_tilde = self.Attention_layer(Y_tilde_prime_hidden)
        output = torch.relu(self.lin_layer(y_tilde))
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

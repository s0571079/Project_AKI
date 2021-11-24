import torch
import torch.nn as nn
import math

"""
Sequenziell -> Inputs zu mehreren Zeitpunkten;
Vermeidet Problem (<-> RNN) - vanishing gradients (Umso weiter entfernt von Zeitpunkt t umso weniger Einfluss)
"Long Short Term Memory"
Gates als Konzept: Welche Inputs sind wichtig und welche können 'vergessen' werden
"""
# QUESTION Bild in Paper: Alles gleich nur separate Input Gates?
# QUESTION Erste 'Gate' ohne Inputs, was auch in den Attention Layer fliesst -> fliesst ja Y ein -> bleibt gleich bei uns?
class CustomMultiInputLSTM(nn.Module):
    def __init__(self, input_sz: int, hidden_sz: int):
        super().__init__()
        self.input_size = input_sz
        self.hidden_size = hidden_sz

        # Definiton of weights and biases
        kennzahl_input_sz = 3

        # i_t
        self.W_i = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_i = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_i = nn.Parameter(torch.Tensor(hidden_sz))

        # x1
        self.W_i_x1 = nn.Parameter(torch.Tensor(kennzahl_input_sz, hidden_sz))
        self.U_i_x1 = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_i_x1 = nn.Parameter(torch.Tensor(hidden_sz))

        # x2
        self.W_i_x2 = nn.Parameter(torch.Tensor(kennzahl_input_sz, hidden_sz))
        self.U_i_x2 = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_i_x2 = nn.Parameter(torch.Tensor(hidden_sz))

        # x3
        self.W_i_x3 = nn.Parameter(torch.Tensor(kennzahl_input_sz, hidden_sz))
        self.U_i_x3 = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_i_x3 = nn.Parameter(torch.Tensor(hidden_sz))

        # x4
        self.W_i_x4 = nn.Parameter(torch.Tensor(kennzahl_input_sz, hidden_sz))
        self.U_i_x4 = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_i_x4 = nn.Parameter(torch.Tensor(hidden_sz))

        # x5
        self.W_i_x5 = nn.Parameter(torch.Tensor(kennzahl_input_sz, hidden_sz))
        self.U_i_x5 = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_i_x5 = nn.Parameter(torch.Tensor(hidden_sz))

        # x6
        self.W_i_x6 = nn.Parameter(torch.Tensor(kennzahl_input_sz, hidden_sz))
        self.U_i_x6 = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_i_x6 = nn.Parameter(torch.Tensor(hidden_sz))

        # x7
        self.W_i_x7 = nn.Parameter(torch.Tensor(kennzahl_input_sz, hidden_sz))
        self.U_i_x7 = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_i_x7 = nn.Parameter(torch.Tensor(hidden_sz))

        # x8
        self.W_i_x8 = nn.Parameter(torch.Tensor(kennzahl_input_sz, hidden_sz))
        self.U_i_x8 = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_i_x8 = nn.Parameter(torch.Tensor(hidden_sz))

        # f_t
        self.W_f = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_f = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_f = nn.Parameter(torch.Tensor(hidden_sz))

        # c_t
        self.W_c = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_c = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_c = nn.Parameter(torch.Tensor(hidden_sz))

        # x1
        self.W_c_x1 = nn.Parameter(torch.Tensor(kennzahl_input_sz, hidden_sz))
        self.U_c_x1 = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_c_x1 = nn.Parameter(torch.Tensor(hidden_sz))

        # x2
        self.W_c_x2 = nn.Parameter(torch.Tensor(kennzahl_input_sz, hidden_sz))
        self.U_c_x2 = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_c_x2 = nn.Parameter(torch.Tensor(hidden_sz))

        # x3
        self.W_c_x3 = nn.Parameter(torch.Tensor(kennzahl_input_sz, hidden_sz))
        self.U_c_x3 = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_c_x3 = nn.Parameter(torch.Tensor(hidden_sz))

        # x4
        self.W_c_x4 = nn.Parameter(torch.Tensor(kennzahl_input_sz, hidden_sz))
        self.U_c_x4 = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_c_x4 = nn.Parameter(torch.Tensor(hidden_sz))

        # x5
        self.W_c_x5 = nn.Parameter(torch.Tensor(kennzahl_input_sz, hidden_sz))
        self.U_c_x5 = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_c_x5 = nn.Parameter(torch.Tensor(hidden_sz))

        # x6
        self.W_c_x6 = nn.Parameter(torch.Tensor(kennzahl_input_sz, hidden_sz))
        self.U_c_x6 = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_c_x6 = nn.Parameter(torch.Tensor(hidden_sz))

        # x7
        self.W_c_x7 = nn.Parameter(torch.Tensor(kennzahl_input_sz, hidden_sz))
        self.U_c_x7 = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_c_x7 = nn.Parameter(torch.Tensor(hidden_sz))

        # x8
        self.W_c_x8 = nn.Parameter(torch.Tensor(kennzahl_input_sz, hidden_sz))
        self.U_c_x8 = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_c_x8 = nn.Parameter(torch.Tensor(hidden_sz))

        # o_t
        self.W_o = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_o = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_o = nn.Parameter(torch.Tensor(hidden_sz))

        # a_t
        # QUESTION: Verständnis -> was ist das? nicht im Schaubild abgebildet
        self.W_a = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_a = nn.Parameter(torch.Tensor(hidden_sz))

        self.init_weights()

    def init_weights(self):
        # Initialisierung der Gewichte
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
        torch.nn.init.zeros_(self.W_c_x1)
        torch.nn.init.zeros_(self.U_c_x1)
        torch.nn.init.zeros_(self.b_c_x1)
        torch.nn.init.zeros_(self.W_c_x2)
        torch.nn.init.zeros_(self.U_c_x2)
        torch.nn.init.zeros_(self.b_c_x2)
        torch.nn.init.zeros_(self.W_c_x3)
        torch.nn.init.zeros_(self.U_c_x3)
        torch.nn.init.zeros_(self.b_c_x3)
        torch.nn.init.zeros_(self.W_c_x4)
        torch.nn.init.zeros_(self.U_c_x4)
        torch.nn.init.zeros_(self.b_c_x4)
        torch.nn.init.zeros_(self.W_c_x5)
        torch.nn.init.zeros_(self.U_c_x5)
        torch.nn.init.zeros_(self.b_c_x5)
        torch.nn.init.zeros_(self.W_c_x6)
        torch.nn.init.zeros_(self.U_c_x6)
        torch.nn.init.zeros_(self.b_c_x6)
        torch.nn.init.zeros_(self.W_c_x7)
        torch.nn.init.zeros_(self.U_c_x7)
        torch.nn.init.zeros_(self.b_c_x7)
        torch.nn.init.zeros_(self.W_c_x8)
        torch.nn.init.zeros_(self.U_c_x8)
        torch.nn.init.zeros_(self.b_c_x8)

    def forward(self, Y, x1, x2, x3, x4, x5, x6, x7, x8):
        # QUESTION -> verändern hier so ok?
        bs, _, seq_sz = Y.size()
        #-> output // seq_sz: 22; bs: 1; _: 5

        # alt: bs, seq_sz, _ = Y.size()
        #-> output // seq_sz: 5; bs: 1; _: 22

        hidden_seq = []
        # init hidden State mit 0: 'leerer' Zustand, da kein State vorhanden im ersten Zeitpunkt
        # C_T auch initialisiert = Carrystate für die 'Langzeitbeobachtung'
        h_t, c_t = (
            torch.zeros(bs, self.hidden_size).to(Y.device),
            torch.zeros(bs, self.hidden_size).to(Y.device),
        )

        # Nach und nach durch die Sequenz (seq_sz müsste t sein -> also 22?)
        # Im Update-Step berechnen wir die States für jede einzelne Zeiteinheit (mit den gleichen Gewichten [=LSTM])
        for t in range(seq_sz):
            # Jede TaLib Klasse bekommt separates Gate

            # Open/Close/Volume ... jeweils zu den verschiedenen (22) Zeitpunkten
            # QUESTION: ist Y_t (Open/Close/Volume/Min/Max) von einem EINZELNEN TAG? - müsste es dann (1,5) sein?
            # Y = 1, 5, 22
            Y_t = Y[:, :, t] # Müsste 1, 5 (Open/Close/Volume/Min/Max) sein oder?
            x1_t = x1[:, :, t] # Müsste 1, 3 (Kennzahl1/Kennzahl2/Kennzahl3) sein oder?
            x2_t = x2[:, :, t] # ...
            x3_t = x3[:, :, t]
            x4_t = x4[:, :, t]
            x5_t = x5[:, :, t]
            x6_t = x6[:, :, t]
            x7_t = x7[:, :, t]
            x8_t = x8[:, :, t]

            # -> Nächster Hidden State der Sequenz wird berechnet (mithilfe der Inputs und Gewichte)
            # QUESTION -> Änderung Input Sizes oben
            i_t = torch.sigmoid(Y_t @ self.W_i + h_t @ self.U_i + self.b_i)
            i_x1_t = torch.sigmoid(x1_t @ self.W_i_x1 + h_t @ self.U_i_x1 + self.b_i_x1)
            i_x2_t = torch.sigmoid(x2_t @ self.W_i_x2 + h_t @ self.U_i_x2 + self.b_i_x2)
            i_x3_t = torch.sigmoid(x2_t @ self.W_i_x3 + h_t @ self.U_i_x3 + self.b_i_x3)
            i_x4_t = torch.sigmoid(x2_t @ self.W_i_x4 + h_t @ self.U_i_x4 + self.b_i_x4)
            i_x5_t = torch.sigmoid(x2_t @ self.W_i_x5 + h_t @ self.U_i_x5 + self.b_i_x5)
            i_x6_t = torch.sigmoid(x2_t @ self.W_i_x6 + h_t @ self.U_i_x6 + self.b_i_x6)
            i_x7_t = torch.sigmoid(x2_t @ self.W_i_x7 + h_t @ self.U_i_x7 + self.b_i_x7)
            i_x8_t = torch.sigmoid(x2_t @ self.W_i_x8 + h_t @ self.U_i_x8 + self.b_i_x8)

            f_t = torch.sigmoid(Y_t @ self.W_f + h_t @ self.U_f + self.b_f)

            C_tilde_t = torch.tanh(Y_t @ self.W_c + h_t @ self.U_c + self.b_c)
            C_x1_tilde_t = torch.tanh(x1_t @ self.W_c_x1 + h_t @ self.U_c_x1 + self.b_c_x1)
            C_x2_tilde_t = torch.tanh(x2_t @ self.W_c_x2 + h_t @ self.U_c_x2 + self.b_c_x2)
            C_x3_tilde_t = torch.tanh(x3_t @ self.W_c_x3 + h_t @ self.U_c_x3 + self.b_c_x3)
            C_x4_tilde_t = torch.tanh(x4_t @ self.W_c_x4 + h_t @ self.U_c_x4 + self.b_c_x4)
            C_x5_tilde_t = torch.tanh(x5_t @ self.W_c_x5 + h_t @ self.U_c_x5 + self.b_c_x5)
            C_x6_tilde_t = torch.tanh(x6_t @ self.W_c_x6 + h_t @ self.U_c_x6 + self.b_c_x6)
            C_x7_tilde_t = torch.tanh(x7_t @ self.W_c_x7 + h_t @ self.U_c_x7 + self.b_c_x7)
            C_x8_tilde_t = torch.tanh(x8_t @ self.W_c_x8 + h_t @ self.U_c_x8 + self.b_c_x8)

            o_t = torch.sigmoid(Y_t @ self.W_o + h_t @ self.U_o + self.b_o)

            # Direkt vor Attention Layer
            l_t = C_tilde_t * i_t
            l_x1_t = C_x1_tilde_t * i_x1_t
            l_x2_t = C_x2_tilde_t * i_x2_t
            l_x3_t = C_x3_tilde_t * i_x3_t
            l_x4_t = C_x4_tilde_t * i_x4_t
            l_x5_t = C_x5_tilde_t * i_x5_t
            l_x6_t = C_x6_tilde_t * i_x6_t
            l_x7_t = C_x7_tilde_t * i_x7_t
            l_x8_t = C_x8_tilde_t * i_x8_t

            # Zwischenschritt / Attention Layer
            u_t = torch.tanh(l_t @ self.W_a * c_t + self.b_a)
            u_x1_t = torch.tanh(l_x1_t @ self.W_a * c_t + self.b_a)
            u_x2_t = torch.tanh(l_x2_t @ self.W_a * c_t + self.b_a)
            u_x3_t = torch.tanh(l_x3_t @ self.W_a * c_t + self.b_a)
            u_x4_t = torch.tanh(l_x4_t @ self.W_a * c_t + self.b_a)
            u_x5_t = torch.tanh(l_x5_t @ self.W_a * c_t + self.b_a)
            u_x6_t = torch.tanh(l_x6_t @ self.W_a * c_t + self.b_a)
            u_x7_t = torch.tanh(l_x7_t @ self.W_a * c_t + self.b_a)
            u_x8_t = torch.tanh(l_x8_t @ self.W_a * c_t + self.b_a)

            alpha_t = torch.softmax(torch.stack([u_t, u_x1_t, u_x2_t, u_x3_t, u_x4_t, u_x5_t, u_x6_t, u_x7_t, u_x8_t]), dim=0)

            # Nach Attention Layer
            L_t = alpha_t[0, :, :] * l_t + alpha_t[1, :, :] * l_x1_t + alpha_t[2, :, :] * l_x2_t + alpha_t[3, :, :] * l_x3_t + alpha_t[4, :, :] * l_x4_t + alpha_t[5, :, :] * l_x5_t + alpha_t[6, :, :] * l_x6_t + alpha_t[7, :, :] * l_x7_t + alpha_t[8, :, :] * l_x8_t

            c_t = f_t * c_t + L_t
            h_t = o_t * torch.tanh(c_t)

            # In Sequenz packen, damit wir die später noch haben -> um die zum nächsten Layer weiterzugeben
            # Unsqueeze -> Shape wird verändert
            hidden_seq.append(h_t.unsqueeze(0))

        # Auch Veränderung shapes -> um später für Hidden Sequenz richtigen Shape zu haben
        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        # Output und die Sequenz (ggf. für nächsten Layer) zurückgeben; Bei nächstem Layer Linear würde h_t reichen
        return h_t, hidden_seq


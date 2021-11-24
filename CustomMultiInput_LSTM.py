import torch
import torch.nn as nn
import math

"""
Describes the custom MultiInput LSTM cell
For visualisation see './Grafiken/CustomLSTM_Architecture.png'

Steps which happen here:

DEFINITION OF ARCHITECTURE OF CUSTOM MULTI INPUT LSTM CELL
- Definition & initialisation of weights and biases
- Implementation of 7 separate gates - one for each TaLib-class 
- Forward step: calculate the next hidden state based on the defined inputs and the weights
(in detail in './Grafiken/CustomLSTM_Architecture.png')
- Return the output and the hidden state for the next LSTM
"""
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

        # o_t
        self.W_o = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_o = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_o = nn.Parameter(torch.Tensor(hidden_sz))

        # a_t
        self.W_a = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_a = nn.Parameter(torch.Tensor(hidden_sz))

        self.init_weights()

    def init_weights(self):
        # Initialisation of the weights
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

    def forward(self, Y, x1, x2, x3, x4, x5, x6, x7):
        bs, _, seq_sz = Y.size()
        # -> output // seq_sz: 22; bs: 1; _: 5

        hidden_seq = []
        # init hidden State with 0: 'empty' state, theres no state in the first point in time
        # C_T init = Carrystate for 'long term observation'
        h_t, c_t = (
            torch.zeros(bs, self.hidden_size).to(Y.device),
            torch.zeros(bs, self.hidden_size).to(Y.device),
        )

        # Calculation of states for each point in time
        # We loop through the 22 points in time
        for t in range(seq_sz):

            # Open/Close/Volume ... at the different points in time
            Y_t = Y[:, :, t]  # Shape: (1, 5) (Open/Close/Volume/Min/Max)
            x1_t = x1[:, :, t]  # Shape: (1, 3) (Kennzahl1/Kennzahl2/Kennzahl3)
            x2_t = x2[:, :, t]  # ...
            x3_t = x3[:, :, t]
            x4_t = x4[:, :, t]
            x5_t = x5[:, :, t]
            x6_t = x6[:, :, t]
            x7_t = x7[:, :, t]

            # -> Calculate the next hidden state based on the inputs and the weights
            i_t = torch.sigmoid(Y_t @ self.W_i + h_t @ self.U_i + self.b_i)  # hier
            i_x1_t = torch.sigmoid(x1_t @ self.W_i_x1 + h_t @ self.U_i_x1 + self.b_i_x1)
            i_x2_t = torch.sigmoid(x2_t @ self.W_i_x2 + h_t @ self.U_i_x2 + self.b_i_x2)
            i_x3_t = torch.sigmoid(x2_t @ self.W_i_x3 + h_t @ self.U_i_x3 + self.b_i_x3)
            i_x4_t = torch.sigmoid(x2_t @ self.W_i_x4 + h_t @ self.U_i_x4 + self.b_i_x4)
            i_x5_t = torch.sigmoid(x2_t @ self.W_i_x5 + h_t @ self.U_i_x5 + self.b_i_x5)
            i_x6_t = torch.sigmoid(x2_t @ self.W_i_x6 + h_t @ self.U_i_x6 + self.b_i_x6)
            i_x7_t = torch.sigmoid(x2_t @ self.W_i_x7 + h_t @ self.U_i_x7 + self.b_i_x7)

            f_t = torch.sigmoid(Y_t @ self.W_f + h_t @ self.U_f + self.b_f)

            C_tilde_t = torch.tanh(Y_t @ self.W_c + h_t @ self.U_c + self.b_c)
            C_x1_tilde_t = torch.tanh(x1_t @ self.W_c_x1 + h_t @ self.U_c_x1 + self.b_c_x1)
            C_x2_tilde_t = torch.tanh(x2_t @ self.W_c_x2 + h_t @ self.U_c_x2 + self.b_c_x2)
            C_x3_tilde_t = torch.tanh(x3_t @ self.W_c_x3 + h_t @ self.U_c_x3 + self.b_c_x3)
            C_x4_tilde_t = torch.tanh(x4_t @ self.W_c_x4 + h_t @ self.U_c_x4 + self.b_c_x4)
            C_x5_tilde_t = torch.tanh(x5_t @ self.W_c_x5 + h_t @ self.U_c_x5 + self.b_c_x5)
            C_x6_tilde_t = torch.tanh(x6_t @ self.W_c_x6 + h_t @ self.U_c_x6 + self.b_c_x6)
            C_x7_tilde_t = torch.tanh(x7_t @ self.W_c_x7 + h_t @ self.U_c_x7 + self.b_c_x7)

            o_t = torch.sigmoid(Y_t @ self.W_o + h_t @ self.U_o + self.b_o)

            # Before the attention layer
            l_t = C_tilde_t * i_t
            l_x1_t = C_x1_tilde_t * i_x1_t
            l_x2_t = C_x2_tilde_t * i_x2_t
            l_x3_t = C_x3_tilde_t * i_x3_t
            l_x4_t = C_x4_tilde_t * i_x4_t
            l_x5_t = C_x5_tilde_t * i_x5_t
            l_x6_t = C_x6_tilde_t * i_x6_t
            l_x7_t = C_x7_tilde_t * i_x7_t

            # Intermediate step / Attention Layer
            u_t = torch.tanh(l_t @ self.W_a * c_t + self.b_a)
            u_x1_t = torch.tanh(l_x1_t @ self.W_a * c_t + self.b_a)
            u_x2_t = torch.tanh(l_x2_t @ self.W_a * c_t + self.b_a)
            u_x3_t = torch.tanh(l_x3_t @ self.W_a * c_t + self.b_a)
            u_x4_t = torch.tanh(l_x4_t @ self.W_a * c_t + self.b_a)
            u_x5_t = torch.tanh(l_x5_t @ self.W_a * c_t + self.b_a)
            u_x6_t = torch.tanh(l_x6_t @ self.W_a * c_t + self.b_a)
            u_x7_t = torch.tanh(l_x7_t @ self.W_a * c_t + self.b_a)

            alpha_t = torch.softmax(torch.stack([u_t, u_x1_t, u_x2_t, u_x3_t, u_x4_t, u_x5_t, u_x6_t, u_x7_t]),
                                    dim=0)

            # After Attention Layer
            L_t = alpha_t[0, :, :] * l_t + alpha_t[1, :, :] * l_x1_t + alpha_t[2, :, :] * l_x2_t + alpha_t[3, :,
                                                                                                   :] * l_x3_t + alpha_t[
                                                                                                                 4, :,
                                                                                                                 :] * l_x4_t + alpha_t[
                                                                                                                               5,
                                                                                                                               :,
                                                                                                                               :] * l_x5_t + alpha_t[
                                                                                                                                             6,
                                                                                                                                             :,
                                                                                                                                             :] * l_x6_t + alpha_t[
                                                                                                                                                           7,
                                                                                                                                                           :,
                                                                                                                                                           :] * l_x7_t

            c_t = f_t * c_t + L_t
            h_t = o_t * torch.tanh(c_t)

            # Save the hidden sequence to pass it to the next layer (see Entire_Model.py)
            hidden_seq.append(h_t.unsqueeze(0))

        # Adjust the shapes of the hidden sequence
        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        # Return the output and the hidden sequence, so we can use both
        return h_t, hidden_seq

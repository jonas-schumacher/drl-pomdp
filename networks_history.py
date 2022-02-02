import torch
import torch.nn as nn

"""
---------------------------- LSTM HISTORY NETWORK ----------------------------
"""


class HistoryNet(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, hps):
        super(HistoryNet, self).__init__()

        torch.manual_seed(hps['env']['SEED'])

        # Here: card played & player who played that card
        self.input_dim = input_dim
        # To be evaluated, how big this internal representation has to be
        self.hidden_dim = hidden_dim
        # Here: who played which card & who wasn't able to follow which suit
        self.output_dim = output_dim

        """
        Input for each node in LSTM layer: 
        1) previous output (h) of size (hidden_dim)
        2) previous hidden cell state (c) of size (hidden_dim)
        3) input (x) from sequence of size (input_dim)
        
        Output of each node LSTM layer:
        1) output (h) for sequence of size (hidden dim) >> this can be further processed by subsequent layers
        2) updated cell state (c) of size (hidden_dim)
        
        If we input a whole sequence of size (L, input_dim) we will receive an output sequence of size (L, hidden_dim)
        We can optionally set the (h,c) of the initial cell and we can receive the (h,c) of the last cell
        """

        self.preprocess = nn.Sequential(
            nn.Linear(in_features=input_dim,
                      out_features=2*input_dim),
            nn.ReLU(),
            nn.Linear(in_features=2*input_dim,
                      out_features=input_dim),
            nn.ReLU()
        )

        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_dim,
                            batch_first=True)

        """
        The output (h) from the LSTM layer of size (hidden_dim) will be transformed into a tensor of shape (output_dim)
        """
        self.fc = nn.Sequential(
            nn.Linear(in_features=hidden_dim,
                      out_features=output_dim),
            nn.ReLU(),
            nn.Linear(in_features=output_dim,
                      out_features=output_dim)
        )

    def forward(self, x, hc):
        """
        :param x: input sequence of shape (L, N, input_dim)
        :param hc: tuple (h_0,c_0) of initial output h_0 & hidden cell state c_0
        :return: sequence of output_cards of shape (L, N, output_dim)
        """
        # x = self.preprocess(x)
        # output shape: (L, N, hidden_dim)
        output_lstm, hidden_tuple = self.lstm(x, hc)
        # output shape: (L, N, output_dim)
        output_final = self.fc(output_lstm)
        return output_final, hidden_tuple

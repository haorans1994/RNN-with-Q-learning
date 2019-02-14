import torch
from torch import nn
from torch.autograd import Variable


LR = 0.02
TIME_STEP = 10



class RNN_stock(nn.Module):
    def __init__(self, INPUT_SIZE):
        super(RNN_stock, self).__init__()

        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
        )
        self.out = nn.Linear(64,1)

    def forward(self,x,h_state):

        r_out, h_state = self.rnn(x, h_state)

        outs = []  # save all predictions
        for time_step in range(r_out.size(1)):  # calculate output for each time step
            outs.append(self.out(r_out[:, time_step, :]))
        return torch.stack(outs, dim=1), h_state


        # print(type(x),type(h_state))




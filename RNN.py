
from torch import nn


INPUT_SIZE = 28

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(
            input_size = INPUT_SIZE,
            hidden_size = 64,
            num_layers = 1,
            batch_first = True,
        )
        self.out = nn.Linear(64,10)

    def forward(self,x):
        r_out,(h_n,h_c) = self.rnn(x,None)
        out = self.out(r_out[:,-1,:])
        return out

rnn = RNN()
print(rnn)

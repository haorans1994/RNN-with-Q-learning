import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import RNN
import numpy as np
import torch
from torch import autograd
from torch.autograd import Variable

NAME = '000001.csv'


def generate_data(data):
    stock = pd.read_csv(data)
    relative_data = stock.drop(columns=['date'],inplace=False)

    return relative_data, stock

new_data,original_data = generate_data(NAME)
Keys = new_data.keys()
train_data = new_data.values
train_data = train_data[495:]
original_data = original_data[495:]
EPOC = len(train_data)

INPUT_SIZE = len(Keys)
rnn = torch.load('Stock1_rnn.pkl')


h_state = None
for step in range(EPOC-RNN.TIME_STEP-1):
    print(step)
    start = step
    end = step + RNN.TIME_STEP
    # steps = np.linspace(start,end,RNN.TIME_STEP,dtype=int)
    x = train_data[start:end]
    x = x.reshape(1,RNN.TIME_STEP,13)
    x = torch.from_numpy(x)
    x = x.float()
    x = autograd.Variable(x)

    prediction, h_state = rnn(x,h_state)

    target = original_data['close'][495+end + 1]
    print(target)
    # print(target)
    print(prediction.data.numpy()[0][0][-1])


    # prediction = np.array(prediction,dtype=float)
    # prediciton = torch.from_numpy(prediction)
    # prediction = autograd.Variable(prediction)







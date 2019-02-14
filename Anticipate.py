import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import RNN
import numpy as np
import torch
from torch import autograd
from torch.autograd import Variable

NAME= '000001.csv'





def generate_data(data):
    stock = pd.read_csv(data)
    relative_data = stock.drop(columns=['date'],inplace=False)

    return relative_data, stock

new_data,original_data = generate_data(NAME)
Keys = new_data.keys()
train_data = new_data.values
EPOC = len(original_data)
State_number = len(Keys)
rnn = RNN.RNN_stock(State_number)
# print(rnn)

optimizer = RNN.torch.optim.Adam(rnn.parameters(),lr=RNN.LR)
loss_function = RNN.nn.MSELoss()



h_state = None
image_x = []
image_y = []
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

    target = original_data['close'][start+1:end+1]
    target = target.values
    target = target.reshape(1,RNN.TIME_STEP,1)
    target = torch.from_numpy(target)
    target = target.float()
    target = autograd.Variable(target)

    prediction, h_state = rnn(x,h_state)


    # prediction = np.array(prediction,dtype=float)
    # prediciton = torch.from_numpy(prediction)
    # prediction = autograd.Variable(prediction)

    loss = loss_function(prediction,target)
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()
    image_x.append(step)
    image_y.append(loss.data.numpy()[0])

plt.plot(image_x,image_y)
plt.xlabel('step')
plt.ylabel('loss value')
plt.title('training loss value')

plt.show()




    # print(input)




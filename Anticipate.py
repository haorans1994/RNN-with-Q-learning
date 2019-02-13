import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import RNN
import numpy as np

NAME= '000001.csv'





def generate_data(data):
    stock = pd.read_csv(data)
    relative_data = stock.drop(columns=['date'],inplace=False)

    return relative_data, stock

new_data,original_data = generate_data(NAME)
keys = new_data.keys()
train_data = new_data.values
EPOC = len(original_data)
State_number = len(keys)
rnn = RNN.RNN_stock(State_number)

optimizer = RNN.torch.optim.Adam(rnn.parameters(),lr=RNN.LR)
loss_function = RNN.nn.MSELoss()



h_state = None
for step in range(EPOC-1):
    # start = step
    # end = step + RNN.TIME_STEP
    # steps = np.linspace(start,end,RNN.TIME_STEP,dtype=int)
    input = train_data[step]
    target = original_data['close'][step+1]

    # prediction, h_state = rnn(input)
    # h_state = RNN.Variable(h_state.data)
    #
    # loss = loss_function(prediction,target)
    # optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()




    # print(input)




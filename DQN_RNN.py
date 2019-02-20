import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
import gym
import numpy as np


BATCH_SIZE = 32
LR = 0.01
EPSILON = 0.9
GAMMA= 0.9
TARGET_REPLACE_ITER = 100
MEMORY_CAPACITY = 2000

env = gym.make('CartPole-v0')
env = env.unwrapped

ALL_ACTIONS = env.action_space.n
OBSERVE_STATES = env.observation_space.shape[0]

ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape

def state_transform(batch_size,Time_step,State):
    State = np.array(State)
    State = State.reshape(batch_size, Time_step, len(State))
    State = torch.from_numpy(State)
    State = State.float()
    State = autograd.Variable(State)
    return State

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(
            input_size=OBSERVE_STATES,
            hidden_size=50,
            num_layers=1,
            batch_first=True,
        )
        self.out = nn.Linear(50,ALL_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)


    def forward(self,x):
        x ,(h_n, h_c)= self.rnn(x)
        x = F.relu(x)
        actions_value=self.out(x[:, -1, :])
        return actions_value


class DQN(object):
    def __init__(self):
        self.eval_net= RNN()
        self.target_net = RNN()

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, OBSERVE_STATES*2+2 ))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(),lr = LR)
        self.loss_func = nn.MSELoss()



    def choose_action(self,x):
        x = state_transform(1,1,x)
        if np.random.uniform() < EPSILON:
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value,1)[1].data.numpy()
            # print(action)
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        else:
            action = np.random.randint(0,ALL_ACTIONS)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action


    def store_transition(self,s,a,r,s_):
        transition = np.hstack((s,[a,r],s_))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter +=1


    def learn(self):
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        sample_index = np.random.choice(MEMORY_CAPACITY,BATCH_SIZE)

        b_memory = self.memory[sample_index, :]


        b_s = b_memory[:,:OBSERVE_STATES]
        print(b_s)


        # print(len(b_s))
        # exit()
        b_a = Variable(torch.LongTensor(b_memory[:, OBSERVE_STATES:OBSERVE_STATES+1].astype(int)))
        b_r = Variable(torch.FloatTensor(b_memory[:,OBSERVE_STATES+1:OBSERVE_STATES+2]))
        b_s_ = Variable(torch.FloatTensor(b_memory[:, -OBSERVE_STATES:]))

        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

dqn = DQN()
record = []
for i_episode in range(1000):
    s = env.reset()
    # print(s)
    ep_r=0
    old_r = 0
    while True:
        env.render()


        a = dqn.choose_action(s)
        # print(a)

        s_, r, done, info = env.step(a)

        old_r+= r

        x, x_dot, theta, theta_dot = s_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        r = r1 + r2




        dqn.store_transition(s, a, r, s_)
        ep_r+=r

        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()
            if done:
                record.append(old_r)
        if done:
            record.append(old_r)
            break
        s = s_
print(record)







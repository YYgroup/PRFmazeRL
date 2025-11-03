# -*- coding: utf-8 -*-
from field_env import Flow_Field
from utils import transform_raw_state, calculate_reward_from_raw_state
import math
import random
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14
#plt.rcParams['font.family'] = 'serif'
#plt.rcParams['font.serif'] = 'Times New Roman'
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

import numpy as np



# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

######################################################################
# Replay Memory
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
        self.transformed_memory = deque([], maxlen=capacity)
    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))
    def transformed_push(self, *args):
        """Save a transition"""
        self.transformed_memory.append(Transition(*args))
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def transformerd_sample(self, batch_size):
        return random.sample(self.transformed_memory, batch_size)
    def popleft(self):
        return self.memory.popleft()
    def __len__(self):
        return len(self.memory)
######################################################################
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        hidden = 128
        mhid = 128
        """
        self.QNet = nn.Sequential(
            nn.Linear(n_observations, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )
        """
        #"""
        # feature net
        self.feat_net = nn.Sequential(
            nn.Linear(n_observations, hidden).double(),
            #nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, mhid).double(),
            #nn.LayerNorm(mhid),
            nn.ReLU(),      
        )
        # A(s,a)
        self.adv_net = nn.Sequential(
            #nn.Linear(mhid, mhid).double(),
            #nn.LayerNorm(mhid),
            #nn.ReLU(),
            #nn.Linear(mhid, mhid).double(),
            #nn.LayerNorm(mhid),
            #nn.ReLU(),
            nn.Linear(mhid, n_actions).double(),
        )
        # V(s)
        self.val_net = nn.Sequential(
            #nn.Linear(mhid, mhid).double(),
            #nn.LayerNorm(mhid),
            #nn.ReLU(),
            #nn.Linear(mhid, mhid).double(),
            #nn.LayerNorm(mhid),
            #nn.ReLU(),
            nn.Linear(mhid, 1).double(),
        )
        #"""
    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        #values = self.QNet(x)
        #"""
        feat = self.feat_net(x) 
        adv = self.adv_net(feat)
        state_value = self.val_net(feat)
        values = state_value + adv - adv.mean(-1, keepdim=True)
        #"""
        return values
######################################################################

BUFFER_SIZE = 40000
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 100
TAU = 0.005
LR = 1e-4

env = Flow_Field()
# Get number of actions from env action space
n_actions = len(env.action_space)
# Get the number of state observations
state = transform_raw_state(env.initialize())
print("=================================initialize DQN network=======================================")
n_observations = len(state)
# default memory filename
default_load_in_filename = '../DQN_replay_memory.txt'
default_write_out_filename = '../DQN_replay_memory.txt'
default_pretrain_file = '../expert_pretrain.txt'
default_network = 'maze_DQN_policy_net.pth'

class DeepQNetwork:
    def __init__(self, n_observations=n_observations, n_actions=n_actions, lr=LR):
        #学习参数
        self.lr = lr
        self.gamma = GAMMA
        # [s, a, s_, r, d]
        self.memory = ReplayMemory(10000)
        # 总学习次数
        self.learn_step_counter = 0
        self.device = device
        self.policy_net = DQN(n_observations, n_actions).double()
        self.target_net = DQN(n_observations, n_actions).double()
        try:
            self.policy_net.load_state_dict(torch.load(default_network, map_location=self.device))
            print("load in network file", default_network)
        except:
            print("no network found")
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.policy_net=self.policy_net.to(self.device)
        self.target_net=self.target_net.to(self.device)
        self.optimizer = optim.AdamW(self.policy_net.parameters(), self.lr, amsgrad=True)
        self.steps_done=0
        self.memory = ReplayMemory(BUFFER_SIZE)
        self.eps_threshold = 0.5
        self.loss_history = []
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=1e5)

    def update_network(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        self.target_net.load_state_dict(target_net_state_dict)

    def cal_eps_threshold(self):
        self.eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * self.steps_done / EPS_DECAY)
    def select_action(self, state, det=False):
        sample = random.random()
        self.cal_eps_threshold()
        self.steps_done += 1
        if sample > self.eps_threshold or det==True:
            with torch.no_grad():
                return self.policy_net(state.to(self.device).double()).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[random.randint(0, n_actions - 1)]], device=self.device, dtype=torch.long)
    
    def is_finished_pretrain(self):
        bs = len(self.memory)
        transitions = self.memory.sample(bs)
        batch = Transition(*zip(*transitions))
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action).squeeze()
        predict = self.policy_net(state_batch).max(1).indices
        count=0
        for i in range(len(predict)):
            if predict[i].item()==action_batch[i].item():
                count+=1
        return predict.equal(action_batch), count/len(predict)

    def pretrain_model(self, expert_set=default_pretrain_file):
        if self.learn_step_counter==0:
            self.load_in(load_in_filename=expert_set,pick_odd=True)
        if len(self.memory) < BATCH_SIZE:
            return
        MAX_ITER = 1e4
        counter = 0
        done = False
        while counter<MAX_ITER and not done:
            counter+=1
            transitions = self.memory.sample(BATCH_SIZE)
            batch = Transition(*zip(*transitions))
            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action)
            predict = self.policy_net(state_batch)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(predict, action_batch.squeeze())
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 100)
            self.optimizer.step()
            if counter%1e3 == 0:
                done, acc = self.is_finished_pretrain()
            self.learn_step_counter+=1
            torch.save(self.policy_net.state_dict(), default_network)
            if self.learn_step_counter%1000==0:
                print('pretrain learning step = ', self.learn_step_counter, 'loss = ', loss.item(), 'acc = ', acc, '\n')

    def optimize_model(self, both_run_and_train=False):
        if both_run_and_train == True:  # run and train at the same time, transformed state stored in RL.memory.transformed_memory
            if len(self.memory.transformed_memory) < BATCH_SIZE:
                return
            transitions = self.memory.transformerd_sample(BATCH_SIZE)
        else:                           # offline train, transformed state stored in RL.memory.memory
            if len(self.memory) < BATCH_SIZE:
                return
            transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)
        done_batch = torch.cat(batch.done)
    
        # 各量形状:
        # state_batch/next_state_batch: [BATCH_SIZE, n_observations]
        # action_batch: [BATCH_SIZE, 1]
        # reward_batch: [BATCH_SIZE]
        # done_batch: [BATCH_SIZE]
        #print(state_batch.shape,action_batch.shape,reward_batch.shape,next_state_batch.shape,done_batch.shape)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        with torch.no_grad():
            next_state_values = (1 - done_batch) * self.target_net(next_state_batch).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch
        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        self.scheduler.step()
        self.learn_step_counter+=1
        torch.save(self.policy_net.state_dict(), default_network)
        if self.learn_step_counter%1000==0:
            print('learning step = ', self.learn_step_counter, 'loss = ', loss.item(), '\n')
        with open('DQN_loss.txt', 'a') as f:
            f.write(str(loss.item()) + '\n')
        self.loss_history.append(loss.item())
        #tempLost = np.array(self.loss_history)
        #if len(self.loss_history)>2000 and np.mean(tempLost[-500:]) > 1.1*np.mean(tempLost[-1500:]):
        #    for param_group in self.optimizer.param_groups:  
        #        param_group['lr'] = param_group['lr']*0.999

    def plot_loss(self):
        with open('DQN_loss.txt', 'r') as f:
            loss = np.loadtxt(f.name, unpack=True).T
        temp = np.zeros_like(loss)
        for i in range(len(temp)):
            temp[i] = loss[max(0, i-40):min(len(temp), i+41)].mean()
        plt.ticklabel_format(style='sci',axis='x')
        plt.yscale('log', base=10)
        plt.plot(np.arange(len(loss)), temp, label='loss')
        plt.xlabel('training steps')
        plt.ylabel('loss')
        plt.legend(loc="best")
        plt.grid(True)
        plt.show()

    def write_out(self, n=0, write_out_filename = default_write_out_filename):
        print("write out memory...\n")
        for _ in range(n):
            self.memory.popleft()
        with open(write_out_filename, 'a') as f:
            while len(self.memory):
                # temp = (state, action, next_state, reward, done)
                # state/next_state: [1, n_observation]
                # action: [1,1]
                # reward: [1]
                # done: [1]
                temp = self.memory.popleft()
                state = temp[0].squeeze()
                f.write('state = [')
                for i, element in enumerate(state):
                    if i == 0:
                        f.write(str(element.item()) + ' ')
                    else:
                        f.write(',' + str(element.item()))
                f.write(']\n')
                f.write('action = ' + str(temp[1].item()) + '\n')
                f.write('reward = ' + str(temp[3].item()) + '\n')
                next_state = temp[2].squeeze()
                f.write('next_state = [')
                for i, element in enumerate(next_state):
                    if i == 0:
                        f.write(str(element.item()) + ' ')
                    else:
                        f.write(',' + str(element.item()))
                f.write(']\n')
                f.write('done = ' + str(temp[4].item()) + '\n')   
    
    def load_in(self, load_in_filename = default_load_in_filename, transform_flag=False, shift=np.array([0,0]),pick_odd=False):
        try:
            f = open(load_in_filename, 'r') 
        except:
            return
        lines = f.readlines()
        f.close()  
        print("open data file", load_in_filename)
        # temp = (state, action, next_state, reward, done)
        # state/next_state: [1, n_observation]
        # action: [1,1]
        # reward: [1]
        # done: [1]
        for i,line in enumerate(lines):
            if 'next_state' in line:
                start = line.index('[')
                end = line.index(']')
                deal = line[start + 1:end].split(',')
                for i, s in enumerate(deal):
                    deal[i] = float(s)
                next_state = torch.tensor(deal, dtype=torch.float64, device=self.device).unsqueeze(0)
            elif 'state' in line:
                start = line.index('[')
                end = line.index(']')
                deal = line[start + 1:end].split(',')
                for i, s in enumerate(deal):
                    deal[i] = float(s)
                state = torch.tensor(deal, dtype=torch.float64, device=self.device).unsqueeze(0)
            elif 'action' in line:
                action = int(line.split()[-1])
            elif 'reward' in line:
                reward = float(line.split()[-1])
            elif 'done' in line:
                done = int(line.split()[-1])
                new_state = state
                new_next_state = next_state
                if pick_odd==True and transform_flag==False and len(state.squeeze())==n_observations and state.squeeze()[-1].item()%2==1:
                    continue

                #reward = -new_next_state.squeeze()[8].item()

                self.memory.push(new_state, torch.tensor([[action]], device=self.device), new_next_state, torch.tensor([reward],dtype=torch.float64, device=self.device), torch.tensor([done], device=self.device))
                if transform_flag==True and len(state.squeeze())>n_observations:
                    new_state  = transform_raw_state(state.squeeze(), shift, shift).unsqueeze(0).to(self.device)
                    new_next_state  = transform_raw_state(next_state.squeeze(), shift, shift).unsqueeze(0).to(self.device)
                    reward = calculate_reward_from_raw_state(next_state.squeeze(), shift, shift)
                    done = 0
                    if new_next_state.squeeze()[8].item()<0.03:
                        done = 1
                    if pick_odd==True and new_state.squeeze()[-1].item()%2==1:
                        continue
                    self.memory.transformed_push(new_state, torch.tensor([[action]], device=self.device), new_next_state, torch.tensor([reward],dtype=torch.float64, device=self.device), torch.tensor([done], device=self.device))
        n = len(self.memory)
        if len(state.squeeze())==n_observations:
            print('load in transformed experience tuple:',n,'\n')
        else:
            print('load in transformerd experience tuple:',n)
            print('load in raw experience tuple:',n,'\n')
   
    def offline_train(self, training_steps, data_filename=default_load_in_filename, transform_flag=True):
        self.load_in(load_in_filename=data_filename, transform_flag=transform_flag)
        if len(self.memory.transformed_memory)>0:
            flag = True
        else:
            flag=False
        for _ in range(training_steps):
            self.optimize_model(both_run_and_train=flag)
        self.plot_loss()

    def evaluate_network(self, state_filename):
        with open(state_filename,"r") as f:
            lines = f.readlines()
        states = []
        actions = []
        for line in lines:
            temp = line.split(',')
            if len(line)==2 and line!="\n":
                actions.append(int(line[0]))
            if len(temp)>1:
                state = []
                for item in temp:
                    if len(item)>2:
                        state.append(float(item)) 
                states.append(torch.tensor(state, dtype=torch.float64))
        actual_actions = []
        count = 0
        for i,state in enumerate(states):
            action_num = self.select_action(state.unsqueeze(0),det=True) 
            count+=1
            actual_actions.append(action_num)
        #print(len(states),len(actions),len(d),len(actual_actions))
        count2 = 0
        with open(state_filename[:-4]+"_evaluate_result.txt","w") as f:
            gg_string = "  "
            for i in range(count):
                if actions[i]!=actual_actions[i]:
                    gg_string = "gg"
                    count2+=1
                f.write(str(actions[i])+"  "+str(actual_actions[i])+"    "+gg_string+'\n')
                gg_string = "  "
            f.write('\nfailed/total states: '+str(count2)+'/'+str(count) + '  '+str(count2/count)+'\n')
            life = 0
            for i in range(count):
                if actions[i]==actual_actions[i] or i%4==1 or i%4==3:
                    life+=0.5
                else:
                    break
            f.write('successful life: ' + str(life))
        print('\nfailed/total states: '+str(count2)+'/'+str(count)+ '  '+str(count2/count))
        print('successful life: ' + str(life))
        return
######################################################################
def plot_reward(reward_his,show_result=False, train_flag=True):
    plt.figure(1)
    total_reward_t = torch.tensor(reward_his, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        if train_flag==True:
            plt.title('Training...')
        else:
            plt.title('Evaluating...')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.plot(total_reward_t.numpy())
    # Take 100 episode averages and plot them too
    slide = 100
    means = torch.zeros(total_reward_t.shape)
    for i, item in enumerate(means):
        means[i]= total_reward_t[max(0,i-slide):i].mean()
    plt.plot(means.numpy())
    plt.pause(0.001) 
  
######################################################################
# Training loop
def run_simulation(RL, env, num_episodes=200, train=True, write_out=False):
    plt.ion()
    total_reward_history = []
    for i_episode in range(num_episodes):
        # Initialize the environment and get its state
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float64, device=RL.device).unsqueeze(0)
        total_reward = 0
        done = False

        while not done:
            action = RL.select_action(state, det=not train)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            total_reward += reward
            reward = torch.tensor([reward], device=RL.device)
            done = terminated or truncated
            new_done = 1 if done==True else 0
            next_state = torch.tensor(observation, dtype=torch.float64, device=RL.device).unsqueeze(0)
            # Store the transition in memory
            # 输入的形状:
            # state/next_state: [1, n_observation]
            # action: [1,1]
            # reward: [1]
            # done: [1]
            #print(state.shape, action.shape, next_state.shape, reward.shape)
            RL.memory.push(state, action, next_state, reward, torch.tensor([new_done], device=RL.device))
            # Move to the next state
            state = next_state
            if train==True:
                # Perform one step of the optimization (on the policy network)
                RL.optimize_model()
                # Soft update of the target network's weights
                RL.update_network()
        
        total_reward_history.append(total_reward)
        plot_reward(total_reward_history, show_result=False, train_flag=train)
    print('Complete')
    plot_reward(total_reward_history,show_result=True, train_flag=train)
    plt.ioff()
    plt.show()
    if train==True:
        RL.plot_loss()
    if write_out==True:
        RL.write_out(0, write_out_filename = 'DQN_replay_memory.txt')
######################################################################



######################################################################




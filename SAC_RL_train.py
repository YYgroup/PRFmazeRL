import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
from collections import deque
#from SAC_field_env import Flow_Field
from SAC_utils import read_in_raw_data_and_generate_sequence_list_and_write_out, load_in_sequence, perform_ope, calculate_reward_from_raw_state, transform_raw_state
import math
from cmath import pi
import time

n_features = 15
BATCH_SIZE = 256
BATCH_SIZE_PRE = 128

default_sequence_file = ''
raw_state_file = 'raw_base.txt'
train_dataset_file = 'train_dataset.txt'
default_pi_network = 'maze_SAC_pi_model.pth'
default_q_model1_network = 'maze_SAC_q_origin_model1.pth'
default_q_model2_network = 'maze_SAC_q_origin_model2.pth'
default_pi_loss_file = 'SAC_pi_loss.txt'
default_q1_loss_file = 'SAC_q1_loss.txt'
default_q2_loss_file = 'SAC_q2_loss.txt'
default_pretrain_loss_file = 'SAC_loss_pretrain.txt'
default_pretrain_network = 'SAC_policy_net_pretrain'
default_pretrain_dataset = 'expert_pretrain.txt'
# 状态缓冲区大小
BUFFER_SIZE = 60000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)

#env = Flow_Field()
n_actions = 4
n_observations = 15

init_alpha = 0.1
gamma = 0.99
tau = 0.002
BATCH_SIZE = 256
lr_pi = 1e-4
lr_q1 = 1e-4
lr_q2 = lr_q1

class replayBuffer:
    def __init__(self, buffer_size: int):
        #self.buffer_size = buffer_size
        self.buffer = deque([], maxlen=buffer_size)
    def push(self, item):
        self.buffer.append(item)
    def sample(self, batch_size):
        items = random.sample(self.buffer, batch_size)
        states   = [i[0] for i in items]
        actions  = [i[1] for i in items]
        rewards  = [i[2] for i in items]
        n_states = [i[3] for i in items]
        dones    = [i[4] for i in items]
        return states, actions, rewards, n_states, dones
    def popleft(self):
        return self.buffer.popleft()
    def __len__(self):
        return len(self.buffer)


# Policy net (pi_theta)
class PolicyNet(nn.Module):
    def __init__(self, input_dim = n_observations, output_dim = n_actions, hidden_dim=128):
        super().__init__()
        self.pNet = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(normalized_shape=hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(normalized_shape=hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
    def forward(self, s):
        outs = self.pNet(s)
        return outs

class categorical:
    def __init__(self, s, pi_model):
        logits = pi_model(s)
        self._prob = F.softmax(logits, dim=-1)
        self._logp = torch.log(self._prob)
    # probability (sum is 1.0) : P
    def prob(self):
        return self._prob
    # log probability : log P()
    def logp(self):
        return self._logp

class QNet(nn.Module):
    def __init__(self, input_dim = n_observations, output_dim = n_actions, hidden_dim=128):
        super().__init__()
        self.qNet = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(normalized_shape=hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(normalized_shape=hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
    def forward(self, s):
        outs = self.qNet(s)
        return outs

class SACNetwork:
    def __init__(self, gamma=gamma, device=device, lr_pi=lr_pi, lr_q1=lr_q1, lr_q2=lr_q2, tau=tau, bs=BATCH_SIZE, n_actions = n_actions):
        self.device = device
        self.pi_model = PolicyNet().to(self.device)
        self.q_origin_model1 = QNet().to(self.device)  # Q_phi1
        self.q_origin_model2 = QNet().to(self.device)  # Q_phi2
        self.q_target_model1 = QNet().to(self.device)  # Q_phi1'
        self.q_target_model2 = QNet().to(self.device)  # Q_phi2'
        try:
            self.pi_model.load_state_dict(torch.load(default_pi_network))
            self.q_origin_model1.load_state_dict(torch.load(default_q_model1_network)) 
            self.q_target_model1.load_state_dict(torch.load(default_q_model1_network))
            self.q_origin_model2.load_state_dict(torch.load(default_q_model2_network))
            self.q_target_model2.load_state_dict(torch.load(default_q_model2_network))
        except:
            pass
        _ = self.q_target_model1.requires_grad_(False)  # target model doen't need grad
        _ = self.q_target_model2.requires_grad_(False)  # target model doen't need grad
        self.gamma = gamma
        self.opt_pi = torch.optim.AdamW(self.pi_model.parameters(), lr=lr_pi)
        self.opt_q1 = torch.optim.AdamW(self.q_origin_model1.parameters(), lr=lr_q1)
        self.opt_q2 = torch.optim.AdamW(self.q_origin_model2.parameters(), lr=lr_q2)
        self.memory = replayBuffer(BUFFER_SIZE)
        self.bs = bs
        self.tau = tau
        self.n_actions = n_actions
        self.learn_step_counter = 0

        self.target_alpha = np.log10(n_actions)*0.98
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device) 
        self.alpha = self.log_alpha.exp()*init_alpha
        self.opt_alpha = torch.optim.AdamW([self.log_alpha], lr=1e-4)


    def update_target(self):
        for var, var_target in zip(self.q_origin_model1.parameters(), self.q_target_model1.parameters()):
            var_target.data = self.tau * var.data + (1.0 - self.tau) * var_target.data
        for var, var_target in zip(self.q_origin_model2.parameters(), self.q_target_model2.parameters()):
            var_target.data = self.tau * var.data + (1.0 - self.tau) * var_target.data
   

    # 根据当前observation给出action,这里给出一个序号,然后再对应action_space中的动作
    def select_action(self, s, det=False):
        with torch.no_grad():
            #   --> size : (1, n_observations)
            s_batch = np.expand_dims(s, axis=0)
            s_batch = torch.tensor(s_batch, dtype=torch.float64).to(device)
            # Get logits from state
            #   --> size : (1, n_actions)
            logits = self.pi_model(s_batch)
            #print(logits)
            #   --> size : (n_actions)
            logits = logits.squeeze(dim=0)
            # From logits to probabilities
            probs = F.softmax(logits, dim=-1)
            # Pick up action's sample
            #   --> size : (1)
            a = torch.multinomial(probs, num_samples=1)
            #   --> size : ()
            a = a.squeeze(dim=0)
            if det==True:
                a = torch.argmax(probs).item()
            # Return
            return a.tolist()

    def write_out_buffer(self,n=0):
        print("write out buffer...\n")
        for _ in range(n):
            self.memory.buffer.popleft()
        with open(train_dataset_file, 'a') as f:
            count = 0
            while count < len(self.memory.buffer):
                temp = self.memory.buffer[count]
                count+=1
                state = temp[0]
                f.write('state = [')
                for i, element in enumerate(state):
                    if i == 0:
                        f.write(str(element.item()) + ' ')
                    else:
                        f.write(',' + str(element.item()))
                f.write(']\n')
                f.write('action = ' + str(temp[1]) + '\n')
                f.write('reward = ' + str(temp[2]) + '\n')
                next_state = temp[3]
                f.write('next_state = [')
                for i, element in enumerate(next_state):
                    if i == 0:
                        f.write(str(element.item()) + ' ')
                    else:
                        f.write(',' + str(element.item()))
                f.write(']\n')
                f.write('done = ' + str(temp[4]) + '\n')


    def load_pretrain_buffer(self, buffer_name=default_pretrain_dataset):
        try:
            f = open(buffer_name, 'r') 
        except:
            return
        lines = f.readlines()
        f.close()  
        # 顺序为 state, action, reward, next_state, done
        for i,line in enumerate(lines):
            if 'next_state' in line:
                start = line.index('[')
                end = line.index(']')
                deal = line[start + 1:end].split(',')
                for i, s in enumerate(deal):
                    deal[i] = float(s)
                next_state = torch.tensor(deal, dtype=torch.float64)
            elif 'state' in line:
                start = line.index('[')
                end = line.index(']')
                deal = line[start + 1:end].split(',')
                for i, s in enumerate(deal):
                    deal[i] = float(s)
                state = torch.tensor(deal, dtype=torch.float64)
            elif 'action' in line:
                action = int(line.split()[-1])
            elif 'reward' in line:
                reward = float(line.split()[-1])
            elif 'done' in line:
                done = int(line.split()[-1])
                new_state = state
                new_next_state = next_state
                self.memory.push([new_state, torch.tensor([action]), reward, new_next_state, done])
        n = len(self.memory.buffer)
        print('load in experience tuple:',n,'\n')
        

    def load_buffer(self, buffer_name=raw_state_file):
        try:
            f = open(buffer_name, 'r') 
        except:
            return
        lines = f.readlines()
        f.close()  
        # 顺序为 state, action, reward, next_state, done
        for i,line in enumerate(lines):
            if 'next_state' in line:
                start = line.index('[')
                end = line.index(']')
                deal = line[start + 1:end].split(',')
                for i, s in enumerate(deal):
                    deal[i] = float(s)
                next_state = torch.tensor(deal, dtype=torch.float64)
            elif 'state' in line:
                start = line.index('[')
                end = line.index(']')
                deal = line[start + 1:end].split(',')
                for i, s in enumerate(deal):
                    deal[i] = float(s)
                state = torch.tensor(deal, dtype=torch.float64)
            elif 'action' in line:
                action = int(line.split()[-1])
            elif 'reward' in line:
                reward = float(line.split()[-1])
            elif 'done' in line:
                done = int(line.split()[-1])
                new_state = state
                new_next_state = next_state
                self.memory.push([new_state, action, reward, new_next_state, done])
        n = len(self.memory.buffer)
        print('load in experience tuple:',n,'\n')
        if len(state) > n_features: # raw dataset
            print('postprocess raw data into normal format...\n')
            new_queue = deque()
            target_shift = np.array([random.random()-0.5, random.random()-0.5])
            #target_shift = np.array([0,0])
            for i, item in enumerate(self.memory.buffer):
                # item's format (state, action, reward, next_state, done)
                #target_shift = np.array([random.random()-0.5, random.random()-0.5]) #+ np.array([-0.4, 0])  # -0.8,0   -0.4,0
                prev_target_shift = target_shift
                new_state = transform_raw_state(item[0], prev_target_shift, target_shift)
                new_next_state = transform_raw_state(item[3], prev_target_shift, target_shift)
                reward = calculate_reward_from_raw_state(item[3], prev_target_shift, target_shift)
                d = new_next_state[8].item()
                done = 1 if d<0.03 else 0
                new_queue.append((new_state, item[1], reward, new_next_state, done))
            self.memory.buffer.clear()
            self.memory.buffer = deque(new_queue)

    def pretrain(self):
        #一次抓batch_size个记忆库中的样本，不够就先不学
        if len(self.memory) < BATCH_SIZE_PRE:
            return
        state, action, reward, next_state, done = self.memory.sample(bs=BATCH_SIZE_PRE)
        state = state.to(self.device)
        action = action.to(self.device).squeeze()
        reward = reward.to(self.device)  
        next_state = next_state.to(self.device)
        done = done.to(self.device)
        predict = self.policy_net(state)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(predict, action)
        self.optimizer_p.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
        self.optimizer_p.step()
        # 写入loss
        self.loss_his_pre.append(loss.item())
        with open(default_pretrain_loss_file, 'a') as f:
            f.write(str(self.loss_his_pre[-1]) + '\n')
        # 保存模型
        torch.save(self.policy_net.state_dict(), default_pretrain_network)
        #torch.save(self.value_net.state_dict(), 'SAC_value_net_pretrain')

        if self.learn_step_counter%200==0:
            print('learning step = ', self.learn_step_counter, 'loss = ', np.log10(loss.item()), '\n')
        # increasing epsilon
        self.epsilon = self.epsilon + self.e_greedy_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        # 记录policy网络已学习的步数
        self.learn_step_counter += 1
        self.value_net_target.update(self.value_net)
    
    def optimize_theta(self, states):
        # Convert to tensor
        #for state in states:
        #    print(state.device)
        states = torch.stack(states, dim=0).to(self.device)
        #print(states.shape)
        #states = torch.tensor(states, dtype=torch.float64).to(device)
        # Disable grad in q_origin_model1 before computation
        for p in self.q_origin_model1.parameters():
            p.requires_grad = False
        # Optimize
        self.opt_pi.zero_grad()
        dist = categorical(states, self.pi_model)
        q_value = self.q_origin_model1(states)
        term1 = dist.prob()
        term2 = q_value - self.alpha.detach() * dist.logp()
        expectation = term1.unsqueeze(dim=1) @ term2.unsqueeze(dim=2)

        #print("term1 shape",term1.shape,"term2 shape",term2.shape,"expectation shape",expectation.shape)

        expectation = expectation.squeeze(dim=1)
        (-expectation).sum().backward()
        torch.nn.utils.clip_grad_norm_(self.pi_model.parameters(), 1)
        self.opt_pi.step()
        # Enable grad again
        for p in self.q_origin_model1.parameters():
            p.requires_grad = True
        with open(default_pi_loss_file, 'a') as f:
            f.write(str((-expectation).mean().item()) + '\n')
        if self.learn_step_counter%1000==0:
            print("learn step =", self.learn_step_counter, "loss_pi =", (-expectation).mean().item())
    def optimize_phi(self, states, actions, rewards, next_states, dones):
        # Convert to tensor
        states = torch.stack(states, dim=0).to(device)
        actions = torch.tensor(actions, dtype=torch.int64).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float64).to(device)
        rewards = rewards.unsqueeze(dim=1)
        next_states = torch.stack(next_states, dim=0).to(device)
        dones = torch.tensor(dones, dtype=torch.float64).to(device)
        dones = dones.unsqueeze(dim=1)
        # print(states.shape, actions.shape, rewards.shape, next_states.shape, dones.shape)
        # shape:
        # states/next_states:  [BATCH_SIZE, n_observations]
        #            actions:  [BATCH_SIZE]
        #            rewards:  [BATCH_SIZE, 1]
        #              dones:  [BATCH_SIZE, 1] 

        # Compute r + gamma * (1 - d) (min Q(s_next,a_next') + alpha * H(P))
        with torch.no_grad():
            # min Q(s_next,a_next')
            q1_tgt_next = self.q_target_model1(next_states)
            q2_tgt_next = self.q_target_model2(next_states)
            dist_next = categorical(next_states, self.pi_model)
            q1_target = q1_tgt_next.unsqueeze(dim=1) @ dist_next.prob().unsqueeze(dim=2)
            q1_target = q1_target.squeeze(dim=1)
            q2_target = q2_tgt_next.unsqueeze(dim=1) @ dist_next.prob().unsqueeze(dim=2)
            q2_target = q2_target.squeeze(dim=1)
            q_target_min = torch.minimum(q1_target, q2_target)
            # alpha * H(P)
            h = dist_next.prob().unsqueeze(dim=1) @ dist_next.logp().unsqueeze(dim=2)
            h = h.squeeze(dim=1)
            h = -self.alpha.detach() * h
            # total
            term2 = rewards + self.gamma * (1.0 - dones) * (q_target_min + h)
        # Optimize critic loss for Q-network1
        self.opt_q1.zero_grad()
        one_hot_actions = F.one_hot(actions, num_classes=self.n_actions).double()
        q_value1 = self.q_origin_model1(states)
        term1 = q_value1.unsqueeze(dim=1) @ one_hot_actions.unsqueeze(dim=2)
        term1 = term1.squeeze(dim=1)
        loss_q1 = F.mse_loss(
            term1,
            term2,
            reduction="none")
        loss_q1.sum().backward()
        torch.nn.utils.clip_grad_norm_(self.q_origin_model1.parameters(), 1e-1)
        self.opt_q1.step()
        # Optimize critic loss for Q-network2
        self.opt_q2.zero_grad()
        one_hot_actions = F.one_hot(actions, num_classes=self.n_actions).double()
        q_value2 = self.q_origin_model2(states)
        term1 = q_value2.unsqueeze(dim=1) @ one_hot_actions.unsqueeze(dim=2)
        term1 = term1.squeeze(dim=1)
        loss_q2 = F.mse_loss(
            term1,
            term2,
            reduction="none")
        loss_q2.sum().backward()
        torch.nn.utils.clip_grad_norm_(self.q_origin_model2.parameters(), 1e-1)
        self.opt_q2.step()
        with open(default_q1_loss_file, 'a') as f:
            f.write(str(loss_q1.mean().item()) + '\n')
        with open(default_q2_loss_file, 'a') as f:
            f.write(str(loss_q1.mean().item()) + '\n')
        if self.learn_step_counter%1000==0:
            print("learn step =", self.learn_step_counter, "loss_q =", (loss_q1.mean().item()+loss_q2.mean().item())/2, ", alpha =",self.alpha.item())
    def optimize_alpha(self,states):
        states = torch.stack(states, dim=0).to(self.device)
        with torch.no_grad():
            dist = categorical(states, self.pi_model)
        alpha_loss = torch.mean(torch.sum(-dist.prob() * (self.alpha*(dist.logp()+self.target_alpha)), dim=1)) 
        # print('alpha loss: ',alpha_loss)
        self.opt_alpha.zero_grad()
        alpha_loss.backward()
        self.opt_alpha.step()
        self.alpha = self.log_alpha.exp()*init_alpha
        #print("alpha loss =", alpha_loss.item())
    def optimize_model(self, update_alpha=True):
        if len(self.memory) < self.bs:
            return
        states, actions, rewards, n_states, dones = self.memory.sample(self.bs)
        if update_alpha==True and abs(self.alpha.item()/self.target_alpha - 1)>0.3:
            self.optimize_alpha(states)
        self.optimize_theta(states)
        self.optimize_phi(states, actions, rewards, n_states, dones)
        self.update_target()
        
        self.learn_step_counter+=1

    def plot_loss(self):
        with open(default_pi_loss_file, 'r') as f:
            loss_pi = np.loadtxt(f.name, unpack=True).T
        smooth_loss_pi = np.zeros_like(loss_pi)
        for i in range(len(smooth_loss_pi)):
            smooth_loss_pi[i] = loss_pi[max(0, i-40):min(len(smooth_loss_pi), i+41)].mean()
        with open(default_q1_loss_file, 'r') as f:
            loss_q1 = np.loadtxt(f.name, unpack=True).T
        smooth_loss_q1 = np.zeros_like(loss_q1)
        for i in range(len(smooth_loss_q1)):
            smooth_loss_q1[i] = loss_q1[max(0, i-40):min(len(smooth_loss_q1), i+41)].mean()
        with open(default_q2_loss_file, 'r') as f:
            loss_q2 = np.loadtxt(f.name, unpack=True).T
        smooth_loss_q2 = np.zeros_like(loss_q2)
        for i in range(len(smooth_loss_q2)):
            smooth_loss_q2[i] = loss_q2[max(0, i-40):min(len(smooth_loss_q2), i+41)].mean()
        fig, ax1 = plt.subplots(figsize=(10, 7.5))
        ax1.set_xlabel('training steps')
        ax1.set_ylabel('policy loss')
        ax1.plot(smooth_loss_pi, label='policy loss')#linestyle='-', linewidth=1, color='blue'
        ax2 = ax1.twinx()
        ax2.set_ylabel('critic loss')  
        ax2.plot(smooth_loss_q1, linestyle='-', color='tab:orange', label='critic loss 1')
        ax2.plot(smooth_loss_q2, linestyle='--', color='tab:red', label='critic loss 2')
        ax2.set_yscale('log', base=10) 
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='best')
        plt.title('loss of actor and critic network')
        plt.tight_layout()
        plt.show()

    def plot_pretrain_loss(self):
        import matplotlib.pyplot as plt
        from matplotlib import rcParams
        rcParams['font.size'] = 14
        with open(default_pretrain_loss_file, 'r') as f:
            loss = np.loadtxt(f.name, unpack=True).T
        #loss = np.log10(loss)
        temp = np.zeros_like(loss)
        for i in range(len(temp)):
            temp[i] = loss[max(0, i-40):min(len(temp), i+41)].mean()
        plt.ticklabel_format(style='sci',axis='x')
        plt.plot(np.arange(len(loss)), temp, label='lossp')
        plt.xlabel('training steps')
        plt.ylabel('loss_log')
        plt.legend(loc="best")
        plt.grid(True)
        plt.show()

    def testPerformance(self, state_filename):
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
            action_num = self.choose_action(state, det=True)
            print(self.policy_net.forward(state.to(self.device)))
            count+=1
            actual_actions.append(action_num)
        count2 = 0
        with open("SAC_"+state_filename[:-4]+"_evaluate_result.txt","w") as f:
            gg_string = "  "
            for i in range(count):
                if actions[i]!=actual_actions[i]:
                    gg_string = "gg"
                    count2+=1
                f.write(str(actions[i])+"  "+str(actual_actions[i])+"  "+gg_string+'\n')
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
       

def calculate_policy_value_based_on_sequence(sequence_file=default_sequence_file):
    return


def start_train(RL, steps):
    RL.load_buffer(buffer_name='transformer_train_set.txt')
    for _ in range(steps):
        RL.learn()
    RL.plot_loss()

def start_pretrain(RL, steps):
    RL.load_buffer()
    for _ in range(steps):
        RL.pretrain()
    RL.plot_pretrain_loss()

def start_mix_train(RL, steps):
    RL.load_buffer()
    RL2 = SACNetwork()
    RL2.load_pretrain_buffer()
    for i in range(steps):
        RL.learn()
        if i!=0 and i%50000==0:
            torch.save(RL.policy_net.state_dict(), 'policy_net_phase_pretrain_'+str(i//50000))
            torch.save(RL.target_net.state_dict(), 'target_net_phase_pretrain_'+str(i//50000))
        if i%10000==0 and i!=0:
            for _ in range(1000):
                RL2.pretrain()
    RL.plot_loss()


def train_multiple_SAC_agent_and_cal_ope(times:int):
    import os
    state_action_reward_chain = load_in_sequence()
    pi_model = PolicyNet()
    ope_value = []
    global default_pi_network
    global default_q_model1_network
    global default_q_model2_network
    try:
        os.remove(default_pi_network)
        os.remove(default_q_model1_network)
        os.remove(default_q_model2_network)
    except:
        pass
    #"""
    RL = SACNetwork()
    RL.load_buffer(buffer_name=train_dataset_file)
    #"""
    for i in range(times):
        # train 10w steps and save as xxx_i and cal ope and save
        t1 = time.time()
        #"""
        RL.pi_model = PolicyNet().to(RL.device)
        RL.q_origin_model1 = QNet().to(RL.device)  # Q_phi1
        RL.q_origin_model2 = QNet().to(RL.device)  # Q_phi2
        RL.q_target_model1 = QNet().to(RL.device)  # Q_phi1'
        RL.q_target_model2 = QNet().to(RL.device)  # Q_phi2'
        RL.learn_step_counter = 0
        _ = RL.q_target_model1.requires_grad_(False)  # target model doen't need grad
        _ = RL.q_target_model2.requires_grad_(False)  # target model doen't need grad
        RL.opt_pi = torch.optim.AdamW(RL.pi_model.parameters(), lr=lr_pi)
        RL.opt_q1 = torch.optim.AdamW(RL.q_origin_model1.parameters(), lr=lr_q1)
        RL.opt_q2 = torch.optim.AdamW(RL.q_origin_model2.parameters(), lr=lr_q2)
        RL.log_alpha = torch.zeros(1, requires_grad=True, device=RL.device) 
        RL.alpha = RL.log_alpha.exp()*init_alpha
        RL.opt_alpha = torch.optim.AdamW([RL.log_alpha], lr=1e-4)
        #"""
        #RL = SACNetwork()
        #RL.load_buffer(buffer_name=train_dataset_file)

        default_pi_network = 'maze_SAC_pi_model_'+str(i)+'.pth'
        default_q_model1_network = 'maze_SAC_q_origin_model1_'+str(i)+'.pth'
        default_q_model2_network = 'maze_SAC_q_origin_model2_'+str(i)+'.pth'
        
        for _ in range(50001):
            RL.optimize_model()
        torch.save(RL.pi_model.state_dict(), default_pi_network)
        torch.save(RL.q_origin_model1.state_dict(), default_q_model1_network)
        torch.save(RL.q_origin_model2.state_dict(), default_q_model2_network)
        pi_model.load_state_dict(torch.load(default_pi_network))
        policy_value = perform_ope(pi_model,state_action_reward_chain)
        q1 = np.percentile(policy_value, 25)
        q3 = np.percentile(policy_value, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        filtered_data = policy_value[(policy_value > lower_bound) & (policy_value < upper_bound)]
        ope_value.append(np.mean(filtered_data))

        with open("ope_value.txt","a") as f:
            f.write(str(np.mean(filtered_data)) + '\n')

        default_pi_network = 'maze_SAC_pi_model.pth'
        default_q_model1_network = 'maze_SAC_q_origin_model1.pth'
        default_q_model2_network = 'maze_SAC_q_origin_model2.pth'
        t2 = time.time()
        print("training time for",i,"th agent:", t2-t1)
    
    with open("ope_value.txt","w") as f:
        for ope in ope_value:
            f.write(str(ope) + '\n')

    plt.figure(figsize=(10, 6))
    plt.bar([str(i) for i in range(len(ope_value))], ope_value, color='skyblue')
    plt.title('OPE for multiple SAC agents')
    plt.xlabel('Number')
    plt.ylabel('OPE value')
    plt.show()




train_multiple_SAC_agent_and_cal_ope(2)




"""

RL = SACNetwork()

RL.load_buffer(buffer_name=train_dataset_file)
for _ in range(10000):
    RL.optimize_model(update_alpha=False)
RL.plot_loss()


state_action_reward_chain = load_in_sequence()
pi_model = PolicyNet()
pi_model.load_state_dict(torch.load(default_pi_network))
policy_value = perform_ope(pi_model,state_action_reward_chain)
#print(policy_value.mean())
q1 = np.percentile(policy_value, 25)
q3 = np.percentile(policy_value, 75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
filtered_data = policy_value[(policy_value > lower_bound) & (policy_value < upper_bound)]
print(np.mean(filtered_data))
#print(np.sort(policy_value))
"""
















"""
FOR TRAIN SET
RL.load_buffer()
RL.write_out_buffer()

FOR EXPERT SET
RL.load_buffer(buffer_name="raw_expert.txt")
train_dataset_file = "expert_dataset.txt"
RL.write_out_buffer()


FOR OPE SEQ GEN
shift=np.random.uniform(low=-0.4,high=0.4,size=2)
#shift = np.array([0,0])
read_in_raw_data_and_generate_sequence_list_and_write_out(shift=shift)





FOR SAC TRAIN
RL.load_buffer(buffer_name=train_dataset_file)
for _ in range(500000):
    RL.optimize_model(update_alpha=False)
RL.plot_loss()


FOR OPE EVAL
state_action_reward_chain = load_in_sequence()
pi_model = PolicyNet()
pi_model.load_state_dict(torch.load(default_pi_network))
policy_value = perform_ope(pi_model,state_action_reward_chain)
print(policy_value.mean())



filename = "state_log_right_phase_standard_refine.txt"
RL.testPerformance(filename)
filename = "state_log_phase_righttrace_refine.txt"
RL.testPerformance(filename)
filename = "state_log_phase_doubleCircle_refine.txt"
RL.testPerformance(filename)
"""
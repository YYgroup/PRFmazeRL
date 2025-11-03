from cmath import pi
from lib2to3 import refactor
from field_env import Flow_Field, filename
from RL_brain import DeepQNetwork
from DQN_utils import calculate_reward_from_raw_state, transform_raw_state
import torch
import numpy as np
import random
from collections import deque
import os
import matplotlib.pyplot as plt


# 只要最原始的状态,不要transform
# raw_state = (前一刻的三点坐标，目标点坐标，质心坐标，当前的三点坐标，目标点坐标，质心坐标,前一刻的phase,当前的phase)
# 经过transform后，变成12维的状态

MAX_EPISODE = 40
scale_factor = torch.tensor([10,10,20,20,10,10, 30,30, 1, 1/pi, 6/pi, 50,25, 1e4, 1/4])
scale_factor = torch.tensor([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
def postProcess():
    with open(os.path.join('viz_IB2d', 'dumps.visit'), "r") as f:
        lines=f.readlines()
        first = lines[0]
        second = lines[1]
        step_size=int(lines[1][11:second.index('/')])-int(lines[0][11:first.index('/')])
        temp = lines[-1]
        final_step=int(lines[-1][11:temp.index('/')])
    pointer=0
    with open(os.path.join('viz_IB2d', 'dumps.visit'), "w+") as f:
        while pointer<=final_step:
            numstr=str(pointer)
            while len(numstr)<5:
                numstr='0'+numstr
            f.write("visit_dump."+numstr+"/summary.samrai\n")
            pointer+=step_size
    with open(os.path.join('viz_IB2d', 'lag_data.visit'), "r") as f:
        lines=f.readlines()
        step_size=int(lines[1][15:21])-int(lines[0][15:21])
        final_step=int(lines[-1][15:21])
    pointer=0
    with open(os.path.join('viz_IB2d', 'lag_data.visit'), "w+") as f:
        while pointer<=final_step:
            numstr=str(pointer)
            while len(numstr)<6:
                numstr='0'+numstr
            f.write("lag_data.cycle_"+numstr+"/lag_data.cycle_"+numstr+".summary.silo\n")
            pointer+=step_size


def calculate_sequence_obs(single_state, n, current_step, RL):
    pointer = current_step
    res = []
    res.append(single_state)
    while len(RL.memory.buffer)>0 and pointer>=0 and len(res)<n and 1-pointer+current_step<=len(RL.memory.buffer) :
        temp = (RL.memory.buffer[-1+pointer-current_step])[0]
        res.append(transform_raw_state(temp))
        pointer-=1
    while len(res)<n:
        temp = res[-1].clone()
        temp[-1] = (temp[-1].item() - 1 + 4)%4
        res.append(temp)
    return torch.stack(res)

####################################################################################
# Policy net (pi_theta)
import torch.nn as nn
from torch.nn import functional as F
class PolicyNet(nn.Module):
    def __init__(self, input_dim = 15, output_dim = 4, hidden_dim=128):
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
    def prob(self):
        return self._prob
    def logp(self):
        return self._logp
def select_action(pi_model, s, det=False):
    with torch.no_grad():
        s_batch = np.expand_dims(s, axis=0)
        s_batch = torch.tensor(s_batch, dtype=torch.float64)
        logits = pi_model(s_batch)
        logits = logits.squeeze(dim=0)
        probs = F.softmax(logits, dim=-1)
        a = torch.multinomial(probs, num_samples=1)
        a = a.squeeze(dim=0).item()
        if det==True:
            a = torch.argmax(probs).item()
        return a
####################################################################################




def start_swim(RL, times:int):
    action_his = open('action_history.txt', 'w+')
    state_log = open("state_log.txt","w+")
    action_his.close()
    state_log.close()
    pi_model = PolicyNet().double()
    pi_model.load_state_dict(torch.load('maze_SAC_pi_model.pth', map_location = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))
    with open('total_reward.txt', 'a') as f:
        for episode in range(times):
            # 初始环境
            env = Flow_Field()
            #=============================随机给定一个目标点========================
            #init_target = np.array([2.2, 2.6]) # maze
            #init_target = np.array([1.2, 2.0])
            #init_target = np.array([1.6, 2.4]) # five star
            #init_target = np.array([1.2,1.2])
            #init_target = np.array([2.0,1.2])
            #init_target = np.array([1.6,2.0]) #double circle
            #init_target = np.array([1.6, 2.4]) # maze

            init_target = np.array([2.75, 1.5]) # standard tunnel
            init_target = np.array([0.5,0.9]) # simple real world cave with target
            init_target = np.array([2.2, 2.6]) # fancy outter flow
            init_target = np.array([1.6, 2.6]) # single obstacle test

            #init_target[0] = 0.8 + 0.15 * random.randint(-4,4)
            #init_target[1] = 0.8 + 0.075 * random.randint(-4,4)
            #=====================================================================
            # 初始物体状态和环境
            #env.initialize_test(init_target)
            autoPilot = True
            observation = torch.tensor(env.initialize(init_target, autoPilot=autoPilot), dtype=torch.float64)
            
            done = 0
            step = 0  # 记录步数
            #RL.steps_done = step
            total_reward = 0
            last_action = -1

            stuck_count = 0
            #=====================================================================

            while not done:
                # all the state/observation use the raw form, only transformed it into 12d vector when choosing action
                # RL choose action based on observation
                # 输入的形状:
                # state/next_state: [1, n_observation]
                # action: [1,1]
                # reward: [1]
                # done: [1]
                print('==============current_time = ', env.currentTime, 'observation shape= ', observation.shape, '==========\n')
                print('\n++++++++++++++++++++++++++ 当前目标点 = ',env.initTarget,'+++++++++++++++++++++++++\n')
                print('\n++++++++++++++++++++++++++ 实时目标点 = ',env.targetPoint,'+++++++++++++++++++++++++\n')
                noise_factor = torch.tensor([0.9+0.2*random.random(), 0.9+0.2*random.random(), 0.9+0.2*random.random(), 0.9+0.2*random.random(), 0.9+0.2*random.random(), 0.9+0.2*random.random(), 0.9+0.2*random.random(), 0.9+0.2*random.random(), 0.9+0.2*random.random(), 0.9+0.2*random.random(), 0.9+0.2*random.random(), 0.9+0.2*random.random(), 0.9+0.2*random.random(), 0.9+0.2*random.random(), 1])
                #actual_obs = (transform_raw_state(observation)*noise_factor).unsqueeze(0)
                actual_obs = (transform_raw_state(observation)).unsqueeze(0)
                #action_num = RL.select_action(actual_obs, not train_flag)
                action_num = select_action(pi_model, actual_obs)

                robs = (actual_obs.squeeze())/scale_factor
                diameter = ((robs[0].item()-robs[4].item())**2+(robs[1].item()-robs[5].item())**2)**0.5
                if diameter>0.15:
                    if step%4==0:
                        action_num=0
                    else:
                        action_num=3
                if diameter<0.03:
                    if step%4==0:
                        action_num=3
                    else:
                        action_num=0
                #if step<40:
                #    action_num = 0
                #tempact = [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 1,3,0,0, 0,0,0,0, 0,0,0,0]
                tempact = [ 0,0,2,3, 3,3,1,3, 0,0,0,0, 2,3,1,3, 0,0,1,3,  #fancy障碍物避障
                           0,0,0,0, 2,3,1,3, 0,0,0,0, 0,0,0,0, 2,3,3,3, 
                           0,0,3,3, 3,3,3,3, 0,0,3,3, 3,3,1,3, 2,3,3,3,
                           0,0,0,0, 0,0,0,0, 1,3,3,3, 1,3,2,3, 1,3,2,3,
                           1,3,2,3, 3,3,3,3, 1,3,0,0, 0,0,0,0, 1,3,1,3,
                           1,3,1,3, 0,0,1,3, 1,3,0,0, 2,3,1,3, 1,3,1,3]
                           #2,3,1,3, 0,0,1,3, 0,0,0,0, 0,0,3,3, 1,3,3,3,
                           #0,0,0,0, 1,3,1,3, 2,3,1,3, 2,3,1,3, 2,3,1,3,
                           #2,3,1,3, 2,3,1,3, 1,3,3,3, 1,3,1,3, 0,0,0,0,
                           #2,3,3,3, 3,3,3,3, 3,3,3,3, 3,3,3,3, 0,0,3,3,
                           #3,3,0,0, 0,0,3,3, 0,0,0,0, 1,3,3,3, 1,3,3,3,
                           #0,0,0,0, 2,3,3,3, 0,0,1,3, 2,3,3,3, 3,3,0,0,
                           #3,3,3,3, 2,3,3,3, 0,0,0,0, 0,0,3,3, 3,3,3,3,
                           #0,0,0,0, 1,3,2,3, 3,3,3,3, 0,0,0,0, 1,3,3,3,
                           #3,3,0,0, 0,0,3,3, 2,3,2,3, 0,0,3,3, 3,3,0,0]
                """
                tempact = [1,3,2,3, 1,3,0,0, 0,0,3,3, 3,3,3,3, 2,3,3,3,  # S turn
                           3,3,3,3, 1,3,3,3, 3,3,1,3, 2,3,3,3, 0,0,0,0,
                           0,0,3,3, 3,3,0,0, 1,3,3,3, 0,0,3,3, 3,3,3,3,
                           2,3,0,0, 1,3,1,3, 0,0,1,3, 1,3,1,3, 0,0,0,0, 
                           2,3,1,3, 2,3,1,3, 2,3,3,3, 1,3,0,0, 2,3,3,3, 
                           1,3,1,3, 2,3,3,3, 1,3,3,3, 3,3,3,3, 1,3,0,0, 
                           2,3,3,3, 0,0,2,3, 2,3,2,3, 1,3,2,3, 0,0,0,0]
                """
                
                #if step<len(tempact):
                #    action_num = tempact[step]

                #action_num = 3 # test gravity
                # handle stuck
                nLag = env.Lagpoints.shape[0]
                hh = 1.5*1.6/512
                stuck_flag = False
                for ilag in range(nLag):
                    for point in env.knownWallPoints:
                        if np.linalg.norm(env.Lagpoints[ilag,:]-point)<hh:
                            stuck_count+=1
                            stuck_flag = True
                            break
                    if stuck_flag:
                        break
                if stuck_count>=10:
                    stuck_count = 0
                    action_num = 3
                    stuck_flag = False


                action_num = 0 # wall effect
                env.initTarget = np.array([2.2, 2.6])



                action_reg = 1
                if action_reg == 1:
                    # action regulation
                    temp = step % 4
                    if temp == 1 or temp == 3:
                        if last_action == 0:
                            action_num = 0
                        else:
                            action_num = 3
                last_action = action_num

                action = env.action_space[action_num]

                print('========action = ', action, '==========\n')
                try:
                    action_num = action_num.item()
                except:
                    pass

                action_his = open('action_history.txt', 'a')
                state_log = open("state_log.txt","a")
                action_his.write(str(action_num) +'\n')
                for jj in range(15):
                    state_log.write(str(actual_obs.squeeze()[jj].item()) + ', ')
                state_log.write('\n'+str(action_num)+'\n\n')
                action_his.close()
                state_log.close()
                # RL take action and get next observation and reward
                #print(action)
                observation_, reward, done = env.step(action)
                print(env.pre_targetPoint, env.targetPoint)
                
                total_reward += reward
                
                # 输入的形状:
                # state/next_state: [1, n_observation]
                # action: [1,1]
                # reward: [1]
                # done: [1]
                s = observation.unsqueeze(0).to(RL.device)
                s_ = observation_.unsqueeze(0).to(RL.device)
                a = torch.tensor([[action_num]], device = RL.device)
                r = torch.tensor([reward], dtype=torch.float64, device=RL.device)
                RL.memory.push(s, a, s_, r, torch.tensor([done], device=RL.device))
                print(observation.shape)

                # swap observation
                observation = observation_
                # train
                

                #####################################################################
                # moving target
                move_flag = 1
                if move_flag==1 and step>0:# and step%4 in [1,3]:
                    print("step = ", step)
                    env.pre_targetPoint[0] = env.targetPoint[0].copy()
                    env.pre_targetPoint[1] = env.targetPoint[1].copy()
                    #env.SenseSurroundingAreaAndChangeCourse()
                    env.SenseSurroundingAreaAndChangeCourse_new()
                # 右双圆环
                elif move_flag == 4:
                    t = step*0.5
                    fff= 1/90 #1/90 #1/120
                    TTT=1/fff
                    radius = 0.3 #0.25
                    env.pre_targetPoint[0] = env.targetPoint[0].copy()
                    env.pre_targetPoint[1] = env.targetPoint[1].copy()
                    mag = 0 #0.03
                    if t<TTT:
                        env.targetPoint[0] = 1.6 + radius - radius*np.cos(2*pi*fff*t) + mag*random.random()
                        env.targetPoint[1] = 2 + radius*np.sin(2*pi*fff*t) + mag*random.random()
                    elif t<2*TTT:
                        env.targetPoint[0] = 1.6 - radius + radius*np.cos(2*pi*fff*(t-TTT)) + mag*random.random()
                        env.targetPoint[1] = 2 + radius*np.sin(2*pi*fff*(t-TTT)) + mag*random.random()
                    if t>2*TTT and abs(observation[8].item()) < 0.02:
                        done = 1
                elif move_flag == 2: # five star traj
                    t = step*0.5
                    TTT=360
                    R=0.8
                    def rotate(point,alpha):
                        x=np.cos(alpha)*point[0]-np.sin(alpha)*point[1]
                        y=np.cos(alpha)*point[1]+np.sin(alpha)*point[0]
                        return [x,y]
                    dis = R*np.sin(0.1*pi)/np.sin(0.3*pi)
                    vertex = [[0,R], [dis*np.cos(0.3*pi), dis*np.sin(0.3*pi)]]
                    temp1 = [0,R]
                    temp2 = [dis*np.cos(0.3*pi), dis*np.sin(0.3*pi)]
                    for _ in range(4):
                        temp1 = rotate(temp1,-2*pi/5)
                        temp2 = rotate(temp2,-2*pi/5)
                        vertex.append(temp1)
                        vertex.append(temp2)
                    vertex.append([0,R])
                    num = int(t//(TTT/10))
                    t_s = (t - num*TTT/10)/(TTT/10)
                    env.pre_targetPoint[0] = env.targetPoint[0].copy()
                    env.pre_targetPoint[1] = env.targetPoint[1].copy()
                    env.targetPoint[0] = vertex[num][0]*(1-t_s) + vertex[num+1][0]*t_s + 1.6
                    env.targetPoint[1] = vertex[num][1]*(1-t_s) + vertex[num+1][1]*t_s + 1.6
                elif move_flag == 3:  # right trace
                    t = step*0.5
                    fff = 1/240 #1/120 #1/240 #1/120  #1/180  #1/240  #240 180
                    center_x = 1.6
                    center_y = 2.0  #0.7
                    theta = np.arctan(abs(center_y-env.initTarget[1])/abs(center_x-env.initTarget[0]))
                    radius = ((env.initTarget[1]-center_y)**2 + (env.initTarget[0]-center_x)**2)**0.5
                    env.pre_targetPoint[0] = env.targetPoint[0].copy()
                    env.pre_targetPoint[1] = env.targetPoint[1].copy()
                    env.targetPoint[0] = center_x + radius * np.cos(theta + 2*pi*fff*t)
                    env.targetPoint[1] = center_y + radius * np.sin(theta + 2*pi*fff*t)
                elif move_flag == 5: # five star complex traj
                    t = step*0.5
                    TTT=360
                    R=0.8
                    def rotate(point,alpha):
                        x=np.cos(alpha)*point[0]-np.sin(alpha)*point[1]
                        y=np.cos(alpha)*point[1]+np.sin(alpha)*point[0]
                        return [x,y]
                    dis = R*np.sin(0.1*pi)/np.sin(0.3*pi)
                    vertex = [[0,R], [dis*np.cos(0.3*pi), dis*np.sin(0.3*pi)]]
                    temp1 = [0,R]
                    temp2 = [dis*np.cos(0.3*pi), dis*np.sin(0.3*pi)]
                    for _ in range(4):
                        temp1 = rotate(temp1,-2*pi/5)
                        temp2 = rotate(temp2,-2*pi/5)
                        vertex.append(temp1)
                        vertex.append(temp2)
                    vertex.append([0,R])
                    complex_vertex = [vertex[i] for i in [0,4,8,2,6,0]]
                    num = int(t//(TTT/5))
                    t_s = (t - num*TTT/5)/(TTT/5)
                    env.pre_targetPoint[0] = env.targetPoint[0].copy()
                    env.pre_targetPoint[1] = env.targetPoint[1].copy()
                    env.targetPoint[0] = complex_vertex[num][0]*(1-t_s) + complex_vertex[num+1][0]*t_s + 1.6
                    env.targetPoint[1] = complex_vertex[num][1]*(1-t_s) + complex_vertex[num+1][1]*t_s + 1.6
                #####################################################################
                step += 1
                print('episode = ', episode, ' curren_time = ', env.currentTime, 'current total_reward = ', total_reward, '\n')
                print('\n=============================================================================================\n')
                print('\n==================================当前质心坐标:',env.massCenter,'==================================\n')
                print('\n==================================当前目标坐标:',env.targetPoint,'==================================\n')
                print('\n==================================当前observation:',observation.shape,'==================================\n')
                print('\n==================================当前与目标距离:',np.linalg.norm(env.massCenter - env.targetPoint),'==================================\n')
                print('\n=============================================================================================\n')
                
            print('=========================Episode ', episode, ' Total Reward = ', total_reward, '================================\n')
            f.write(str(total_reward) + '\n')
            print("当前质心坐标",env.massCenter, "目标点", env.targetPoint, "当前与目标距离",np.linalg.norm(env.massCenter - env.targetPoint),'\n')
            env.plot_traj()
            #action_his.write('\n\n')
            #state_log.write('\n\n')
            #plot_total_reward()
    
    # 训练结束
    print('train finished!\n')
    postProcess()
    #action_his.close()
    #state_log.close()


def start_mix_train(steps):
    RL=DeepQNetwork(lr=1e-2)
    RL.load_in(load_in_filename='../new_phase_base_refine_train_newReward.txt', transform_flag=False)
    RL2 = DeepQNetwork(lr=1e-4)
    for i in range(steps):
        RL.optimize_model()
        if i!=0 and i%1e3==0 and np.mean(np.array(RL.loss_history[-1000:]))<0.1:
            RL2.policy_net.load_state_dict(torch.load('DQN_policy_net'))
            RL2.target_net.load_state_dict(RL2.policy_net.state_dict())
            RL2.pretrain_model(expert_set='../expert_LeftAndRightFix_righttrace_pretrain.txt')
            RL.policy_net.load_state_dict(torch.load('DQN_policy_net'))
            RL.target_net.load_state_dict(RL.policy_net.state_dict())
    RL2.pretrain_model(expert_set='../expert_LeftAndRightFix_righttrace_pretrain.txt')
    RL.plot_loss()

def load_in_raw_and_transform_with_shifted_target(shifts):
    for shift in shifts:
        RL = DeepQNetwork()
        RL.load_in(transform_flag=True, shift=shift)
        RL.write_out()

def plot_total_reward(show_result=False):
    plt.figure(1)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    total_reward = np.loadtxt("total_reward.txt")
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(total_reward)
    #plt.show()
    



if __name__ == "__main__":
    RL = DeepQNetwork()
    n = len(RL.memory)
    start_swim(RL,1)

    
            
  

    
    
    

    
    
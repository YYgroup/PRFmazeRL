import torch
import numpy as np

def calculate_reward_from_raw_state(state, shift_prev=np.array([0.0, 0.0]), shift_curr=np.array([0.0, 0.0])):
    # 需要输入raw的next_state
    try:
        statelist = np.array(state.cpu())
    except:
        statelist = state
    current_mass = np.array([statelist[22], statelist[23]]) #上一刻的质心坐标
    prev_mass = np.array([statelist[8], statelist[9]])
    curr_target = np.array([statelist[20], statelist[21]]) + shift_curr #上一刻的目标点坐标
    prev_target = np.array([statelist[6], statelist[7]]) + shift_prev
    current_first =  np.array([statelist[16], statelist[17]]) - current_mass
    prev_first = np.array([statelist[2], statelist[3]]) - prev_mass
    s = current_first
    t = curr_target - current_mass
    theta = np.arccos((s[0]*t[0]+s[1]*t[1])/(np.linalg.norm(s)+1e-5)/(np.linalg.norm(t)+1e-5)) * np.sign(t[0]*s[1]-t[1]*s[0])
    distance = np.linalg.norm(curr_target - current_mass)
    vel = current_mass - prev_mass
    target_vel = curr_target - prev_target
    vel = vel - target_vel
    reward = -min(theta**2, 3) + max(0, min(3, 100*(vel[0]*t[0] + vel[1]*t[1])/np.linalg.norm(t))) - distance
    #print("theta penalty",-min(theta**2, 3), "v_parallel",(vel[0]*t[0] + vel[1]*t[1])/np.linalg.norm(t), "distance", -distance)
    return reward

def transform_raw_state(state, shift_prev=np.array([0.0, 0.0]), shift_curr=np.array([0.0, 0.0])):
    # 输入的state为前一个时刻的[6点坐标+target+masscenter+力+力矩+相位]+当前时刻的[6点坐标+target+masscenter+力+力矩+相位]:14+14=28
    try:
        statelist = np.array(state.cpu())
    except:
        statelist = state
    current_mass = np.array([statelist[22], statelist[23]]) 
    prev_mass = np.array([statelist[8], statelist[9]])
    curr_target = np.array([statelist[20], statelist[21]]) + shift_curr
    prev_target = np.array([statelist[6], statelist[7]]) + shift_prev
    # 中间点坐标
    current_first =  np.array([statelist[16], statelist[17]]) - current_mass
    prev_first = np.array([statelist[2], statelist[3]]) - prev_mass
    # theta
    s = current_first # 质心指向中间点,即对称轴方向
    t = curr_target - current_mass # 质心指向目标
    theta = np.arccos((s[0]*t[0]+s[1]*t[1])/(np.linalg.norm(s)+1e-5)/(np.linalg.norm(t)+1e-5)) * np.sign(t[0]*s[1]-t[1]*s[0])
    # omega
    t = prev_first
    s = current_first
    omega = np.arccos((s[0]*t[0]+s[1]*t[1])/np.linalg.norm(s)/np.linalg.norm(t)) * np.sign(t[0]*s[1]-t[1]*s[0])
    # 质心相对目标速度
    vel = current_mass - prev_mass
    vel_target = curr_target - prev_target
    vel = vel - vel_target
    # 左中右点相对质心的坐标
    left = np.array([statelist[14], statelist[15]]) - current_mass
    first = np.array([statelist[16], statelist[17]]) - current_mass
    right = np.array([statelist[18], statelist[19]]) - current_mass
    # 最终状态
    # 加入mass相对初始位置的位移+目标相对初始位置的位移？
    temp = [left[0], left[1], first[0], first[1], right[0], right[1], vel[0], vel[1], np.linalg.norm(current_mass - curr_target), theta, omega, statelist[-4], statelist[-3], statelist[-2], statelist[-1]]
    #if left[0] > right[0]:
    #    temp = [-left[0], -left[1], -first[0], -first[1], -right[0], -right[1], -vel[0], -vel[1], np.linalg.norm(current_mass- curr_target), theta, omega, statelist[-1]]
    return torch.tensor(temp, dtype = torch.float64)


raw_dataset_filename = 'raw_base.txt'
output_dataset_filename = 'sequence_for_ope.txt'
from collections import deque
def read_in_raw_data_and_generate_sequence_list_and_write_out(filename=raw_dataset_filename, output_file=output_dataset_filename, shift=np.array([0,0])):
        # we are loading raw states
        try:
            f = open(filename, 'r') 
        except:
            return
        lines = f.readlines()
        f.close()  
        # 顺序为 state, action, reward, next_state, done
        raw_queue = deque()
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
                raw_queue.append([new_state, action, reward, new_next_state, done])
        n = len(raw_queue)
        print('load in raw experience tuple:',n,'\n')
        # sequence list: store each episode's trajectory
        sequence_list = []
        single_squence = []
        while raw_queue:
            temp = raw_queue.popleft()
            if len(single_squence)==0:
                single_squence.append(temp)
            else:
                last_element_of_current_sequence = single_squence[-1]
                if last_element_of_current_sequence[3].equal(temp[0]):
                    single_squence.append(temp)
                else:
                    sequence_list.append(single_squence)
                    single_squence = [temp]
       
        #sequence_list.clear()
        print('load in', len(sequence_list), 'sequences\n')
        # for each sequence, we split them into a full transformed one hot phase 15d-state chain and action chain
        # state_action_chain =  [ [[s0,s1,...,sn],[a0,a1,...,an-1]], ... ]
        state_action_chain = []
        for i,single_squence in enumerate(sequence_list):
            state_chain = []
            action_chain = []
            reward_chain = []
            pre_shift = shift #+ np.array([-0.4,0]) + np.random.uniform(low=-0.4,high=0.4,size=2)
            curr_shift = pre_shift
            for dqn_tuple in single_squence:
                state_chain.append(transform_raw_state(dqn_tuple[0], shift_prev=pre_shift, shift_curr=curr_shift))
                action_chain.append(dqn_tuple[1])
                reward_chain.append(calculate_reward_from_raw_state(dqn_tuple[3], shift_prev=pre_shift, shift_curr=curr_shift))
            state_action_chain.append([state_chain, action_chain, reward_chain])
        print('output dataset:',output_file,'...\n')
        # write out sequence
        with open(output_file,"a") as f:
            for seq in state_action_chain:
                state_chain = seq[0]
                action_chain = seq[1]
                reward_chain = seq[2]
                f.write('state chain = [')
                for state in state_chain:
                    for i, element in enumerate(state):
                        f.write(str(element.item()) + ' ')     
                f.write(']\n')
                f.write('action chain = [')
                for action in action_chain:
                    f.write(str(action) + ' ')  
                f.write(']\n')
                f.write('reward chain = [')
                for reward in reward_chain:
                    f.write(str(reward) + ' ')  
                f.write(']\n')

n_observations = 15 
def load_in_sequence(filename=output_dataset_filename, n_observations = n_observations):
    with open(filename, "r") as f:
        lines = f.readlines()
    state_action_chain = []
    for line in lines:
        if 'state' in line:
            start = line.index('[')
            end = line.index(']')
            deal = line[start+1:end-1].split(' ')
            for i, s in enumerate(deal):
                deal[i] = float(s)
            #print(deal)
            num = len(deal)//n_observations
            #print(len(deal), num)
            state_chain = []
            for i in range(num):
                state_chain.append(torch.tensor(deal[i*n_observations:i*n_observations+n_observations], dtype=torch.float64)) 
        elif 'action' in line:
            start = line.index('[')
            end = line.index(']')
            action_chain = line[start+1:end-1].split(' ')
            for i, s in enumerate(action_chain):
                action_chain[i] = int(s)
        elif 'reward' in line:
            start = line.index('[')
            end = line.index(']')
            reward_chain = line[start+1:end-1].split(' ')
            for i, s in enumerate(reward_chain):
                reward_chain[i] = float(s)
            state_action_chain.append([state_chain, action_chain, reward_chain])
    print('load in dataset file:', filename, 'with', len(state_action_chain), 'sequence\n')
    return state_action_chain

def perform_ope(pi_model, state_action_chain, GAMMA=0.99):
    import torch.nn as nn
    from torch.nn import functional as F
    policy_value = []
    origin_value = []
    for seq in state_action_chain:
        state_chain = seq[0]
        action_chain = seq[1]
        reward_chain = seq[2]
        total_reward = 0
        prob=1
        for i,action in enumerate(action_chain):
            with torch.no_grad():
                logits = pi_model(state_chain[i])
            probs = F.softmax(logits, dim=-1)
            prob*=(probs[action].item()*4)
            total_reward+=(GAMMA**i * prob * reward_chain[i])
        policy_value.append(total_reward)
    return np.array(policy_value)

def perform_ope_compare_base(pi_model, state_action_chain, GAMMA=0.99):
    import torch.nn as nn
    from torch.nn import functional as F
    policy_value = []
    origin_value = []
    def baseline_policy(state,last_action):
        theta = state[9].item()
        phase = state[-1].item()
        from cmath import pi
        if abs(theta)<pi/6:
            res = 0
        elif theta>pi/6:
            res = 1
        elif theta<-pi/6:
            res = 2
        if phase==1 or phase==3:
            if last_action in [1,2,3]:
                res = 3
            elif last_action==0:
                res = 0
        return res    
    for seq in state_action_chain:
        state_chain = seq[0]
        action_chain = seq[1]
        reward_chain = seq[2]
        total_reward = 0
        prob=1
        last_action = -1
        for i,action in enumerate(action_chain):
            with torch.no_grad():
                logits = pi_model(state_chain[i])
            probs = F.softmax(logits, dim=-1)
            base_action = baseline_policy(state_chain[i],last_action)
            base_prob = 0.7 if action==base_action else 0.1
            prob*=(probs[action].item()/base_prob)
            last_action = action
            total_reward+=(GAMMA**i * prob * reward_chain[i])
        policy_value.append(total_reward)
    return np.array(policy_value)

def perform_ope_multistep(pi_model, state_action_chain, seq_len, GAMMA=0.99):
    import torch.nn as nn
    from torch.nn import functional as F
    def merge_multistep_state(i,state_chain,action_chain,reward_chain,seq_len):
        res = []
        res.append(state_chain[i])
        pointer = i
        while pointer>0 and len(res)<seq_len:
            pointer-=1
            action = action_chain[pointer]
            if action==0:
                action_reward = torch.tensor([0.003,0.003,reward_chain[i]])
            elif action==1:
                action_reward = torch.tensor([0.001,0.003,reward_chain[i]])
            elif action==2:
                action_reward = torch.tensor([0.003,0.001,reward_chain[i]])
            elif action==3:
                action_reward = torch.tensor([0,0,reward_chain[i]])
            res.append(torch.hstack([state_chain[i],action_reward]))
        while len(res)<seq_len:
            res.append(torch.zeros(len(state_chain[i])+3))
        return torch.hstack(res[::-1])
    policy_value = []
    origin_value = []
    for seq in state_action_chain:
        state_chain = seq[0]
        action_chain = seq[1]
        reward_chain = seq[2]
        total_reward = 0
        prob=1
        for i,action in enumerate(action_chain):
            multistep_state = merge_multistep_state(i,state_chain,action_chain,reward_chain,seq_len)
            with torch.no_grad():
                logits = pi_model(multistep_state)
            probs = F.softmax(logits, dim=-1)
            prob*=(probs[action].item()*4)
            total_reward+=(GAMMA**i * prob * reward_chain[i])
        policy_value.append(total_reward)
    return np.array(policy_value)
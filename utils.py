import torch
import numpy as np
from cmath import pi
scale_factor = torch.tensor([10,10,20,20,10,10, 30,30, 1, 1/pi, 6/pi, 50,25, 1e4, 1/4])

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

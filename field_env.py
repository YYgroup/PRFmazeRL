"""
流场环境,调用IBAMR由当前时间步求解下一个时间步
由下一时间步的数据计算得出下一时间步物体的状态
"""
import os
from turtle import pos
import numpy as np
from typing import Callable, Tuple
from utils import calculate_reward_from_raw_state, transform_raw_state
import torch
from cmath import pi
import random
import math
import time

#读文件名
with open('input2d','r') as f:
    lines = f.readlines()
    for line in lines:
        if 'structure_names' in line:
            filename = line.split()[-1][1:-1]

#读beam的masterPoint的编号顺序
beam_index=[]
with open(filename+".beam",'r') as f:
    num=int(f.readline())
    for i in range(num):
        line=f.readline()
        beam_index.append(int(line.split()[1]))
        k_max = float(line.split()[3]) # k_beam最大值
step_size = 250 #IBAMR中单步的间隔
with open('input2d', 'r') as f:
    lines = f.readlines()
    for line in lines:
        tmp = line.split()
        if len(tmp) > 0 and tmp[0] == 'DUMP':
            step_size = int(tmp[2])
            break    

action_space = []
big = 0.003 
small = 0.001 
temp = [[big, big], [small, big], [big, small], [0, 0]]
for i in temp:
    action_space.append(torch.tensor(i, dtype=torch.float64))
class gridInfo:  
    def __init__(self):  
        self.i = -1
        self.j = -1
        self.level = -1
        self.parent = [-1,-1]
class Flow_Field:
    def __init__(self):
        self.startTime = 0  # 模拟起始时间
        self.endTime = self.read_end_time()  # 终止时间
        self.dt = self.read_dt()  # 时间间隔
        self.currstep = 0
        
        self.currenTimeStep = 0  # 当前时间步，第0步为t=0
        self.currentTime = 0  # 当前时间
        self.Lagpoints = self.read_Lagpoint(self.currenTimeStep)  # 读入当前时间步的拉格朗日点和力
        self.massCenter = self.calculate_mass_center(self.Lagpoints)  # 计算质心坐标
        self.pre_location = self.massCenter

        self.Force = self.read_force(self.currenTimeStep)  # 计算拉格朗日点受力受力
        self.torque = self.calculate_torque(self.Lagpoints, self.massCenter, self.Force)  # 计算对质心的力矩
        #self.actual_force = self.Force.copy()
        #self.actual_torque = self.torque.copy()

        self.start_point = self.massCenter.copy()

        # need to call initialize function
        self.targetPoint = None
        self.pre_targetPoint = None
        self.initTarget = None
        self.targetList = []

        self.target_vel = np.array([0.0, 0.0])

        self.massCenterList = []
        self.massCenterList.append(self.massCenter)

        self.new_obs = np.zeros(22)
        self.action_space = action_space

        self.unknownWallPoints = self.readWallPoints()  # wall points set
        self.knownWallPoints = set()

        self.senseRadius = 0.4
        self.roadWidth = 0.05
        self.deltaRotate = pi/18
        self.last_delta_target = np.array([0, 0])
        
        #self.modifyTime = os.path.getmtime("guidanceVector.txt")

        self.ref_orig = None
        self.grid_wid = 0.05
        self.maze = None
        #self.convertKnownWallPoints2Maze()
        self.target_traj = None
        self.path_idx = None

    # 从输入文件读入终止时间
    @staticmethod
    def read_end_time():
        with open('input2d', 'r') as f:
            lines = f.readlines()
        for line in lines:
            tmp = line.split()
            if len(tmp) != 0 and tmp[0] == 'END_TIME':
                return float(tmp[2])

    # 读入dt
    @staticmethod
    def read_dt():
        with open('input2d', 'r') as f:
            lines = f.readlines()
        dump, dt = 0, 0
        for line in lines:
            tmp = line.split()
            if len(tmp) > 0:
                if tmp[0] == 'DT':
                    dt = float(tmp[2])
                elif tmp[0] == 'DUMP':
                    dump = float(tmp[2])
                    #step_size = int(dump)
                    break
        return dump * dt

   
    @staticmethod
    def rearrange_new(data):
        n=len(data)
        gg = data.copy()
        for i in range(41):
            temp = gg[i].copy()
            gg[i] = gg[79-i].copy()
            gg[79-i] = temp.copy()
        return gg

    @staticmethod
    def readWallPoints():
        with open(filename+'.vertex', 'r') as f:
            lag = np.loadtxt(f.name, unpack=True, skiprows=1)
        wall = lag.T[159:]
        wall = np.array(wall)
        [n, _] = wall.shape
        wallPoints = set()
        for i in range(n):
            wallPoints.add(tuple(wall[i,:]))
        return wallPoints


    def updateKnownWallPoints(self):
        # 修改逻辑:挡着的点是不能被看到的
        temp = set()
        for item in self.unknownWallPoints:
            distance = np.linalg.norm(self.massCenter-item)
            if distance<self.senseRadius:
                temp.add(item)
        for item in self.knownWallPoints:
            distance = np.linalg.norm(self.massCenter-item)
            if distance<self.senseRadius:
                temp.add(item)
        #if self.knownWallPoints:
        possible = set()
        for p1 in temp:
            if p1 in self.knownWallPoints:
                continue
            theta1 = math.atan2(p1[1]-self.massCenter[1],p1[0]-self.massCenter[0])/pi*180
            flag = True
            for p2 in temp:
                if p2[0]!=p1[0] or p2[0]!=p1[0]:
                    theta2 = math.atan2(p2[1]-self.massCenter[1],p2[0]-self.massCenter[0])/pi*180
                    if abs(theta1-theta2)<3 and np.linalg.norm(self.massCenter-np.array(p2))<np.linalg.norm(self.massCenter-np.array(p1))-0.05 and np.linalg.norm(np.array(p2)-np.array(p1))>1.5*1.6/512:
                        flag = False
            if flag:
                possible.add(p1)
        self.knownWallPoints = self.knownWallPoints.union(possible) 
        #else:
        #    self.knownWallPoints = self.knownWallPoints.union(temp)
        """
        for item in self.unknownWallPoints:
            distance = np.linalg.norm(self.massCenter-item)
            if distance<self.senseRadius:
                self.knownWallPoints.add(item)
        """
        print("known points:", len(self.knownWallPoints))
        self.unknownWallPoints = self.unknownWallPoints - self.knownWallPoints

    def outputMassInitTargetKnownWall(self):
        with open("currentWallpoints.txt","w") as f:
            f.write(str(self.massCenter[0]) + ' ' +str(self.massCenter[1]) + '\n')
            f.write(str(self.initTarget[0]) + ' ' +str(self.initTarget[1]) + '\n')
            for point in self.knownWallPoints:
                f.write(str(point[0]) + ' ' +str(point[1]) + '\n')

    def isThereWallBetweenMassAndTarget(self, target=None):
        if target is None:
            target = self.initTarget
        # mass:(x2,y2)   target:(x1,y1)
        A = (self.massCenter[1]-target[1]) #y2-y1
        B = -(self.massCenter[0]-target[0]) #-x2+x1
        C = self.massCenter[0]*target[1] - self.massCenter[1]*target[0] #x2*y1-x1*y2
        pm = self.roadWidth*(A**2+B**2)**0.5
        x1 = target[0]
        y1 = target[1]
        x2 = self.massCenter[0]
        y2 = self.massCenter[1]
        distance = np.linalg.norm(target-self.massCenter)
        roadWidth = self.roadWidth
        for item in self.knownWallPoints:
            x = item[0]
            y = item[1]
            #if (A*item[0]+B*item[1]+C-pm)*(A*item[0]+B*item[1]+C+pm)<0 and (-B*item[0]+A*item[1]-A*self.massCenter[1]+B*self.massCenter[0])*(-B*item[0]+A*item[1]-A*target[1]+B*target[0])<0:
            #if ((y2-y1)*y+(x2-x1)*x-(x2**2-x1**2)/2-(y2**2-y1**2)/2)**2/(((x1-x2)**2+(y1-y2)**2)/4) + (-(y2-y1)*x+(x2-x1)*y+(x1*y2-x2*y1))**2/roadWidth**2 <= (x1-x2)**2+(y1-y2)**2:
            if ((y2-y1)*y+(x2-x1)*x-(x2**2-x1**2)/2-(y2**2-y1**2)/2)**2/(((x1-x2)**2+(y1-y2)**2)/4) + (-(y2-y1)*x+(x2-x1)*y+(x1*y2-x2*y1))**2/roadWidth**2 <= (x1-x2)**2+(y1-y2)**2:
                print(item,"in the way!")
                return True
            if (A*x+B*y+C-pm)*(A*x+B*y+C+pm)<0 and (A*y-B*x-(x2**2-x1**2)/2-(y2**2-y1**2)/2)*(-B*x+A*y-A*y1+B*x1)<0:
                print(item,"in the way!")
                return True
        return False

    def rotate(self,tempTarget, angle):
        [x,y]=tempTarget-self.massCenter
        delta = np.array([x*np.cos(angle)-y*np.sin(angle), y*np.cos(angle)+x*np.sin(angle)])
        return self.massCenter+delta

    def wait4MatlabToCalculateUpdatedTarget(self):
        import time
        mdft = os.path.getmtime("guidanceVector.txt")
        while mdft==self.modifyTime:
            time.sleep(1)
            mdft = os.path.getmtime("guidanceVector.txt")
        self.modifyTime = os.path.getmtime("guidanceVector.txt")
        with open("guidanceVector.txt","r") as f:
            line = f.readline()
        temp = line.split()
        temp1 = self.initTarget - self.massCenter
        dx = float(temp[0])
        dy = float(temp[1])
        self.targetPoint = self.massCenter + min(self.senseRadius,np.linalg.norm(temp1))*np.array([dx,dy])
        
        if dx==0 and dy==0:
            self.targetPoint = self.massCenter + min(self.senseRadius,np.linalg.norm(temp1))*temp1/np.linalg.norm(temp1)
        #if self.isThereWallBetweenMassAndTarget():
        #    self.targetPoint = self.massCenter + min(self.senseRadius,np.linalg.norm(temp1))*temp1/np.linalg.norm(temp1)

        tempTarget = self.targetPoint 
        count_plus = 0
        newTar_plus = tempTarget.copy()
        while self.isThereWallBetweenMassAndTarget(target=newTar_plus) and count_plus<pi/self.deltaRotate*0.75:
            newTar_plus = self.rotate(newTar_plus, self.deltaRotate)
            count_plus+=1 
        count_minus = 0
        newTar_minus = tempTarget.copy() 
        while self.isThereWallBetweenMassAndTarget(target=newTar_minus) and count_minus<pi/self.deltaRotate*0.75:
            newTar_minus = self.rotate(newTar_minus, -self.deltaRotate)
            count_minus+=1
        
        if count_plus<=count_minus and count_plus!=0:
            if np.linalg.norm(self.last_delta_target)==0:
                self.targetPoint = newTar_plus
                self.last_delta_target = newTar_plus - self.massCenter
            else:
                possibleDirection = newTar_plus - self.massCenter
                theta = np.arccos(np.clip(np.dot(possibleDirection,self.last_delta_target)/np.linalg.norm(possibleDirection)/np.linalg.norm(self.last_delta_target),-1,1))
                if abs(theta)/pi*180 > 150 and count_minus!=0:
                    self.targetPoint = newTar_minus
                    self.last_delta_target = newTar_minus - self.massCenter
                else:
                    self.targetPoint = newTar_plus
                    self.last_delta_target = newTar_plus - self.massCenter

        elif count_minus<=count_plus and count_minus!=0:
            if np.linalg.norm(self.last_delta_target)==0:
                self.targetPoint = newTar_minus
                self.last_delta_target = newTar_minus - self.massCenter
            else:
                possibleDirection = newTar_minus - self.massCenter
                theta = np.arccos(np.clip(np.dot(possibleDirection,self.last_delta_target)/np.linalg.norm(possibleDirection)/np.linalg.norm(self.last_delta_target),-1,1))
                if abs(theta)/pi*180 > 150 and count_plus!=0:
                    self.targetPoint = newTar_plus
                    self.last_delta_target = newTar_plus - self.massCenter
                else:
                    self.targetPoint = newTar_minus
                    self.last_delta_target = newTar_minus - self.massCenter

        if not self.isThereWallBetweenMassAndTarget() and np.linalg.norm(temp1)<self.senseRadius:  # 如果指向目标点且近 目标点应该是初始目标点
            self.targetPoint = self.initTarget  #self.massCenter + min(self.senseRadius,np.linalg.norm(temp1))*np.array([dx,dy])
            
        
        """
        tempTarget = (self.targetPoint + self.massCenter)/2
        count_plus = 0
        newTar_plus = tempTarget.copy()
        while self.isThereWallBetweenMassAndTarget(target=newTar_plus) and count_plus<pi/self.deltaRotate*0.75:
            newTar_plus = self.rotate(newTar_plus, self.deltaRotate)
            count_plus+=1 
        count_minus = 0
        newTar_minus = tempTarget.copy() 
        while self.isThereWallBetweenMassAndTarget(target=newTar_minus) and count_minus<pi/self.deltaRotate*0.75:
            newTar_minus = self.rotate(newTar_minus, -self.deltaRotate)
            count_minus+=1
        
        if count_plus<=count_minus and count_plus!=0:
            if np.linalg.norm(self.last_delta_target)==0:
                self.targetPoint = 2*newTar_plus - self.massCenter
                self.last_delta_target = newTar_plus - self.massCenter
            else:
                possibleDirection = newTar_plus - self.massCenter
                theta = np.arccos(np.clip(np.dot(possibleDirection,self.last_delta_target)/np.linalg.norm(possibleDirection)/np.linalg.norm(self.last_delta_target),-1,1))
                if abs(theta)/pi*180 > 120 and count_minus!=0:
                    self.targetPoint = 2*newTar_minus - self.massCenter
                    self.last_delta_target = newTar_minus - self.massCenter
                else:
                    self.targetPoint = 2*newTar_plus - self.massCenter
                    self.last_delta_target = newTar_plus - self.massCenter

        elif count_minus<=count_plus and count_minus!=0:
            if np.linalg.norm(self.last_delta_target)==0:
                self.targetPoint = 2*newTar_minus - self.massCenter
                self.last_delta_target = newTar_minus - self.massCenter
            else:
                possibleDirection = newTar_minus - self.massCenter
                theta = np.arccos(np.clip(np.dot(possibleDirection,self.last_delta_target)/np.linalg.norm(possibleDirection)/np.linalg.norm(self.last_delta_target),-1,1))
                if abs(theta)/pi*180 > 120 and count_plus!=0:
                    self.targetPoint = 2*newTar_plus - self.massCenter
                    self.last_delta_target = newTar_plus - self.massCenter
                else:
                    self.targetPoint = 2*newTar_minus - self.massCenter
                    self.last_delta_target = newTar_minus - self.massCenter
        """
        

    def SenseSurroundingAreaAndChangeCourse(self):
        self.updateKnownWallPoints()
        self.outputMassInitTargetKnownWall()
        self.wait4MatlabToCalculateUpdatedTarget()

    def SenseSurroundingAreaAndChangeCourse_new(self):
        self.updateKnownWallPoints()
        self.convertKnownWallPoints2Maze()

        #self.planNewInit()   #探索未知环境
        #self.convertKnownWallPoints2Maze()
        
        print("init:",self.initTarget,"ref:",self.ref_orig)
        self.planPath()
        self.changeTargetBaseOnTraj()  
        self.showMaze()
        self.outputMassInitTargetKnownWall()
       


    def convertCoord2idx(self, point):
        x = np.floor((point[0]-self.ref_orig[0]-self.grid_wid/2)/self.grid_wid)+1
        if point[0]-self.ref_orig[0]-self.grid_wid/2 <= 0:
            x = 0
        y = np.floor((point[1]-self.ref_orig[1]-self.grid_wid/2)/self.grid_wid)+1
        if point[1]-self.ref_orig[1]-self.grid_wid/2 <=0:
            y = 0
        if np.array_equal(point, self.massCenter) and self.maze[int(x),int(y)]==0:
            center = self.convertIdx2Coord([x,y])
            delta = self.massCenter - center
            angle = math.atan2(delta[1], delta[0])/pi*180 #-180,180
            if abs(angle)<=30:
                x+=1
            elif 30<angle<=60:
                x+=1
                y+=1
            elif 60<angle<=120:
                y+=1
            elif 120<angle<=150:
                x-=1
                y+=1
            elif 150<angle<=180 or -180<=angle<-150:
                x-=1
            elif -150<=angle<-120:
                x-=1
                y-=1
            elif -120<=angle<-60:
                y-=1
            elif -60<=angle<-30:
                x+=1
                y-=1
        return np.array([x,y]).astype(int)
    def convertIdx2Coord(self,idx):
        iidx = np.array(idx)
        return self.ref_orig + self.grid_wid*iidx
    def convertKnownWallPoints2Maze(self):
         # wall 0, road 1
        if self.initTarget is not None:
            minX = min(self.initTarget[0],self.massCenter[0])
            minY = min(self.initTarget[1],self.massCenter[1])
            maxX = max(self.initTarget[0],self.massCenter[0])
            maxY = max(self.initTarget[1],self.massCenter[1])
        else:
            minX = self.massCenter[0]
            minY = self.massCenter[1]
            maxX = self.massCenter[0]
            maxY = self.massCenter[1]
        for point in self.knownWallPoints:
            minX = min(minX,point[0])
            minY = min(minY,point[1])
            maxX = max(maxX,point[0])
            maxY = max(maxY,point[1])
        minX-=0.5
        minY-=0.5
        maxX+=0.5
        maxY+=0.5
        self.ref_orig = np.array([minX,minY])
        nx = np.floor((maxX-minX-self.grid_wid/2)/self.grid_wid).astype(int)+2
        ny = np.floor((maxY-minY-self.grid_wid/2)/self.grid_wid).astype(int)+2
        self.maze = np.ones([nx,ny])
        for point in self.knownWallPoints:
            [x,y] = self.convertCoord2idx(point)
            self.maze[x, y] = 0
        mass = self.convertCoord2idx(self.massCenter)
        maze = self.maze.copy()
        for i in range(2,self.maze.shape[0]-2):
            for j in range(2,self.maze.shape[1]-2):
                if self.maze[i,j]==0 or (i==mass[0] and j==mass[1]):
                    continue
                if self.maze[i-1,j]==0 and self.maze[i+1,j]==0:
                    maze[i,j] = 0
                if self.maze[i,j-1]==0 and self.maze[i,j+1]==0:
                    maze[i,j] = 0
                if self.maze[i+1,j]==0 and self.maze[i,j+1]==0:
                    maze[i,j] = 0
                if self.maze[i+1,j]==0 and self.maze[i,j-1]==0:
                    maze[i,j] = 0
                if self.maze[i-1,j]==0 and self.maze[i,j+1]==0:
                    maze[i,j] = 0
                if self.maze[i-1,j]==0 and self.maze[i,j-1]==0:
                    maze[i,j] = 0
                if self.maze[i-1,j+1]==0 and self.maze[i+1,j-1]==0:
                    maze[i,j] = 0
                if self.maze[i-1,j-1]==0 and self.maze[i+1,j+1]==0:
                    maze[i,j] = 0
                if self.maze[i,j+1]==0 and self.maze[i-1,j-1]==0:
                    maze[i,j] = 0
                if self.maze[i,j+1]==0 and self.maze[i+1,j-1]==0:
                    maze[i,j] = 0
                if self.maze[i-1,j]==0 and self.maze[i+1,j+1]==0:
                    maze[i,j] = 0
                if self.maze[i-1,j]==0 and self.maze[i+1,j-1]==0:
                    maze[i,j] = 0
                if self.maze[i,j-1]==0 and self.maze[i-1,j+1]==0:
                    maze[i,j] = 0
                if self.maze[i,j-1]==0 and self.maze[i+1,j+1]==0:
                    maze[i,j] = 0
                if self.maze[i+1,j]==0 and self.maze[i-1,j+1]==0:
                    maze[i,j] = 0
                if self.maze[i+1,j]==0 and self.maze[i-1,j-1]==0:
                    maze[i,j] = 0
        self.maze = maze


    def showMaze(self):
        from matplotlib.colors import ListedColormap
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('agg')
        colors = [
            (1, 1, 1),   # 白色 (通路，值为 0)
            (1, 0, 0),
            (0, 1, 0),   # 绿色 (特殊区域，值为 0.25)
            (0, 0, 1),   # 蓝色 (路径，值为 0.5)
            (0, 0, 0)    # 黑色 (障碍，值为 1)
        ]
        cmap = ListedColormap(colors)
        maze = self.maze.copy()
        mass = self.convertCoord2idx(self.massCenter)
        target = self.convertCoord2idx(self.initTarget)
        if self.path_idx is not None:
            for idx in self.path_idx:
                maze[idx[0],idx[1]] = 0.75
        maze[mass[0],mass[1]] = 0.25
        maze[target[0],target[1]] = 0.5
        target = self.convertCoord2idx(self.targetPoint)
        maze[target[0],target[1]] = 0.5
        plt.imshow(1-np.flipud(maze.T), cmap=cmap, interpolation='none', vmin=0, vmax=1)  # 使用灰度颜色映射
        plt.title("Maze Map")  # 添加标题
        plt.xticks([])  # 隐藏 x 轴刻度
        plt.yticks([])  # 隐藏 y 轴刻度
        plt.show()
        plt.savefig("currentMaze.png")
        plt.close()
    @staticmethod
    def compute_wall_distances(maze):
        from collections import deque
        rows, cols = maze.shape
        left_dist = np.full((rows, cols), np.inf)
        right_dist = np.full_like(left_dist, np.inf)
        up_dist = np.full_like(left_dist, np.inf)
        down_dist = np.full_like(left_dist, np.inf)
        # 计算左/右方向距离
        for i in range(rows):
            current_dist = 0
            for j in range(cols):
                if maze[i, j] == 0:
                    current_dist = 0
                else:
                    current_dist += 1
                left_dist[i, j] = current_dist
            current_dist = 0
            for j in range(cols-1, -1, -1):
                if maze[i, j] == 0:
                    current_dist = 0
                else:
                    current_dist += 1
                right_dist[i, j] = current_dist
        # 计算上/下方向距离
        for j in range(cols):
            current_dist = 0
            for i in range(rows):
                if maze[i, j] == 0:
                    current_dist = 0
                else:
                    current_dist += 1
                up_dist[i, j] = current_dist
            current_dist = 0
            for i in range(rows-1, -1, -1):
                if maze[i, j] == 0:
                    current_dist = 0
                else:
                    current_dist += 1
                down_dist[i, j] = current_dist
        # BFS计算最近墙壁距离（曼哈顿距离）
        nearest_wall_dist = np.full((rows, cols), np.inf)
        queue = deque()
        # 初始化：所有墙壁的位置距离为0
        for i in range(rows):
            for j in range(cols):
                if maze[i, j] == 0:
                    nearest_wall_dist[i, j] = 0
                    queue.append((i, j))
        # 八方向移动
        dirs = [(-1,-1), (-1,0), (-1,1),
                (0,-1),          (0,1),
                (1,-1),  (1,0), (1,1)]
        while queue:
            x, y = queue.popleft()
            current_dist = nearest_wall_dist[x][y]
            for dx, dy in dirs:
                nx, ny = x+dx, y+dy
                if 0 <= nx < rows and 0 <= ny < cols:
                    if nearest_wall_dist[nx][ny] > current_dist + 1:
                        nearest_wall_dist[nx][ny] = current_dist + 1
                        queue.append((nx, ny))
        return left_dist, right_dist, up_dist, down_dist, nearest_wall_dist
    def pathNeedAdjust(self):
        if self.target_traj is None:
            return True
        current_init = self.convertCoord2idx(self.initTarget)
        pre_init = self.convertCoord2idx(self.target_traj[-1])
        if current_init[0]==pre_init[0] and current_init[1]==pre_init[1]:
            return True
        for point in self.target_traj:
            idx = self.convertCoord2idx(point)
            if self.maze[idx[0],idx[1]]==0:
                return True
        return False
    def planPath(self, start=None, end=None):
        #print(self.target_traj is not None, not self.pathNeedAdjust())
        if self.target_traj is not None and not self.pathNeedAdjust():
            if self.target_traj[-1][0]==self.initTarget[0] and self.target_traj[-1][1]==self.initTarget[1]:
                return
        import heapq
        if start is None:
            start = tuple(self.convertCoord2idx(self.massCenter))
        if end is None:
            end = tuple(self.convertCoord2idx(self.initTarget))
        maze = self.maze
        rows, cols = maze.shape
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        print("start block at ({}, {}), target block at ({}, {})".format(start[0],start[1],end[0],end[1]))
        # 预处理距离矩阵

        left_dist, right_dist, up_dist, down_dist, nearest_wall_dist = self.compute_wall_distances(maze)
        # 方向编码表（8个方向）
        direction_map = {
            (-1, -1): 0,  # 左上
            (-1, 0): 1,   # 上
            (-1, 1): 2,   # 右上
            (0, 1): 3,    # 右
            (1, 1):4,     # 右下
            (1, 0):5,     # 下
            (1, -1):6,    # 左下
            (0, -1):7     # 左
        }
        open_list = []
        initial_f = heuristic(start, end)
        heapq.heappush(open_list, (initial_f, 0, start))
        g_scores = {start: 0}
        parents = {start: (None, None)}  # (parent_node, direction_from_parent)
        closed_set = set()
        while open_list:
            current_f, current_g, current_node = heapq.heappop(open_list)
            if current_node in closed_set:
                continue
            current_g_real = g_scores.get(current_node, float('inf'))
            if current_g > current_g_real:
                continue
            if current_node == end:
                break
            closed_set.add(current_node)
            # 获取父方向
            parent_node, direction_from_parent = parents[current_node]
            # 八方向移动（支持对角线移动）
            for dx, dy in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]: #maybe不让8方向
            #for dx, dy in [(-1,0),(0,-1),(0,1),(1,0)]:
                nx = current_node[0] + dx
                ny = current_node[1] + dy
                if 0 <= nx < rows and 0 <= ny < cols:
                    if maze[nx, ny] != 0:  # 通路判断
                        neighbor = (nx, ny)
                        if neighbor in closed_set:
                            continue
                        # 计算方向编码
                        direction_to_child = direction_map.get((dx, dy), None)
                        if direction_to_child is None:
                            continue  # 非法方向
                        # 计算方向变化惩罚
                        if parent_node is not None:
                            D_Change = abs(direction_from_parent - direction_to_child)
                            if D_Change > 4:
                                D_Change = 8 - D_Change  # 取最小角度差
                            direction_penalty = D_Change * 2.0
                        else:
                            direction_penalty = 0.0  # 起点无惩罚
                        # 墙壁惩罚（要求至少4格远离）
                        wall_dist = nearest_wall_dist[nx][ny]
                        thres = 4
                        wall_penalty = abs(thres - wall_dist)/thres*min(thres, current_g+1) if wall_dist <thres else 0  #np.exp(thres - wall_dist) * 5.0 if wall_dist <thres else 0 
                        # 中心惩罚（保持通路中间）
                        center_penalty = (abs(left_dist[nx, ny] - right_dist[nx, ny]) + abs(up_dist[nx, ny] - down_dist[nx, ny])) * 0.8                        
                        # 总代价计算
                        tentative_g = current_g + 1 + wall_penalty #+ direction_penalty #+ center_penalty
                        # 更新节点信息
                        if (neighbor not in g_scores or 
                            tentative_g < g_scores.get(neighbor, float('inf'))):
                            g_scores[neighbor] = tentative_g
                            parents[neighbor] = (current_node, direction_to_child)
                            new_f = tentative_g + heuristic(neighbor, end)
                            heapq.heappush(open_list, (new_f, tentative_g, neighbor))
        path = []
        current = end
        while current is not None:
            path.append(current)
            current = parents[current][0]
        if path:
            self.path_idx = path[::-1]
            self.target_traj = [self.convertIdx2Coord(idx) for idx in self.path_idx]
            

    def changeTargetBaseOnTraj(self):
        if self.targetPoint is not None:
            self.pre_targetPoint = self.targetPoint.copy()
        if not self.isThereWallBetweenMassAndTarget() and np.linalg.norm(self.massCenter-self.initTarget)<self.senseRadius:  # 如果指向目标点且近 目标点应该是初始目标点
            self.targetPoint = self.initTarget
        else:
            miniDis = 999
            miniIdx = -1
            for i,point in enumerate(self.target_traj):
                if np.linalg.norm(self.massCenter-point)<miniDis:
                    miniDis = np.linalg.norm(self.massCenter-point)
                    miniIdx = i
            idx = miniIdx #0
            dis = np.linalg.norm(self.massCenter-self.target_traj[idx])
            while idx<len(self.target_traj) and dis<self.senseRadius and not self.isThereWallBetweenMassAndTarget(target=self.target_traj[idx]):
                dis = np.linalg.norm(self.massCenter-self.target_traj[idx])
                idx+=1
            idx = max(0, idx-1)
            potential = self.target_traj[idx]
            actual = self.adjustTarget(potential)
            self.targetPoint = actual
        if self.pre_targetPoint is None:
            self.pre_targetPoint = self.targetPoint.copy()

    def adjustTarget(self, potential):
        surrounding = []
        for point in self.knownWallPoints:
            if np.linalg.norm(potential-point)<=0.2:
                surrounding.append(np.array(point))
        #surrounding = np.array(surrounding)
        def obstacleNear(surrouding, point):
            for p in surrouding:
                if np.linalg.norm(p-point)<0.2:
                    return True
            return False
        if surrounding:
            obstacle_center = np.mean(surrounding, axis=0)
            delta = potential - obstacle_center
            delta = delta/np.linalg.norm(delta)
            new_target = potential
            idx = self.convertCoord2idx(new_target)
            count = 0
            while obstacleNear(surrounding, new_target) and count<=10 and np.linalg.norm(self.massCenter-new_target)<0.2 and self.maze[idx[0],idx[1]]==1:
                new_target = new_target + 0.02*delta
                idx = self.convertCoord2idx(new_target)
                count+=1
            if count>0 and self.maze[idx[0],idx[1]]==0:
                new_target = new_target - 0.02*delta
        else:
            new_target = potential
        return new_target

    def planNewInit(self):
        # wall 0
        currentInit = self.initTarget
        potential = []
        maze = self.maze
        minI = maze.shape[0]-1
        minJ = maze.shape[1]-1
        maxI = 0
        maxJ = 0
        for i in range(maze.shape[0]):
            for j in range(maze.shape[1]):
                if maze[i,j]==0:
                    minI = min(minI,i)
                    minJ = min(minJ,j)
                    maxI = max(maxI,i)
                    maxJ = max(maxJ,j)
        mass = self.convertCoord2idx(self.massCenter)
        from collections import deque
        queue = deque()
        queue.append(mass)
        history = set()
        current_level = 0
        while queue:
            point = queue.popleft()
            current_level+=1
            history.add(tuple(point))
            #print(len(queue))
            neibor = [(point[0]-1,point[1]), (point[0]+1,point[1]), (point[0],point[1]+1), (point[0],point[1]-1)]
            for candidate in neibor:
                #if candidate[0]<minI or candidate[0]>maxI or candidate[1]<minJ or candidate[1]>maxJ or maze[candidate[0]][candidate[1]]==0 or tuple(candidate) in history:
                #    continue                 
                #print("1111111111111111111",candidate)
                if candidate[0]<minI or candidate[0]>maxI or candidate[1]<minJ or candidate[1]>maxJ:
                    continue
                if maze[candidate[0]][candidate[1]]==0 or candidate in history:
                    continue

                if minI<candidate[0]<maxI and minJ<candidate[1]<maxJ and maze[candidate[0]][candidate[1]]==1 and candidate not in set(queue):
                    queue.append(candidate)
                if (candidate[0]==minI or candidate[0]==maxI or candidate[1]==minJ or candidate[1]==maxJ) and maze[candidate[0]][candidate[1]]==1:
                    potential.append(self.convertIdx2Coord(candidate))

        if self.currentTime!=0 and self.initTarget is not None and len(potential)>0:
            potential.sort(key=lambda x:np.linalg.norm(x-currentInit))
            self.initTarget = potential[0]
        elif len(potential)==0:
            self.initTarget = self.start_point
        #print("wtf???",self.initTarget)
        elif self.currentTime==0 and len(potential)>0:
            self.initTarget = potential[0]
        
           
                


    # 读入当前时间步的拉格朗日点集合和受力
    @staticmethod
    def read_Lagpoint(timestep:int):
        # t=0初始化
        if timestep == 0:
            with open(filename+'.vertex', 'r') as f:
                lag = np.loadtxt(f.name, unpack=True, skiprows=1)
            lag = lag.T[:159]
        else:
            lag = []
            tag = str(timestep)
            while len(tag) < 5:
                tag = '0' + tag
            lagPointer = "X."+tag
            print('============== reading ' + lagPointer + '================\n')
            with open(os.path.join('hier_data_IB2d', lagPointer), "r") as f:
                for _ in range(3):
                    f.readline()
                lines = f.readlines()
                n = len(lines)//2
                n = 159
                for i in range(n):
                    lag.append([float(lines[2*i]),float(lines[2*i+1])])                
            print('============= read in lagrange points number: ' + str(len(lag)) + '\n')
        return np.array(lag)

    # 计算当前物体质心
    @staticmethod
    def calculate_mass_center(lag):
        return np.mean(lag, axis=0)

    # 计算对心力矩
    @staticmethod
    def calculate_torque(lag, center, Force):
        fx=[]
        fy=[]
        for element in Force:
            fx.append(element[0])
            fy.append(element[1])
        fx=np.array(fx)
        fy=np.array(fy)
        tmp = (lag - center) * np.stack([fy, -fx], axis=1)
        return np.sum(tmp)

    # 读入拉格朗日点上力分布
    @staticmethod
    def read_force(timestep:int):
         # t=0初始化
        if timestep == 0:
            with open(filename+'.vertex', 'r') as f:
                lag = np.loadtxt(f.name, unpack=True, skiprows=1)
            lag = lag.T[:159]
            lag = lag * 0
        else:
            lag = []
            tag = str(timestep)
            while len(tag) < 5:
                tag = '0' + tag
            lagPointer = "F."+tag
            print('============== reading ' + lagPointer + '================\n')
            with open(os.path.join('hier_data_IB2d', lagPointer), "r") as f:
                for _ in range(3):
                    f.readline()
                lines = f.readlines()
                n = len(lines)//2
                n = 159
                for i in range(n):
                    lag.append([float(lines[2*i]),float(lines[2*i+1])])                
            print('============= read in lagrange forces number: ' + str(len(lag)) + '\n')
        return np.array(lag)
    
    @staticmethod
    def update_file_with_line_func(filepath: str, line_func: Callable[[str], Tuple[bool, str]]):
        with open(filepath, 'r') as f:
            lines = f.readlines()
        iter = (line_func(lines) for lines in lines)
        iter = filter(lambda x: x[0], iter)
        iter = (x[1] for x in iter)
        with open(filepath, "w+") as f:
            f.writelines(iter)
  


    def initialize(self, target=np.array([0, 0]),autoPilot=False):
        self.initTarget = target #初始目标点
        # 初始物体状态，读入vertex文件中的构成物体的拉格朗日点
        with open(filename + '.vertex', 'r') as f:
            lag = np.loadtxt(f.name, unpack=True, skiprows=1)
        lag = lag.T[:159]
        n=len(lag)
        # 把点的顺序改成从左到右的
        for i in range(41):
            temp=lag[i].copy()
            lag[i] = lag[79-i].copy()
            lag[79-i]=temp.copy()
        mass_center = np.mean(lag,axis=0)

        if autoPilot==False:
            self.targetPoint = target
        elif autoPilot==True:

            # 自主探索用None
            self.initTarget = None
            # 有目标的要设为targte
            self.initTarget = target

            self.targetPoint = target
            self.SenseSurroundingAreaAndChangeCourse_new() #记得改自主探索时需要改

        self.pre_targetPoint = self.targetPoint.copy()
        self.targetList.append(target) #记录每个时刻目标点
        pre_state = np.array([lag[0][0], lag[0][1], lag[n//2][0], lag[n//2][1], lag[-1][0], lag[-1][1], self.targetPoint[0], self.targetPoint[1], mass_center[0], mass_center[1], 0, 0, 0, 3])
        curr_state = np.array([lag[0][0], lag[0][1], lag[n//2][0], lag[n//2][1], lag[-1][0], lag[-1][1], self.targetPoint[0], self.targetPoint[1], mass_center[0], mass_center[1], 0, 0, 0, 0])
        realfinal=np.append(pre_state, curr_state)
        return realfinal

    
    def plot_traj(self):
        with open('massCenter.txt','w+') as f:
            for element in self.massCenterList:
                f.write(str(element[0]) + ' ' +str(element[1]) + '\n')
        with open('targetTraj.txt','w+') as f:
            for element in self.targetList:
                f.write(str(element[0]) + ' ' + str(element[1]) + '\n')
        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pyplot as plt
        
        from matplotlib import rcParams
        rcParams['font.size'] = 14
        mass = np.array(self.massCenterList)
        target = np.array(self.targetList)
      
        plt.plot(mass[:,0], mass[:,1], label='mass center trajectory',color="blue")

        plt.plot(target[:,0], target[:,1], label='target trajectory',color="green") #临时

        plt.scatter(self.initTarget[0], self.initTarget[1],color="red",s=40,marker='o')
        plt.scatter(self.start_point[0], self.start_point[1],color="red",s=40,marker='o')
        """
        if target[0][0]!=target[3][0] and target[0][1]!=target[3][1]:
            plt.plot(target[:,0], target[:,1], label='target trajectory',color="green")
            n=target.shape[0]
            num=10
            for i in range(2,num+1):
                temp=min(n,n//num*i)
                plt.scatter(target[temp][0], target[temp][1],color="green",s=20,marker='o')
                plt.scatter(mass[temp][0], mass[temp][1],color="blue",s=20,marker='o')
        """
        with open(filename+'.vertex', 'r') as f:
            lag = np.loadtxt(f.name, unpack=True, skiprows=1)
        lag = lag.T
        if lag.shape[0]>159:
            plt.scatter(lag[159:,0], lag[159:,1],color="black",s=4,marker='o')
        if self.knownWallPoints is not None:
            for point in self.knownWallPoints:
                plt.scatter(point[0], point[1],color="yellow",s=4,marker='o')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim([0.1,3.1])
        plt.ylim([0.1,3.1])
        ax=plt.gca()
        ax.set_aspect(1) #bigger for y longer
        plt.legend(loc="best")
        plt.grid(True)
        plt.show()
        plt.savefig("trajectory.png")
        plt.close()

    # 输入action后根据当前的流场信息求解下一步的流场并更新当前的流场状态
    # action 是一个tensor
    # 并返回s_:下一时刻的拉格朗日点的速度+质心3x3的速度分量张量
    # 奖励正比于当前质心与目标点距离的相反数
    def step(self, action):
        with open('../context.txt','r') as f:
            lines=f.readlines()
        with open('../context.txt','w+') as f:
            temp=lines[0].split()
            temp[0]=str(action[0].item())
            temp[1]=str(action[1].item())
            lines[0]=' '.join(temp)+'\n'
            for line in lines:
                f.write(line)

        # 修改input2d中的终止参数
        def update_input2d(line: str) -> Tuple[bool, str]:
            tmp = line.split()
            if len(tmp) > 0:
                if tmp[0] == 'END_TIME':
                    #print('currenTime='+str(self.currentTime)+'正在将tfinal修改为'+str(self.currentTime+self.dt)+'\n')
                    return True, f"END_TIME = {str(self.currentTime+self.dt)}\n"  #去掉了+0.001
            return True, line

        self.update_file_with_line_func('input2d', update_input2d)

        # 脚本调用IBAMR
        currenTime = time.time()

        command_line = "./main2d input2d"
        if self.currenTimeStep!=0:
            command_line = command_line + " ./restart_IB2d/ "+str(self.currenTimeStep)
        os.system(command_line)

    
        lag = str(self.currenTimeStep+step_size)
        while len(lag)<6:
            lag="0"+lag
        file_path = "restart_IB2d/restore."+lag+"/nodes.00001/proc.00000"
        if os.path.exists(file_path):
            mt = os.path.getmtime(file_path)
        else:
            mt=-1
        while not os.path.exists(file_path) or mt<currenTime:
            os.system(command_line)
            if os.path.exists(file_path):
                mt = os.path.getmtime(file_path)
            else:
                mt=-1



        self.currenTimeStep += step_size
        self.currentTime += self.dt
        self.currstep += 1

        # 记录原先位置
        pre_massCenter = self.massCenter
        pre_lag = self.rearrange_new(self.Lagpoints)
        pre_force = np.sum(self.Force, axis=0)
        pre_torque = self.torque
        
        # 更新环境状态
        self.Lagpoints = self.read_Lagpoint(self.currenTimeStep)  # 读入当前时间步的拉格朗日点
        self.Force = self.read_force(self.currenTimeStep)  # 计算拉格朗日点受力受力
        curr_force = np.sum(self.Force, axis=0)
        self.torque = self.calculate_torque(self.Lagpoints, self.massCenter, self.Force) 
        curr_lag = self.rearrange_new(self.Lagpoints)
        self.massCenter = self.calculate_mass_center(self.Lagpoints)  # 计算质心坐标
        self.massCenterList.append(self.massCenter)
        self.targetList.append(np.array([self.targetPoint[0],self.targetPoint[1]]))
        
        distance=np.linalg.norm(self.targetPoint - self.start_point) #初始位置和目标位置的距离

        n = len(pre_lag)
        self.new_obs = np.array([pre_lag[0][0], pre_lag[0][1], pre_lag[n//2][0], pre_lag[n//2][1], pre_lag[-1][0], pre_lag[-1][1], \
                                 self.pre_targetPoint[0], self.pre_targetPoint[1], pre_massCenter[0], pre_massCenter[1], pre_force[0], pre_force[1], pre_torque, (self.currstep+3)%4, \
                                curr_lag[0][0], curr_lag[0][1], curr_lag[n//2][0], curr_lag[n//2][1], curr_lag[-1][0], curr_lag[-1][1], \
                                    self.targetPoint[0], self.targetPoint[1], self.massCenter[0], self.massCenter[1], curr_force[0], curr_force[1], self.torque, self.currstep%4])

        xxx = curr_lag[:,0]
        yyy = curr_lag[:,1]
        # 判断终止
        #if self.currentTime < self.endTime and max(xxx)<3.1 and min(xxx)>0.1 and max(yyy)<3.1 and min(yyy)>0.1 and np.linalg.norm(self.massCenter - self.targetPoint)> 0.05:
        minX = 0.1
        minY = 0.1
        maxX = 3.1 #3.1
        maxY = 3.1 #3.1
        if self.currentTime < self.endTime and max(xxx)<maxX and min(xxx)>minX and max(yyy)<maxY and min(yyy)>minY and np.linalg.norm(self.massCenter - self.initTarget)> 0.05:
            done = 0
        else:
            done = 1
            # 还要把input2d中的终止时间改回初始值
            def recover_input2d(line: str) -> Tuple[bool, str]:
                tmp = line.split()
                if len(tmp) > 0:
                    if tmp[0] == 'END_TIME':
                        return True, f"END_TIME = {str(self.endTime)}\n"
                return True, line

            self.update_file_with_line_func('input2d', recover_input2d)



        s_ = torch.tensor(self.new_obs, dtype=torch.float64)
        reward = calculate_reward_from_raw_state(s_)
        return s_, reward, done





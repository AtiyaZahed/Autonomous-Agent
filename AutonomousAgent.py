import os
import string
import concurrent.futures
from collections import deque

import numpy as np
import time
import queue
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T
import random
import pandas as pd

N = 5  # number of functions max 21 in file
W = 4  # number of terminals max 22
U = 3  # number of edge devices max 23
V = 2  # number of cloud devices max 10
numberofEP = 5000
# edge_res_asli=[[1, 1024, 1024 ,2],[2, 512, 1024, 4],[1 ,256, 1024, 8]]
# server_res_asli=[[4 ,2049, 10240, 4096],[5 ,5120 ,10240, 4096]]
# cpu 1->.2 5->1

edge_res=[[.2, 1024/5120, 1024/10240 ,1024/4096],[.4, 512/5120, 1024/10240, 2048/4096],[.2 ,256/5120, 1024/10240, 4096/4096]]
server_res=[[.8 ,2049/5120, 10240/10240, 4096/4096],[1 ,5120/5120,10240/10240, 4096/4096]]

sum_of_u_e = edge_res[0][0] + edge_res[1][0] + edge_res[2][0]
sum_of_u_s = server_res[0][0] + server_res[1][0]
Fun_name = ['cpu', 'mem meg', 'Disk meg', 'BW kbpersec', 'deadline msec']
# func_res_asli=[[1, 128, 128, 128, 10],
#           [2, 256, 128, 256 ,20],
#           [4 ,128 ,256 ,256, 40],
#           [1, 100 ,10  ,10 ,80],
#           [2 ,200, 20, 10, 160]]

func_res=[[1/5, 128/5120, 128/10240, 128/4096, 10],
          [2/5, 256/5120, 128/10240, 256/4096 ,20],
          [4/5 ,128/5120 ,256/10240 ,256/4096, 40],
          [1/5, 100/5120 ,10/10240  ,10/4096 ,80],
          [2/5 ,200/5120, 20/10240, 10/4096, 160]]


func_res_runtim = [.002, .004, .008, .001, .004]
latency = [[1, 2, 1, 10, 20], [1.5, 2.5, 1.5, 10, 20], [3, 1, 2, 15, 15], [2, 2, 1, 12, 16]]

#os.remove('environment.csv')
#os.remove('end_quality_parameter.csv')
#os.remove('output.csv')
#os.remove('quality_parameter.csv')
batch_size = 256


# from util import plot_learning_curve

class LinearDeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, input_dims):
        super(LinearDeepQNetwork, self).__init__()

        self.fc1 = nn.Linear(input_dims, 128)
        self.fc2 = nn.Linear(128, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    # takes the current and returns list of actions
    def forward(self, state):
        layer1 = F.relu(self.fc1(state))
        actions = self.fc2(layer1)
        return actions


class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def store(self, state, action, reward, next_state, done):
        self.buffer.append([state, action, reward, next_state, done])

    def sample(self):
        sample = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, done = map(np.asarray, zip(*sample))
        states = np.array(states).reshape(batch_size, -1)
        next_states = np.array(next_states).reshape(batch_size, -1)
        return states, actions, rewards, next_states, done

    def size(self):
        return len(self.buffer)


class Agent():
    def __init__(self, input_dims, n_actions, eps_dec, lr, gamma=0.99,
                 epsilon=1.0, eps_min=0.01):
        self.lr = lr
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = eps_dec
        self.eps_min = eps_min
        self.action_space = [i for i in range(self.n_actions)]
        self.exploration = 0
        self.expolitition = 0
        self.buffer = ReplayBuffer()
        self.Q = LinearDeepQNetwork(self.lr, self.n_actions, self.input_dims)
        self.target_Q = LinearDeepQNetwork(self.lr, self.n_actions, self.input_dims)
        self.update_target()
        self.lo = 0

    def update_target(self):
        # param =self.Q.parameters()
        # self.target_Q=copy.deepcopy(param)
        self.target_Q = copy.deepcopy(self.Q)

    def choose_action(self, state):
        if np.random.random() > self.epsilon:
            state1 = T.tensor(state, dtype=T.float).to(self.Q.device)
            # state =
            actions = self.Q.forward(state1)
            action = T.argmax(actions).item()
            self.expolitition += 1
            # print("notrandom")
        else:
            action = np.random.choice(self.action_space)
            self.exploration += 1
            # print("random")
        # print("action:",action)
        return action

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
            if self.epsilon > self.eps_min else self.eps_min

    def learn(self, state, action, reward, state_):
        self.Q.optimizer.zero_grad()
        states = T.tensor(state, dtype=T.float).to(self.Q.device)
        actions = T.tensor(action).to(self.Q.device)
        rewards = T.tensor(reward).to(self.Q.device)
        states_ = T.tensor(state_, dtype=T.float).to(self.Q.device)

        q_pred = self.Q.forward(states)[actions]

        # q_next = self.Q.forward(states_).max()
        q_next = self.target_Q.forward(states_).max()

        q_target = reward + self.gamma * q_next

        loss = self.Q.loss(q_target, q_pred).to(self.Q.device)
        self.lo = loss.detach()
        loss.backward()
        self.Q.optimizer.step()
        self.decrement_epsilon()

    def processDQN_stage(self, initial_state):
        action = self.choose_action(initial_state)
        return action


class createTASK:

    def __init__(self, numOFtask, memory_size, minibatch_size):
        # تابع برای تولید تاخیر با توزیع نمایی
        self.num_functions_to_produce = numOFtask  # تعداد توابع مورد نظر برای تولید
        # self.produced_functions_count = 0  # شمارنده توابع تولید شده
        self.Alltask = []
        self.Terminal_Fun_Lambda = [[] for _ in range(W + 1)]
        self.TFL = []
        self.Q = queue.Queue()
        with open('zTFLambda.txt', 'r') as file:
            lines = file.readlines()
            for line in lines:
                row = [int(num) for num in line.strip().split()]
                self.TFL.append(row)
        for r in range(0, W + 1):
            for c in range(0, N + 1):
                self.Terminal_Fun_Lambda[r].append(self.TFL[r][c])

    def exponential_delay(self, lam):
        if lam > 0:
            return np.random.exponential(1 / lam)
        else:
            return None  # برای جلوگیری از تاخیر صفر

    # تابع برای تولید شناسه یکتا
    def generate_unique_id(self):
        length = 4
        characters = string.ascii_letters + string.digits
        unique_id = ''.join(random.choice(characters) for i in range(length))
        return unique_id

    # تابع تولید کننده
    def produce_function(self, id_fun, id_Dv):

        # اگر lam برای این تابع صفر است، از تابع خارج شو.
        if self.Terminal_Fun_Lambda[id_Dv][id_fun] == 0:
            # print("notok")
            return
        # تولید شناسه یکتا برای هر فراخوانی
        unique_id = self.generate_unique_id()
        # print(unique_id)
        # اضافه کردن عنصر به صف
        self.Q.put((unique_id, id_fun, id_Dv, time.time()))
        print(f"Function {id_fun} of Device {id_Dv} with ID {unique_id} produced at {time.time()}")
        # تاخیر تصادفی
        delay = self.exponential_delay(self.Terminal_Fun_Lambda[id_Dv][id_fun])
        if delay is not None:
            time.sleep(delay)

    def createTASK(self):
        for _ in range(self.num_functions_to_produce):
            # id_Dv = np.random.randint(1, W + 1)  # انتخاب دستگاه به صورت تصادفی
            # id_fun = np.random.randint(1, N + 1)  # انتخاب تابع به صورت تصادفی
            alpha = 0.8
            id_Dv = np.random.randint(0, W)  # انتخاب دستگاه به صورت تصادفی
            id_fun = np.random.randint(0, N)
            priority = (alpha * (1 / func_res[id_fun][4])) + ((1 - alpha) * (1 / (time.time())))

            unique_id = self.generate_unique_id()
            # print(id_fun,id_Dv)
            # if self.Terminal_Fun_Lambda[id_Dv ][id_fun] > 0

            if self.Terminal_Fun_Lambda[id_Dv + 1][
                id_fun + 1] > 0:  # اطمینان حاصل کنید که lam برای هر تابع بیشتر از صفر است
                self.Alltask.append(Task(unique_id, float(priority),
                                         float(func_res[id_fun][0]),
                                         float(func_res[id_fun][1]),
                                         float(func_res[id_fun][2]),
                                         float(func_res[id_fun][3]),
                                         1, id_Dv, float(func_res[id_fun][4]), id_fun))
                # with open('requests.txt', 'w') as file:
                #     file.write(' '.join(unique_id+ str(float (priority))+
                #                          str(func_res[id_fun][0])+
                #                          str(func_res[id_fun][1])+
                #                          str(func_res[id_fun][2])+
                #                          str(func_res[id_fun][3])+
                #                          "1"+str(id_Dv) + '\n'))

    def printtask(self):
        for j in self.Alltask:
            print(j.jobID, ",", j.CPU, ",", j.RAM, ",", j.disk, ",", j.bw, ",",
                  j.runtime, ",", j.ddl, ",", j.teminaldevice, ",", j.status)
        print("number of tasks:", len(self.Alltask))


class Task(object):

    def __init__(self, jobID, priority, CPU, RAM, disk, BW, status, terminaldevice, ddl,
                 id_func):  # ,deadline random select

        self.jobID = jobID

        self.priority = priority
        self.CPU = CPU
        self.RAM = RAM
        self.disk = disk
        self.bw = BW
        self.runtime = func_res_runtim[id_func]

        self.ddl = self.runtime + time.time() + (ddl * 10000)
        self.endtime = 0

        self.status = status  # -1: rejected, 0: finished, 1: ready, 2: running
        self.teminaldevice = terminaldevice
        self.processordevice = -1000
        self.id_func = id_func
        self.cost=0
        self.uti=0
        self.retime=0


class environment(object):

    def __init__(self, numTASK):
        self.task = []
        # self.copytask=[]
        self.crtask = createTASK(numTASK, 0, 0)

        self.edge_res = edge_res
        self.server_res = server_res
        self.edge_av = []
        self.server_av = []
        self.ResourcesTemp = self.Resources = self.reshape_res_list(self.edge_res, self.server_res)
        self.RunTask = []
        self.RejecTask = []
        self.Reward = []
        self.exporation_rate = []
        self.duration_time = []
        self.edge_num = []
        self.server_num = []
        self.loss_ep = []
        self.cost_av = 0
        self.utility_av = 0
        self.delay_av = 0
        self.task_counter = 0

    def rewardFcn(self, action, task):
        # resid = action * 4
        self.task_counter = self.task_counter + 1
        # #############################################
        # ########################### delay

        d1 = 2 * (latency[task.teminaldevice][action] / 1000)
        d2 = d1 + task.runtime
        self.delay_av = ((self.task_counter - 1) * self.delay_av + d2) / self.task_counter
        ###########
        if action == 0 or action == 1 or action == 2:
            utility = 1
        else:
            utility = 0
        self.utility_av = ((self.task_counter - 1) * self.utility_av + utility) / self.task_counter
        ####################
        A = 1
        B = 5
        D = 10
        H = 100
        costlist_edg = [(10 / A), (10 / B), (5 / D), (5 / H)]
        costlist_server = [(30 / A), (30 / B), (10 / D), (10 / H)]

        if action > 2:
            cost = task.CPU * costlist_server[0] + task.RAM * costlist_server[1] + task.disk * costlist_server[
                2] + task.bw * costlist_server[3]

        else:
            cost = task.CPU * costlist_edg[0] + task.RAM * costlist_edg[1] + task.disk * costlist_edg[2] + task.bw * \
                   costlist_edg[3]

        self.cost_av = ((self.task_counter - 1) * self.cost_av + cost) / self.task_counter

        ###############################
        #print("d2 cost utility av", self.task_counter, d2, self.delay_av, cost, self.cost_av, utility, self.utility_av)
        if ((d2<0.0018348) or (utility>=0.5) or (cost<=4)):
            reward = 1
        else:
            reward = -1
        df = pd.DataFrame({
            'cost_av': [self.cost_av],
            'delay_av': [self.delay_av],
            'utilization_av': [self.utility_av]
        })

        # اضافه کردن به فایل CSV
        df.to_csv('quality_p0035000.csv', mode='a', header=False, index=False)

        return reward,self.cost_av,self.delay_av,self.utility_av

    def reshape_res_list(self, list1, list2):
        res = np.array(list1 + list2)
        # print(res.reshape(res.shape[0]*res.shape[1]))
        return res.reshape(res.shape[0] * res.shape[1])

    def generateQueue(self):
        self.crtask.create_task()
        # self.crtask.taskQueue()
        self.task = self.crtask.task

    def AssignResCheckRej(self, action, task):

        resid = action * 4
        # print("resid , device",resid,action)

        if self.ResourcesTemp[resid] >= task.CPU:
            self.ResourcesTemp[resid] -= float(task.CPU)
        else:

            return 1
        if self.ResourcesTemp[resid + 1] >= task.RAM:
            self.ResourcesTemp[resid + 1] -= float(task.RAM)

        else:

            return 1
        if self.ResourcesTemp[resid + 2] >= task.disk:
            self.ResourcesTemp[resid + 2] -= float(task.disk)

        else:

            return 1
        if self.ResourcesTemp[resid + 3] >= task.bw:
            self.ResourcesTemp[resid + 3] -= float(task.bw)

        else:

            return 1

        return 2

    def releaseRES(self, task):

        resid = task.processordevice * 4
        self.ResourcesTemp[resid] += float(task.CPU)
        self.ResourcesTemp[resid + 1] += float(task.RAM)
        self.ResourcesTemp[resid + 2] += float(task.disk)
        self.ResourcesTemp[resid + 3] += float(task.bw)

    def UpdateState(self):
        return self.ResourcesTemp

    def train(self):
        # self.generateQueue()
        # self.crtask.printtask()
        self.crtask.createTASK()
        self.task = copy.deepcopy(self.crtask.Alltask)
        # self.task = self.crtask.Alltask
        # self.copytask.extend(self.task)
        print("befor train,number of all task:", len(self.task))
        input_stage = self.reshape_res_list(self.edge_res, self.server_res)
        input_stage = self.ResourcesTemp
        input_actions = len(self.edge_res) + len(self.server_res)
        # print("input_stage,input_actions",input_stage,input_actions)
        Agent_stage = Agent(lr=0.003, input_dims=len(input_stage),
                            n_actions=input_actions, eps_dec=1e-5)
        stage_current_state = input_stage

        for i in range(numberofEP):
            sumR = 0
            sumCost=0
            sumDelay=0
            sumUti=0
            DUC=0
            self.task_counter = 0
            if len(self.task) == 0:
                # self.task.extend(self.copytask)
                self.task = copy.deepcopy(self.crtask.Alltask)
            print("number of episode and number of task:", i, len(self.task))
            starttime = time.time()
            while len(self.task) != 0:

                for t in self.task:

                    if t.ddl < time.time():
                        self.RejecTask.append(t.jobID)
                        self.task.remove(t)
                        # print("reject", t.jobID, t.status)

                    elif t.status == 1:  # ready

                        stage_action = Agent_stage.processDQN_stage(stage_current_state)
                        assignSUC = self.AssignResCheckRej(stage_action, t)
                        # print("statuse 1 stage_action assignSUC ",stage_action,assignSUC)
                        # if assignSUC==1:# RESource not enough
                        #     #print("continue")
                        #     continue
                        if assignSUC == 2:
                            # print("part 2")

                            t.endtime = time.time() + t.runtime
                            t.status = 2
                            t.processordevice = stage_action
                            # print("kkk",t.jobID,t.status,t.device,t.endtime)
                            stage_next_state = self.UpdateState()
                            reward_stage,Co,De,Uti = self.rewardFcn(stage_action, t)
                            Agent_stage.buffer.store(stage_current_state,
                                                     stage_action,
                                                     reward_stage,
                                                     stage_next_state,
                                                     True if len(self.task) > 0 else False)
                            sumR = sumR + reward_stage
                            sumCost=sumCost+Co
                            sumDelay=sumDelay+De
                            sumUti=sumUti+Uti
                            DUC=DUC+1

                            # Agent_stage.learn(stage_current_state, stage_action,reward_stage, stage_next_state)
                            stage_current_state = stage_next_state
                    elif t.status == 2 and t.endtime < time.time():  # cheak rejection
                        self.releaseRES(t)
                        t.status = 0
                        self.RunTask.append(t)
                        # print("add:", t.jobID)
                        self.task.remove(t)
            # replay_experience
            if (Agent_stage.buffer.size() >= batch_size):
                states, actions, rewards, next_states, done = Agent_stage.buffer.sample()
                for i in range(len(states)):
                    Agent_stage.learn(states[i], actions[i], rewards[i],
                                      next_states[i])
            Agent_stage.update_target()

            self.report()
            self.Reward.append(sumR)
            # aa=(Agent_stage.lo.real)
            print(Agent_stage.lo.real)
            self.loss_ep.append(Agent_stage.lo)
            self.exporation_rate.append(Agent_stage.expolitition / Agent_stage.exploration)
            self.duration_time.append(time.time() - starttime)
            print("sum of rewards", sumR)
            print("exploration rate", Agent_stage.exploration, Agent_stage.expolitition,
                  Agent_stage.expolitition / Agent_stage.exploration)
            # print("epsilon value", Agent_stage.epsilon)
            print("duration time:", time.time() - starttime)

        torch.save(Agent_stage.Q, "entire_model.pt")
        print("avrage e s :", np.mean(self.edge_av), np.mean(self.server_av))

        #df = pd.DataFrame({
        #    'cost_av': [sumCost / DUC],
        #    'delay_av': [sumDelay / DUC],
        #    'utilization_av': [sumUti / DUC]
        #})

        # اضافه کردن به فایل CSV
        #df.to_csv('quality_parameter.csv', mode='a', header=False, index=False)
        return sumCost,sumDelay,sumUti,DUC



    def report(self):
        cloudnum = 0
        edgenum = 0
        for j in self.RunTask:
            # print(j.jobID,",",j.priority, ",", j.CPU, ",", j.RAM, ",", j.disk, ",",
            #       j.bw, ",",j.runtime, ",", j.ddl, ",",j.endtime,",",
            #       j.teminaldevice, ",", j.status,",",j.processordevice)
            if (j.processordevice > 2):
                cloudnum += 1
            else:
                edgenum += 1
        self.edge_av.append(edgenum)
        self.server_av.append(cloudnum)
        print("number of run task", len(self.RunTask))
        self.RunTask.clear()
        print("number of reject task", len(self.RejecTask))
        self.RejecTask.clear()
        print("edge numb, cloud numb", edgenum, cloudnum)
        self.edge_num.append(edgenum)
        self.server_num.append(cloudnum)



    def savez(self):
        np.savez('npsave.npz', Reward=np.asarray(self.Reward), Duration=np.asarray(
            self.duration_time), edge_num=np.asarray(self.edge_num)
                 , server_num=np.asarray(self.server_num)
                 , loss=np.asarray(self.loss_ep))
        #df = pd.DataFrame({
        #    'Reward': self.Reward,
        #    'Duration': self.duration_time,
        #    'edge_num': self.edge_num,
        ##    'server_num': self.server_num,
        #    'loss': self.loss_ep
        #})

        # ذخیره دیتافریم به فرمت CSV
        #df.to_csv('output.csv', index=False)

    def saveparameter(self,CO,DE,UT,CDU):
        df = pd.DataFrame({
            'task_counter': self.task_counter,
            'Cost': self.cost_av,
            'utilization': self.utility_av,
            'responcetime': self.delay_av,
            'select': self.edge_av
        })

        # ذخیره دیتافریم به فرمت CSV
        df.to_csv('coutre0035000.csv', index=False)
        df = pd.DataFrame({
            'cost_av': [CO/CDU],
            'delay_av': [DE/CDU],
            'utilization_av': [UT/CDU]
        })

        # اضافه کردن به فایل CSV
        df.to_csv('end_quality_p0035000.csv', mode='a', header=False, index=False)
        df = pd.DataFrame({
            'Reward': self.Reward,
            'Duration': self.duration_time,
            'edge_num': self.edge_num,
            'server_num': self.server_num,
            'loss': self.loss_ep
        })

        # ذخیره دیتافریم به فرمت CSV
        df.to_csv('finaly0035000.csv', index=False)



    def loadfile(self):
        import matplotlib.pyplot as plt
        # np.load(path, allow_pickle=True)
        f1 = np.load('npsave.npz', allow_pickle=True)


        print(f1.files)
        re = f1.files[0]
        edge = f1.files[2]
        server = f1.files[3]
        lo = f1.files[4]
        print(f1[lo])
        xdata = np.arange(numberofEP)

        # plot the data
        fig = plt.figure()
        yd = f1[re]
        # ydata=[x for idx,x in enumerate(yd)  if idx%50==0 ]
        # ymax=yd[0]
        # xdata=[]
        # ydata=[]
        # for x,y in enumerate(yd):
        #     if y>ymax:
        #         ymax=y
        #         xdata.append(x)
        #         ydata.append(y)

        ax = fig.add_subplot(1, 1, 1)
        ax.plot(xdata, yd, color='tab:blue')
        # ax.plot(xdata1, f1[server], color='tab:orange')
        # ax.plot(xdata1, f1[lo], color='tab:blue')
        plt.show()
        #df = pd.DataFrame(f1)

        # ذخیره دیتافریم به فرمت CSV
       # df.to_csv('output.csv', index=False)


e1 = environment(100)
CO,DE,UT,CDU=e1.train()
e1.savez()
e1.saveparameter(CO,DE,UT,CDU)
e1.loadfile()
# e1.crtask.createTASK()
# e1.crtask.printtask()

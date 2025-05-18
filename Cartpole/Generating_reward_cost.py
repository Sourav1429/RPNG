# -*- coding: utf-8 -*-
"""
Created on Sun May 18 11:00:36 2025

@author: Sourav
"""
import numpy as np
import gym
import pickle
env = gym.make('CartPole-v1')

OBS_BOUNDS = [
    (-4.8, 4.8),         # Cart position
    (-5.0, 5.0),         # Cart velocity
    (-0.418, 0.418),     # Pole angle (~24 degrees)
    (-5.0, 5.0)          # Pole angular velocity
]

def create_bins(n_bins):
    return [np.linspace(low, high, n_bins - 1) for low, high in OBS_BOUNDS]

def discretize_state(state, bins):
    clipped = np.clip(state, [b[0] for b in OBS_BOUNDS], [b[1] for b in OBS_BOUNDS])
    return tuple(np.digitize(clipped[i], bins[i]) for i in range(len(clipped)))

def wrap_states(s,n_bins):
    return sum((d-1) * (n_bins ** i) for i, d in enumerate(reversed(s)))

def unwrap_states(s,n_bins):
    ret = np.zeros(4,dtype=np.int8)
    count=1
    while(s>0):
        ret[4-count] = s%n_bins
        s = s//n_bins
        count+=1
    return ret
    '''s_0 = s//int(np.power((n_bins-1),3))
    s_1 = int((s - s_0*np.power((n_bins-1),3)))//int(np.power((n_bins-1),2))
    s_2 = int((s - s_0*np.power((n_bins-1),3)-s_1*np.power((n_bins-1),2)))//(n_bins-1)
    s_3 = s - s_0*np.power((n_bins-1),3)-s_1*np.power((n_bins-1),2) - s_2*(n_bins-1)
    s_3 = s%(n_bins)
    s_2 = (int(s//(n_bins-1)))%(n_bins)
    s_1 = (int(s//np.power((n_bins-1),2)))%(n_bins)
    s_0 = (int(s//np.power((n_bins-1),3)))%(n_bins)
    return [s_0,s_1,s_2,s_3]'''

n_bins = 4
bins = create_bins(n_bins+1)
# state = np.array([-1.6, -5, -0.418, -5])
# s = discretize_state(state,bins)
# aw = wrap_states(s,n_bins)
# print(s)
# print("Wrapping=",aw)
# print("Unwrapping=",unwrap_states(aw,n_bins))
nS = int(np.power(n_bins,4))
nA = env.action_space.n
R = np.zeros((nS,nA))
C = np.zeros((nS,nA))
#print(nS)
#print(nA)
obs = env.reset()
for a in range(nA):
    for s in range(nS):
        obs = np.array(unwrap_states(s,n_bins),dtype=np.float16)
        #print(obs)
        obs[0] = bins[0][int(obs[0])]
        obs[1] = bins[1][int(obs[1])]
        obs[2] = bins[2][int(obs[2])]
        obs[3] = bins[3][int(obs[3])]
        #print(obs)
        #input()
        env.state = obs
        action = a
        _,r,_,_,_ = env.step(a)
        R[s,a] = r
        C[s,a] = np.abs(np.abs(obs[0]) - 0)/4.80078125;

with open('rew_Cartpole.pkl','wb') as f:
    pickle.dump(R,f)
f.close()

with open('constraint_cost_Cartpole.pkl','wb') as f:
    pickle.dump(C,f)
f.close()    

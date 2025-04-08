# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 20:47:19 2025

@author: Sourav
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from KL_uncertainity_evaluator import Robust_pol_Kl_uncertainity
import time

class TabularPolicy(nn.Module):
    def __init__(self, n_states, n_actions):
        super().__init__()
        self.logits = nn.Parameter(torch.randn(n_states, n_actions))

    def forward(self, state):
        probs = F.softmax(self.logits[state], dim=-1)
        return torch.distributions.Categorical(probs=probs)

class RNPG:
    def __init__(self,env,r_oracle_obj,alpha,lambda_,theta):
        self.env = env
        self.oracle_obj = r_oracle_obj
        self.alpha = alpha
        self.lambda_ = lambda_
        self.theta = theta
        self.value_func_store = []
        self.cost_func_store = []
        self.value_func_grad_store = []
        self.cost_func_grad_store = []
        self.policy = TabularPolicy(self.env.nS, self.env.nA)
    def find_choice(self,pol):
        J,J_grad = None,None
        J_v,J_v_grad = self.oracle_obj(pol)#send_for_value function
        J_c,J_c_grad = self.oracle_obj(pol)#send_for_cost_function
        self.value_func_store.append(J_v)
        self.value_func_grad_store.append(J_v_grad)
        self.cost_func_store.append(J_c)
        self.cost_func_grad_store.append(J_c_grad)
        choice = np.max([J_v/self.lambda_,J_c])
        if(choice == 0):
            J,J_grad = J_v,J_v_grad
        else:
            J,J_grad = J_c,J_c_grad
        return J,J_grad
    def get_pol(self):
        states,actions = self.env.nS,self.env.nA
        ret_pol = np.zeros((states,actions))
        for s, a in zip(states, actions):
            ret_pol[s] = self.policy(s)
        return ret_pol
    def compute_fisher(self):
        grads = []
        states = np.arange(self.env.nS)
        actions = np.arange(self.env.nA)
        for s, a in zip(states, actions):
            dist = self.policy(s)
            log_prob = dist.log_prob(a)
            self.policy.zero_grad()
            log_prob.backward(retain_graph=True)
            grad = torch.cat([
                p.grad.view(-1) if p.grad is not None else torch.zeros_like(p).view(-1)
                for p in self.policy.parameters()
            ])
            grads.append(grad)
        grad_matrix = torch.stack(grads)
        return grad_matrix.T @ grad_matrix / grad_matrix.size(0)
    def train_all(self,T):
        for t in range(T):
            #pol = self.get_policy(self.theta)
            pol = self.get_pol()
            J,J_grad = self.find_choice(pol)
            F = self.compute_fisher()
            self.theta = 1/self.lambda_*(self.theta - 0.5*np.matmul(np.linalg.pinv(F),J_grad))
            
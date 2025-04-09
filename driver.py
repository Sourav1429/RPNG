# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 13:35:17 2025

@author: Sourav
"""
import numpy as np
from KL_uncertainity_evaluator import Robust_pol_Kl_uncertainity
import time
from Machine_Rep import Machine_Replacement
from RPNG import *

env = Machine_Replacement()
r,c = env.gen_expected_reward(),env.gen_expected_cost()
cost_list = [r,c]
init_dist = np.exp(np.random.randn(env.nS))
init_dist = init_dist/np.sum(init_dist)
oracle = Robust_pol_Kl_uncertainity(env.nS, env.nA, cost_list, init_dist)
alpha = 0.01
lambda_ = 10
theta = 
rpng_obj = RPNG(env,oracle,alpha,lambda_,theta)


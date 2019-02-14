# -*- coding: utf-8 -*-
"""
Spyder Editor

Heart Disease Prediction using PSO based clustering method
"""

import numpy as np
import matplotlib.pyplot as plt
#'exec(%matplotlib auto)'

def g_best_cal(p_best):
    return( max(map(lambda x: x,p_best)))

def f_func(parc,fval):
    best=0
    for item in parc:
        for i in parc and i != item:
            

n_gen = 50 #no of generations
n_vel = np.zeros([100,2]) #velocity of particles
p_best = np.zeros([100,2]) 
f_val = np.zeros([100,1]) #fitness value of each particle
pos_x = np.random.uniform(low=-500,high=500,size=(100,2)) #postion of each particle
g_best = np.zeros([1,2])
U1,U2 = np.random.rand(),np.random.rand() #user  variables
np_parc = np.array(np.random.randint(2,size=14)) #binary vector particles
for i in range(99):
    to_app = np.random.randint(2,size=14)
    np_parc = np.append(np_parc,to_app)
np_parc=np_parc.reshape(100,14)
#p_best[0]=[5,2]
g_best[0]= g_best_cal(p_best[:,0])
g_best[1]= g_best_cal(p_best[:,1])
#p_best_g[:,1] = g_best
#p_best_g = p_best_g.reshape(100,2)
#p_best = p_best.reshape(100,2)

plt.scatter(pos_x[:,0],pos_x[:,1])
plt.show()

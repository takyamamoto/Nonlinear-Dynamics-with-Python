# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 07:59:41 2018

@author: user
"""
#https://www.math.auckland.ac.nz/~hinke/preprints/lko_puzzle.pdf
#http://www.k.mei.titech.ac.jp/members/nakao/Etc/phasereduction-iscie.pdf

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.integrate as integrate
from tqdm import tqdm

I = 0.34 #external stimulus
a = 0.7
b = 0.8
c = 10

# 原点
u0 = 1.710
v0 = 0.374
X0 = [u0, v0]

#time
dt = 0.001
idt = 1/dt
t = np.arange(0, 40, dt)
T = 4095 #msec

def FHN(state, t):
    """
    FitzHugh-Nagumo added pulse perturbation model
    u : the membrane potential
    v : a recovery variable
    """
    u, v = state
    dudt = c * (-v + u - pow(u,3)/3 + I)
    dvdt = u - b * v + a
    return dudt, dvdt

def PhaseField(u,v):
    #軌道を計算
    Xp = integrate.odeint(FHN, [u,v], t)
    
    #軌道中の点と原点の距離を計算
    d = Xp-X0
    L2 = d[:,0]**2 + d[:,1]**2
    
    #軌道中の点で最も原点に近い最初の点のindex(=time)を取得
    tau_0 = np.argmin(L2)
    
    #周期で割って余りを出す
    tau = tau_0 % T
    
    #位相を2piで割った値
    theta_per_2pi = 1 - tau / T
    
    #位相
    #theta = theta_per_2pi * 2*np.pi
    return theta_per_2pi

ds = 0.01
U, V = np.meshgrid(np.arange(-3, 3, ds),
                   np.arange(2, -1, -ds))
ulen, vlen = U.shape
arr_PhaseField = np.zeros((ulen,vlen))

#網羅的に位相を計算
pbar = tqdm(total=ulen*vlen)
for i in range (ulen):
    for j in range(vlen):
        u = U[i,j]
        v = V[i,j]
        arr_PhaseField[i,j] = PhaseField(u,v)
        pbar.update(1)
        
pbar.close()

# ヒートマップを出力
plt.figure(figsize=(5,5))
sns.heatmap(arr_PhaseField, cmap = "hsv",
            xticklabels = False, yticklabels = False,
            cbar = False)
plt.xlabel("u : membrane potential")
plt.ylabel("v : recovery variable")
plt.title("FitzHugh-Nagumo Phase Field")
plt.savefig('FHN_PhaseField.png')
plt.show()

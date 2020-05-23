"""
Diffusion Model used in Fig.4D of Ye, X. et.al. PNAS 2019

@author: yexiang
"""

import math
import numpy as np

##Simulation Part:

M = 100 #number of simulating molecules
p_num = 6 #number of protomers per oligomer
k_df = 0.0015 #the Special Subunit Diffusion Rate (1/sec)
k_hx = 10 #the HX rate of the Special Subunit (1/sec) 
totalTime = 1000 #the total simulation time (sec) 

#calculate step time of simulation:
kmax=max(k_df, k_hx)
deltaT=min(1.0/kmax, totalTime/1000.0)

#calculate micro step probability:
p_hx = 1 - math.exp(-k_hx * deltaT) #probability of HX during 'deltaT'
p_df = 1 - math.exp(-k_df * deltaT) #probability of special subunit diffusion (either direction) during 'deltaT'

#each row is a molecule:
HDmatrix=np.zeros((M,p_num),dtype=int) #initialize HD matrix(0=H; 1=D) 
p_nonS=p_num-1
SUmatrix=np.concatenate((np.ones((M,1)), np.zeros((M,p_nonS))),axis=1) #initialize Special Unit position matrix (1=SU; 1=regular subunit)

#start simulation:
sim_time_pt=np.arange(deltaT, totalTime+deltaT, deltaT)
num_pt=len(sim_time_pt)
RandomArray = np.random.rand(M, 3, num_pt)
FractionLabeled = np.zeros((1, len(sim_time_pt)))
t=0
for time in sim_time_pt:
   
   #HX process
   for i in range(M):
       for j in range(p_num):
         if p_hx >= RandomArray[i, 0, t] and SUmatrix[i,j]==1: #HX will occur
             HDmatrix[i,j] = 1 #being labeled
        
   #Special Subunit diffusion process
   for i in range(M):
       x0=np.where(SUmatrix[i,:] == 1) #locate which subunit is current special
       if p_df >= RandomArray[i, 1, t]: #diffusion will occur
           x=math.ceil(p_num*RandomArray[i, 2, t]); #special protomer randomly diffuse
           SUmatrix[i, x0] = 0;
           SUmatrix[i, int(x)-1] = 1;
           
   FractionLabeled[0,t] = sum(sum(HDmatrix))/(M*p_num*1.0);
   t=t+1

##After-Simulation:

import matplotlib.pyplot as plt

plt.figure()

plt.semilogx(sim_time_pt, FractionLabeled[0,:], 'r')
plt.xlabel('Simulation Time (sec)')
plt.ylabel('Fraction of HX Labeled')
plt.title('Random Model Simulation. Dissamble Rate={}, HX Rate={}'.format(k_df,k_hx))
plt.grid()











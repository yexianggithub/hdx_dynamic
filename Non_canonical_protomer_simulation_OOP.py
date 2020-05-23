"""
Diffusion Model used in Fig.4D of Ye, X. et.al. PNAS 2019

@author: yexiang
"""

import math
import numpy as np

class Oligomer_Simulation:
    
    def __init__(self, num_mol,num_protomer,k_diffuse,k_HX,simu_time):
        """
        Create the oligomer with num_protomer of subunits for simulation
        """
        
        self.num_mol=num_mol
        self.num_protomer=num_protomer
        self.k_df=k_diffuse
        self.k_hx=k_HX
        self.simu_time=simu_time
        
        
    def cal_probability(self,rate):
        """
        #calculate micro step probability with rate
        """
        return 1 - math.exp(-rate * self.delta_time)
    
    def perform_simu(self):
        """
        Perform acutal Monte-Carlo simulation
        """
        #calculate step time of simulation:
        kmax=max(self.k_df, self.k_hx)
        self.delta_time=min(1.0/kmax, self.simu_time/1000.0)
        
        #Matrix holding results, each row is a molecule
        HDmatrix=np.zeros((self.num_mol,self.num_protomer),dtype=int) #initialize HD matrix(0=H; 1=D) 
        #initialize Special Unit position matrix (1=SU; 1=regular subunit)
        p_nonS=self.num_protomer-1
        SUmatrix=np.concatenate((np.ones((self.num_mol,1)), np.zeros((self.num_mol,p_nonS))),axis=1) 
        
        #Begin simulation:
        self.sim_time_pt=np.arange(self.delta_time, self.simu_time+self.delta_time, self.delta_time)
        num_pt=len(self.sim_time_pt)
        RandomArray = np.random.rand(self.num_mol, 3, num_pt)
        self.FractionLabeled = np.zeros((1, num_pt))
        p_hx=self.cal_probability(self.k_hx) # probability of HX occurs during self.deltaT
        p_df=self.cal_probability(self.k_df) # probability of protomer diffusion occurs during self.delta_time
        
        t=0
        for time in self.sim_time_pt:
   
            #HX process
            for i in range(self.num_mol):
                for j in range(self.num_protomer):
                    if p_hx >= RandomArray[i, 0, t] and SUmatrix[i,j]==1: #HX will occur
                        HDmatrix[i,j] = 1 #being labeled
        
            #Special Subunit diffusion process
            for i in range(self.num_mol):
                x0=np.where(SUmatrix[i,:] == 1) #locate which subunit is current special
                if p_df >= RandomArray[i, 1, t]: #diffusion will occur
                    x=math.ceil(self.num_protomer*RandomArray[i, 2, t]); #special protomer randomly diffuse
                    SUmatrix[i, x0] = 0;
                    SUmatrix[i, int(x)-1] = 1;
           
            self.FractionLabeled[0,t] = sum(sum(HDmatrix))/(self.num_mol*self.num_protomer*1.0);
            t=t+1

    def plot_results(self):
        """
        visulize simulation results
        """
        import matplotlib.pyplot as plt

        plt.figure()

        plt.semilogx(self.sim_time_pt, self.FractionLabeled[0,:], 'r')
        plt.xlabel('Simulation Time (sec)')
        plt.ylabel('Fraction of HX Labeled')
        plt.title('Random Model Simulation. Dissamble Rate={}, HX Rate={}'.format(self.k_df,self.k_hx))
        plt.grid()
        
# Simulation example
if __name__=='__main__':    
    
    simu_Obj=Oligomer_Simulation(num_mol=100,num_protomer=6,k_diffuse=0.0015,k_HX=10,simu_time=1000)
    simu_Obj.perform_simu()
    simu_Obj.plot_results()
    
        
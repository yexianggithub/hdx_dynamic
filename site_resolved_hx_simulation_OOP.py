"""
Diffusion Model used in Fig.4D of Ye, X. et.al. PNAS 2019
with HX process incorperated

@author: yexiang
"""

import Non_canonical_protomer_simulation_OOP as NC
import numpy as np
import scipy.stats as stat
import scipy.optimize as op
import math

class hx_embeded_sim(NC.Oligomer_Simulation):
    
    def  __init__(self,num_mol,num_protomer,k_diffuse,k_HX,simu_time,k_chem=[], pf=1):
        """
        Create the oligomer with num_protomer of subunits for simulation
        and with HX occur to each site at rates stored in k_chem
        k_chem is a list of HX rate with protection factor = 1
        Pro has a HX rate of zero
        pf is the factor by which the HX rate at the protected state is slowed 
        """
        NC.Oligomer_Simulation.__init__(self,num_mol,num_protomer,k_diffuse,k_HX,simu_time)
        self.k_chem=k_chem
        self.hx_sites=len(k_chem)
        self.pf=pf
        self.fit_flag=0
        self.timeseries=[]
        self.c13=[]
        
    def perform_simu(self):
        """
        Perform acutal Monte-Carlo simulation
        with HX occurs to individual sites
        """
        
        if not self.k_chem:  # if no site specific HX rate provided, just perform simulation without them
            NC.Oligomer_Simulation.perform_simu(self)
            return
        
        #calculate step time of simulation:
        kmax=max(self.k_df, max(self.k_chem))
        self.delta_time=min(1.0/kmax, self.simu_time/1000.0)
        
        #initialize Special Unit position matrix (1=SU; 1=regular subunit)
        p_nonS=self.num_protomer-1
        SUmatrix=np.concatenate((np.ones((self.num_mol,1)), np.zeros((self.num_mol,p_nonS))),axis=1) 
        
        #Begin simulation:
        self.sim_time_pt=np.arange(self.delta_time, self.simu_time+self.delta_time, self.delta_time)
        num_pt=len(self.sim_time_pt)
        RandomArray = np.random.rand(self.num_mol, 2, num_pt)
        p_df=self.cal_probability(self.k_df) # probability of protomer diffusion occurs during self.delta_time
        
        #Matrix holding results, each row is a molecule
        HDmatrix=np.zeros((self.num_mol,self.num_protomer,self.hx_sites),dtype=int) #initialize HD matrix(0=H; 1=D) 
        self.FractionLabeled = np.zeros((self.hx_sites, num_pt))
        # probability of HX occurs during self.deltaT
        p_hx_1=[self.cal_probability(self.k_chem[i]) for i in range(self.hx_sites)] # HX with pf=1
        p_hx_pf=[self.cal_probability(self.k_chem[i]/float(self.pf)) for i in range(self.hx_sites)] # HX with pf=self.pf
        
        t=0
        for time in self.sim_time_pt:
            
            HX_random=np.random.rand(self.num_mol,self.hx_sites) #decide HX process occur or not
   
            #HX process
            for i in range(self.num_mol):
                for j in range(self.num_protomer):
                    for l in range(self.hx_sites):
                        if SUmatrix[i,j]==1 and p_hx_1[l] >= HX_random[i, l]: #HX will occur with pf=1
                            HDmatrix[i,j,l] = 1 #being labeled
                        elif SUmatrix[i,j]==0 and p_hx_pf[l] >= HX_random[i, l]: #HX will occur with pf=self.pf
                            HDmatrix[i,j,l] = 1
                            
            #Special Subunit diffusion process
            for i in range(self.num_mol):
                x0=np.where(SUmatrix[i,:] == 1) #locate which subunit is current special
                if p_df >= RandomArray[i, 0, t]: #diffusion will occur
                    x=math.ceil(self.num_protomer*RandomArray[i, 1, t]); #special protomer randomly diffuse
                    SUmatrix[i, x0] = 0;
                    SUmatrix[i, int(x)-1] = 1;
           
            for site in range(self.hx_sites):
                self.FractionLabeled[site,t] = sum(sum(HDmatrix[:,:,site]))/(self.num_mol*self.num_protomer*1.0);
            t=t+1
             
    
    def generate_isotopic_dis(self,c13=[],num_pops=2,fit_flag=0,timeseries=[1,3,10,30,100]):
        """
        Generate isotopic envelop for ploting
        c13 is a list of carbon 13 peaks that are going to be convoluted with
        deuteron distribution as a result of simulation
        """
        self.fit_flag=fit_flag
        self.num_pops=num_pops
        if c13:
            self.c13=np.array(c13)/(sum(c13)*1.0)  # normalize c13 distribution for convolution
        
        # deal with output timeseries first
        if timeseries:
            # trim the input timeseries to make sure it fall within the simulation time range
            timeseries=[time for time in timeseries if time>self.delta_time and time<self.simu_time]
        else:
            curr_time=self.delta_time
            for i in range(9): # output nine timepoints of HX Mass spec isotope plots
                if curr_time<=self.simu_time:
                    timeseries.append(curr_time)
                    curr_time=curr_time**2
        self.timeseries=timeseries

        # find the closest timepoints that are in self.sim_time_pt
        timepoint_indx=[]
        for time in timeseries:
            diff=self.sim_time_pt-time
            pos=np.argmin(np.absolute(diff))
            timepoint_indx.append(pos)
        
        self.fit_paras=np.zeros((len(timepoint_indx),num_pops*2))
        self.distributions=np.zeros((len(timepoint_indx),self.hx_sites+len(c13)))
        timepoint_count=0
        
        for tp_idx in timepoint_indx:
            curr_tp=self.FractionLabeled[:,tp_idx]
            curr_dis=np.array([1])
            
            for site in range(self.hx_sites):
                curr_dis=np.convolve([1-curr_tp[site],curr_tp[site]],curr_dis)
            
            if self.fit_flag:
                curr_fit_paras=fit_binomial_dis(curr_dis,num_pops)
                self.fit_paras[timepoint_count,:]=curr_fit_paras
            
            if self.c13:
                curr_dis=np.convolve(self.c13,curr_dis)
            self.distributions[timepoint_count,:]=curr_dis
            timepoint_count+=1
            
        
    def plot_results(self):
        """
        visulize simulation results
        by plotting the mass spec spectrum of the isotopic envelope of the peptide in question
        """
        import matplotlib.pyplot as plt
        
        num_subplots=len(self.timeseries)
        x=np.array([idx for idx in range(np.shape(self.distributions)[1])]) # x axis index for isotopic peaks

        fig, axs = plt.subplots(num_subplots,1,sharex='col')
        
        for i in range(num_subplots):
            curr_dis=self.distributions[i,:]
            axs[i].vlines(x,0,curr_dis,colors='k')
            bottom, top = plt.ylim()
            
            if self.fit_flag:
                curr_trace=np.zeros((self.num_pops+1,len(curr_dis))) #the first num_pops rows are individual population
                # the last row is the sum of all populations
                for pop in range(self.num_pops):
                    p, frac = self.fit_paras[i][2*pop:2*pop+1]
                    hd_envelop=[stat.binom.pmf(k, self.hx_sites, p) for k in range(self.hx_sites+1)]
                    
                    if self.c13: # convolute with c13 distribution
                        hd_envelop=np.convolve(self.c13,hd_envelop)
                    curr_trace[pop,:]=hd_envelop*frac
                    axs[i].plot(x,curr_trace[pop,:],'b--')
                    text_x=x[-2] # text label x index
                    text_y=top-(top-bottom)*pop*0.1   # text label y index
                    axs[i].text(text_x,text_y,'{}, {}'.format(p*self.hx_sites,frac),fontsize=5)
                
                curr_trace[pop+1,:]=np.sum(curr_trace[0:-1,:],axis=1)
                axs[i].plot(x,curr_trace[pop+1,:],'r-')
                
            else:
                axs[i].plot(x,curr_dis,'r-')
                
        plt.show()
        

# helper functions to perform binomial fitting of the distribution
def estimate_initial_value(distribution,num_pops=2):
    """
    a function used to estimate the initial guess for binom fitting to distribution
    """
    estimate=[]
    fraction=1.0/num_pops
    num_sites=len(distribution)
    non_zero=len(distribution[distribution>0.01])
    center_to_center_distance=non_zero/(1.0*num_pops)
    curr_center=center_to_center_distance/2.0
    for pop in range(num_pops):
        p=curr_center/num_sites
        estimate.extend([p,fraction])
        curr_center+=center_to_center_distance
    
    
def binom_resi(x0,distribution,num_pops=2):
    """
    A function to calculate residual between the actual distribution and a
    theoretical binomial distribution with parameter x0 takes the form [p1,f1,p2,f2,...]
    in which, p1 and p2 stand for the probability of the first and second population
    p1<p2, and f1 and f2 stand for the fraction of the two populations with f1+f2=1
    """
    num_site=len(distribution)-1
    dist_binom=np.zeros((1,num_site+1))
    
    for count in range(num_pops):
        p,f = x0[2*count:2*count+1]
        curr_dis=[stat.binom.pmf(k, num_site, p) for k in range(num_site+1)]
        curr_dis=np.array(curr_dis)
        dist_binom=dist_binom+curr_dis*f
    
    return distribution-dist_binom

def fit_binomial_dis(distribution,num_pops=2):
    """
    helper function to find the binomial distribution centers of more than two populations
    """
    x0=estimate_initial_value(distribution)
    result=op.least_squares(binom_resi,x0,distribution,num_pops)
    return result.x
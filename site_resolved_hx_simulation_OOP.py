"""
Diffusion Model used in Fig.4D of Ye, X. et.al. PNAS 2019
with HX process incorperated
"""

import Non_canonical_protomer_simulation_OOP as NC
import numpy as np
import scipy.stats as stat
import scipy.optimize as op
import math

class Mass_Spec_Plot():
    """
    object use to perform plotting mass spec isotopic peak distributions
    used for both the HX_Embeded_Sim object and the pep_tree objects
    """
    
    def __init__ (self):
        """
        """
        self.timeseries=[]
        self.distributions=[]
        self.hx_sites=0
        self.fit_flag=0
        self.c13=[]
        self.num_pops=2
        
    def plot_MSresults(self):
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
                    p, frac = self.fit_paras[i][2*pop:2*pop+2]
                    hd_envelop=[stat.binom.pmf(k, self.hx_sites, p) for k in range(self.hx_sites+1)]
                    hd_envelop=np.array(hd_envelop)
                    
                    if len(self.c13): # convolute with c13 distribution
                        hd_envelop=np.convolve(self.c13,hd_envelop)
                    curr_trace[pop,:]=hd_envelop*frac
                    
                    if frac: # if this pop is not empty
                        axs[i].plot(x,curr_trace[pop,:],'b--')
                        text_x=x[-2] # text label x index
                        text_y=top-(top-bottom)*pop*0.1   # text label y index
                        axs[i].text(text_x,text_y,'{}, {}'.format(p*self.hx_sites,frac),fontsize=5)
                
                curr_trace[pop+1,:]=np.sum(curr_trace[0:-1,:],axis=0)
                axs[i].plot(x,curr_trace[pop+1,:],'r-')
                
            else:
                axs[i].plot(x,curr_dis,'r-')
                
        plt.show()
        
        
class HX_Embeded_Sim(NC.Oligomer_Simulation, Mass_Spec_Plot):
    
    def  __init__(self,num_mol,num_protomer,k_diffuse,k_HX,simu_time,k_chem=[], pf=1):
        """
        Create the oligomer with num_protomer of subunits for simulation
        and with HX occur to each site at rates stored in k_chem
        k_chem is a list of HX rate with protection factor = 1
        Pro has a HX rate of zero
        pf is the factor by which the HX rate at the protected state is slowed 
        """
        NC.Oligomer_Simulation.__init__(self,num_mol,num_protomer,k_diffuse,k_HX,simu_time)
        Mass_Spec_Plot.__init__(self)
        self.k_chem=k_chem[:]
        self.hx_sites=len(k_chem)
        self.pf=pf
        
        
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
        self.HD_distr = np.zeros((num_pt, self.hx_sites+1))
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
                    x=math.ceil(self.num_protomer*RandomArray[i, 1, t]) #special protomer randomly diffuse
                    SUmatrix[i, x0] = 0;
                    SUmatrix[i, int(x)-1] = 1;
           
            for mol in range(self.num_mol): # compile the H-D distribution per molelcule per protomer
                for protomer in range(self.num_protomer):
                    num_D=sum(HDmatrix[mol,protomer,:])
                    self.HD_distr[t,num_D] = self.HD_distr[t,num_D] + 1
            
            # normalize the distribution
            self.HD_distr[t,:]=self.HD_distr[t,:]/(self.num_mol*self.num_protomer)
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
        if len(self.c13):
            self.distributions=np.zeros((len(timepoint_indx),self.hx_sites+len(c13)))
        else:
            self.distributions=self.HD_distr[timepoint_indx,:]
        timepoint_count=0
        
        # convert the monte-carlo simulation results into isotopic distribution
        for tp_idx in timepoint_indx:
            curr_dis=self.HD_distr[tp_idx, :]
            
            """
            # wrong way to generate H D distribution by assuming each HX site exchange independently
            for site in range(self.hx_sites):
                curr_dis=np.convolve([1-curr_tp[site],curr_tp[site]],curr_dis)
            """
            if self.fit_flag:
                curr_fit_paras=fit_binomial_dis(curr_dis,num_pops)
                self.fit_paras[timepoint_count,0:len(curr_fit_paras)]=curr_fit_paras
            
            if len(self.c13):
                curr_dis=np.convolve(self.c13,curr_dis)
                self.distributions[timepoint_count,:]=curr_dis
            timepoint_count+=1
        
##################################################################
# helper functions to perform binomial fitting of the distribution
##################################################################

def estimate_initial_value(distribution,num_pops):
    """
    a function used to estimate the initial guess for binom fitting to distribution
    of num_pops of populations, i.e.
    sum(frac_i*B(n,p_i)), i=1,2,..,num_pops
    """
    estimate=[] 
    low_bounds=[]
    upper_bounds=[]
    fraction=1.0/num_pops  # assuming equal fraction frac_i
    num_sites=len(distribution)  # n
    
    non_zero=len(distribution[distribution>0.01])
    center_to_center_distance=non_zero/(1.0*num_pops)
    curr_center=center_to_center_distance/2.0 # estimate the distances between populations
    for pop in range(num_pops):
        p=curr_center/num_sites
        if pop==num_pops-1:
            estimate.append(p)
            low_bounds.append(0)
            upper_bounds.append(1)
        else:
            estimate.extend([p,fraction])
            low_bounds.extend([0,0])
            upper_bounds.extend([1,1])
        curr_center+=center_to_center_distance
    return estimate, (low_bounds, upper_bounds)
    
    
def binom_resi(x0,distribution,num_pops):
    """
    A function to calculate residual between the actual distribution and a
    theoretical binomial distribution with parameter x0 takes the form [p1,f1,p2,f2,...]
    in which, p1 and p2 stand for the probability of the first and second population
    p1<p2, and f1 and f2 stand for the fraction of the two populations with f1+f2=1
    """
    num_site=len(distribution)-1
    dist_binom=np.zeros(num_site+1)
    running_sum_f=0
    
    for count in range(num_pops):
        # make sure fraction of all pops add up to 1
        if count==num_pops-1:
            p=x0[-1]
            f=1-running_sum_f
            if f<0: # make sure no population with negative fraction
                f=0
        else:
            p,f = x0[2*count:2*count+2]
            running_sum_f+=f
        curr_dis=[stat.binom.pmf(k, num_site, p) for k in range(num_site+1)]
        curr_dis=np.array(curr_dis)
        dist_binom=dist_binom+curr_dis*f
    
    res=distribution-dist_binom
    return res


def AIC(num_pops,residuals):
    """
    This function implement Akaike's Information Criterion (AIC) to determine
    over-fitting. AIC=n*ln(SSE)-n*ln(n)+2*p, 
    in which n is the number of data points, SSE is sum of squared error, and p
    is the number of fitting parameters
    """
    p=num_pops*2-1
    n=len(residuals)
    sse=sum([x**2 for x in residuals])
    aic=n*math.log(sse)-n*math.log(n)+2*p
    return aic
    
def fit_binomial_dis(distribution,num_pops=2):
    """
    helper function to find the binomial distribution centers of more than two populations
    """
    x0,fit_bounds=estimate_initial_value(distribution,num_pops)
    result=op.least_squares(binom_resi,x0,args=(distribution,num_pops),bounds=fit_bounds)
    
    # use information based criteria to determine whether over fitting the data
    aic_curr=AIC(num_pops,result.fun)
    aic_best=aic_curr
    num_pops-=1
    
    while num_pops>=1:
        x0,fit_bounds=estimate_initial_value(distribution,num_pops)
        result_curr=op.least_squares(binom_resi,x0,args=(distribution,num_pops),bounds=fit_bounds)
        aic_curr=AIC(num_pops,result_curr.fun)
        
        if aic_curr<=aic_best:
            result=result_curr
            aic_best=aic_curr
        
        num_pops-=1
        
    results=result.x
    
    # replace the fraction of the last pop with 1-sum(all other fractions)
    running_sum_f=0
    for count in range((len(results)-1)/2):
        # make sure fraction of all pops add up to 1
        p,f = results[2*count:2*count+2]
        running_sum_f+=f

    return np.append(results,[1-running_sum_f])

if __name__=="__main__":
    """
    perform the simulation and plot the resulted MS spec
    """
    import random
    
    num_mol=100
    num_protomer=6
    k_diffuse=0.001
    k_HX=10
    simu_time=500
    k_chem=[1*random.random(),5*random.random(),5*random.random(),10*random.random(),\
            10*random.random(),50*random.random(),100*random.random()]
    protection_factor=500
    # assuming five carbon per amino acid and 1% C13 abundance
    c13_peaks=[stat.binom.pmf(k, len(k_chem), 0.01) for k in range(4)]
    #c13_peaks=[]
    number_fit_population=2
    to_fit_with_binomial=1
    time_points_to_plot=[1,3,10,30,100,300]
    
    hx_simulation=HX_Embeded_Sim(num_mol,num_protomer,k_diffuse,k_HX,simu_time,k_chem=k_chem, pf=protection_factor)
    hx_simulation.perform_simu()
    hx_simulation.generate_isotopic_dis(c13=c13_peaks,num_pops=number_fit_population,\
                                        fit_flag=to_fit_with_binomial,timeseries=time_points_to_plot)
    hx_simulation.plot_MSresults()
    
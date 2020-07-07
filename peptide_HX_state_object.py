"""
modeling hx by peptide level macroscopic state and ordinary differential equations
used in Table 1 and Fig S3 of Ye, X. et.al. PNAS 2020
"""

import numpy as np
import scipy.stats as stat
from site_resolved_hx_simulation_OOP import estimate_initial_value, binom_resi, fit_binomial_dis, Mass_Spec_Plot

class pep_node():
    """
    model peptide level H/D microscopic state in the form of e.g. '100100' 1-D, 0-H
    no proline is modeled here b/c its HX rate equals zero
    """
    
    def __init__(self,hd_state,concentration=0,pf=1):
        """
        hd_state is the identifier of this node of the form '100100a' or '100100b'
            in which 'a' or 'b' stands for the protected/unprotected state 
        down_stream_adjacent is the adjacent dictionary connecting to exchange 
            one H with D on a particular site with a HX rate
        up_stream_adjacent is the adjacent dictionary as a result of back exchange, 
            i.e. exchange D with H
        concentration is the concentration of peptides in this state at a given time
        pf protect factor which scale down exchange rate
        """
        
        self.hd_state=hd_state  # include the appendix letter
        # site-specific hd status in numpy array
        self.hd_list=np.array([int(site) for site in self.hd_state[:-1]]) 
        self.down_stream_adjacent={}
        self.up_stream_adjacent={}
        self.concentration=concentration
        self.pf=pf
        
    def add_down_stream_node(self,add_node,curr_k_chem):
        """
        add entry to the downstream adjacent dictionary
        node is a pep_node class object
        curr_k_chem is the corresponding site-specific HX rate
        """
        self.down_stream_adjacent[add_node]=curr_k_chem/self.pf
        
    def add_up_stream_node(self,add_node,curr_k_chem,back_exchange_factor=1000):
        """
        add entry to the downstream adjacent dictionary
        node is a pep_node class object
        """
        self.up_stream_adjacent[add_node]=curr_k_chem/back_exchange_factor
        
    def add_a_b_transition(self):
        """
        connect peptide of the same H-D profile but in different states (a or b)
        transition between a and b follows rates self.k_ab and self.k_ba
        """
        if self.hd_state[-1] == 'a':
            self.a_b_complmt=self.hd_state[:-1]+'b'
        else:
            self.a_b_complmt=self.hd_state[:-1]+'a'
        
    def set_concentration(self,new_conc):
        """
        set the concentration of the peptide represented by this node
        """
        self.concentration=new_conc
        
    def cal_num_deuteron(self):
        """
        calculate the number of D carried by this peptide
        """
        return sum(self.hd_list)
            
        
class pep_tree(Mass_Spec_Plot):
    """
    model all pep H/D microscopic states
    """
    
    def __init__(self,k_chem,pf_protected_state,k_ab,k_ba):
        """
        sites is the number of exchangable sites
        k_chem is a list of unprotected HX rates
        len(k_chem)=sites
        self.nodes is a list of pep_node objects that belongs to this tree
        k_ab conversion rate from state a to state b
        """
        Mass_Spec_Plot.__init__(self)
        self.k_chem=k_chem[:]
        self.sites=len(self.k_chem)
        self.hx_sites=self.sites
        self.nodes={}
        self.pf_protected=pf_protected_state
        self.k_ab=float(k_ab) # make sure that in the right numerical type
        self.k_ba=float(k_ba)
        
    def populate_tree(self):
        """
        populate the tree with nodes of class pep_node
        """
        # generate the permuation of all possible H/D profile of the peptide
        # 'a' and 'b' are peptide under two different conditions with different pf
        # 'a' is less protected than 'b'
        base=['a','b']
        for site in range(self.sites):
            new_elements=[]
            for element in base:
                new_elements.append('0'+element)
                new_elements.append('1'+element)
            base=[]
            base.extend(new_elements)
        
        # add nodes to the three        
        for string in base:
            if string[-1]=='b': # peptide in unprotected state
                self.nodes[string]=pep_node(string)
            else: # peptide in protected state
                self.nodes[string]=pep_node(string,pf=self.pf_protected)
            self.nodes[string].add_a_b_transition() # add a_b_complmt for this node
            
    def connect_nodes_hx(self):
        """
        Establishing up_stream(back exchange) and down_stream (forward H to D exchange)
        connection between nodes
        """
        for node_name in self.nodes.keys():
            node=self.nodes[node_name]
            hx_list=np.copy(node.hd_list)  # need to copy the hd_list to avoid accidentally change its value
            #node.add_a_b_transition  # add a-b complement of this node
            
            if sum(hx_list) == self.sites: # allD state only back exchange 
                continue
            
            for site in range(self.sites):
            
                curr_down_stream_hx_list=np.copy(hx_list) # need to copy the hd_list to avoid accidentally change its value
                if curr_down_stream_hx_list[site]==0:
                    curr_down_stream_hx_list[site]=1
                    curr_down_stream_hx_state=''.join(map(str,curr_down_stream_hx_list))
                    curr_down_stream_node_name=curr_down_stream_hx_state+node_name[-1]
        
                    node.add_down_stream_node(curr_down_stream_node_name,\
                                                               self.k_chem[site])
                
                    curr_down_stream_node=self.nodes[curr_down_stream_node_name]
                    curr_down_stream_node.add_up_stream_node(node_name,self.k_chem[site])
                
    def set_initial_state(self):
        """
        set the initial state assuming starting from allH at equilibrium of higher level 
        binding/association reaction
        """
        for node_name in self.nodes:
            
            node=self.nodes[node_name]
            if sum(node.hd_list) == 0:
                if node.hd_state[-1] == 'a': # set allH peptide in a (protected) state
                    node.set_concentration(self.k_ba/(self.k_ba+self.k_ab))
                else:
                    node.set_concentration(self.k_ab/(self.k_ba+self.k_ab))
            else:
                node.set_concentration(0)
                
    def cal_dy_dt(self):
        """
        the delta concentration over time of this node expressed by connecting nodes
        k_ab conversion rate from state a to state b
        """
        self.dy_dt={}
        
        # flow out of this node
        # flow out of this node due to HX to downstream nodes
        for node_name in self.nodes:
            node=self.nodes[node_name]
            curr_dy_dt=0
            sum_rates=0
            
            if node.down_stream_adjacent:
                for rate in node.down_stream_adjacent.values():
                    sum_rates+=rate
        
            # flow out of this node due to backexchange to upstream nodes
            if node.up_stream_adjacent:
                for rate in node.up_stream_adjacent.values():
                    sum_rates+=rate
            
            # flow out of this node due to higher level kinetics (e.g. binding/dissociation)
            if node.hd_state[-1] == 'a':
                sum_rates+=self.k_ab
                complmt_rate=self.k_ba
            else:
                sum_rates+=self.k_ba
                complmt_rate=self.k_ab
        
            curr_dy_dt+=sum_rates*node.concentration*(-1)
        
            #flow into this node
            # flow into this node due to HX from upstream nodes
            if node.up_stream_adjacent:
                for up_node in node.up_stream_adjacent.keys():
                    curr_node=self.nodes[up_node]
                    curr_dy_dt+=curr_node.concentration*curr_node.down_stream_adjacent[node_name]
            
            # flow into this node due to backe exchange from downstream nodes
            if node.down_stream_adjacent:
                for down_node in node.down_stream_adjacent.keys():
                    curr_node=self.nodes[down_node]
                    curr_dy_dt+=curr_node.concentration*curr_node.up_stream_adjacent[node_name]
                
            # flow into this node due to higher level kinetics (e.g. binding/dissociation)
            compl_node_name=node.a_b_complmt
            compl_node=self.nodes[compl_node_name]
            curr_dy_dt+=compl_node.concentration*complmt_rate
            
            self.dy_dt[node.hd_state]=curr_dy_dt
            
    def ode_solver(self,time_step,time_duration):
        """
        implement Euler's method to solve the ode numerically
        return a dictionary of node:[concentration(time)] and a list of time points
        """
        self.delta_time=time_step
        self.simu_time=time_duration
        
        time=0
        num_step=int(round(time_duration/time_step))
        self.t_points=np.zeros(num_step)
        conc_by_node={}   # store concentration(time) for output
        self.set_initial_state() # reset the initial concentration
        self.cal_dy_dt() # calculate the dy/dt for each node
        
        # record the inital condition into the data dictionary conc_by_node
        for node_name in self.nodes:
            node=self.nodes[node_name]
            conc_by_node[node.hd_state]=np.zeros(num_step)
            conc_by_node[node.hd_state][0]=node.concentration
        
        for step in range(num_step-1):
            time+=time_step
            self.t_points[step+1]=time
            
            for node_name in self.nodes:
                node=self.nodes[node_name]
                
                # Euler's formula y(t+1)=y(t)+dy/dt*delta(t)
                curr_conc=node.concentration+self.dy_dt[node.hd_state]*time_step 
                node.set_concentration(curr_conc) # update conc for this node at this time point
                conc_by_node[node.hd_state][step+1]=curr_conc
            
            self.cal_dy_dt() #update dy_dt for next time point
                
        self.conc_by_node=conc_by_node
            
    def generate_isotopic_dis(self,c13=[],num_pops=2,fit_flag=0,timeseries=[1,3,10,30,100]):
        """
        Generate isotopic envelop for ploting
        c13 is a list of carbon 13 peaks that are going to be convoluted with
        deuteron distribution as a result of running ode solver
        """
        self.fit_flag=fit_flag
        self.num_pops=num_pops
        if len(c13):
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
            diff=self.t_points-time
            pos=np.argmin(np.absolute(diff))
            timepoint_indx.append(pos)
        
        self.fit_paras=np.zeros((len(timepoint_indx),num_pops*2))
        self.distributions=np.zeros((len(timepoint_indx),self.sites+len(c13)))
        timepoint_count=0
        
        # convert the ode results into isotopic distribution
        for tp_idx in timepoint_indx:

            curr_dis=np.zeros(self.sites+1)
            
            # compile the distribution of the isotopic peaks
            for node_name, conc_node in self.conc_by_node.items():
                num_D=self.nodes[node_name].cal_num_deuteron()
                curr_dis[num_D]+=conc_node[tp_idx]
            
            # if needed fit with binomial distribution model
            if self.fit_flag:
                curr_fit_paras=fit_binomial_dis(curr_dis,num_pops)
                self.fit_paras[timepoint_count,0:len(curr_fit_paras)]=curr_fit_paras
            
            # if needed convolute with c13 distribution
            if len(self.c13):
                curr_dis=np.convolve(self.c13,curr_dis)
            
            self.distributions[timepoint_count,:]=curr_dis
            timepoint_count+=1
            
          
if __name__=="__main__":
    """
    perform the simulation and plot the resulted MS spec
    """
    import random
    
    simu_time=1000
    k_chem=[1*random.random(),5*random.random(),5*random.random(),10*random.random(),\
          10*random.random(),50*random.random(),100*random.random()]
    slow_factor=10.0 # to account for the slower than expected exchange in the less protected b state
    k_chem=[rate/slow_factor for rate in k_chem]
    #k_chem=[1*random.random(),10*random.random(),100*random.random()]
    protection_factor=500
    k_ab=50
    k_ba=1
    time_step=1.0/max([max(k_chem),k_ab,k_ba])
    # assuming five carbon per amino acid and 1% C13 abundance
    c13_peaks=[stat.binom.pmf(k, len(k_chem), 0.01) for k in range(4)]
    c13_peaks=[peak for peak in c13_peaks if peak>=0.01] # trim off the c13 peaks of extremely low abundance 
    number_fit_population=2
    to_fit_with_binomial=1
    time_points_to_plot=[1,3,10,30,100,300]
    
    hx_sim=pep_tree(k_chem,protection_factor,k_ab,k_ba)
    hx_sim.populate_tree()
    hx_sim.connect_nodes_hx()
    hx_sim.set_initial_state()
    #hx_sim.cal_dy_dt()
    hx_sim.ode_solver(time_step,simu_time)
    hx_sim.generate_isotopic_dis(c13=c13_peaks,num_pops=number_fit_population,\
                                        fit_flag=to_fit_with_binomial,timeseries=time_points_to_plot)
    hx_sim.plot_MSresults()    
"""
This script is for the paper: mobile power source 
"""

import pandas as pd
import numpy as np
import itertools
import re
import gurobipy as gp
from gurobipy import GRB
import os
# import gurobipy_pandas as gppd

########## define helper functions
def return_omega_index(node, q_thresh):
    """
        take a set of scenario values of ens;
        return index for sets Omega_0, Omega_down and Omega_up
    """
    ind_0 = np.where(node == 0)[0] + 1
    node_wo_0 = node[node != 0]
    cut = node_wo_0.quantile(q_thresh)
    ind_up = np.where((node <= cut) & (node > 0))[0] + 1
    ind_down = np.where(node > cut)[0] + 1
    return (ind_0, ind_up, ind_down)

def is_joint_sufficient(recomb, sample, p):
    """
        Check if a multi-dimensional recomb is p-sufficient for a sample
    """    
    conf_mtrx = sample <= recomb
    suffi_ind = conf_mtrx.mean(axis = 1)
    suffi_ind_conf = suffi_ind == 1
    cum_prob = suffi_ind_conf.mean()
    if cum_prob >= p:
        return True
    else:
        return False
    
def return_p_sufficient_ind(zero_ind, up_ind, down_ind, sample, 
                            phi_max, gamma_max, c, alpha):
    """
     return scenarios which chould be p-sufficient under distorted probability.
 
    """
    C_B = {}
    for col in sample:
        alpha_j = alpha[col]
        p_0 = 1 / sample.shape[0] # p_0 does not change
        p_up = 1 / sample.shape[0] * (1 + c[col] * gamma_max * phi_max)
        zero_cnt = len(zero_ind[col])
        if zero_cnt == sample.shape[0]:
            print('node ' + str(col) + ': all scenarios are 0')
        else:    
            up_cnt = len(up_ind[col])
            down_cnt = len(down_ind[col])
            if 1 - p_0 * zero_cnt - p_up * up_cnt < 0:
                print(f'node {col} has negative distorted prob')
                p_up = (1 - p_0 * zero_cnt) / up_cnt
            p_down = (1 - p_0 * zero_cnt - p_up * up_cnt) / down_cnt
            new_prob_dist = sample[col].to_frame().copy()
            new_prob_dist['prob'] = p_0
            new_prob_dist.loc[up_ind[col], 'prob'] = p_up
            new_prob_dist.loc[down_ind[col], 'prob'] = p_down
            new_prob_dist.sort_values(col, inplace = True)
            new_prob_dist['cum_prob'] = new_prob_dist['prob'].cumsum()
            lowest_p_suffi = np.min(new_prob_dist.loc[new_prob_dist.cum_prob >= alpha_j, col].values)
            C_B[col] = new_prob_dist.loc[new_prob_dist[col] >= lowest_p_suffi, col].values
            
    return C_B     

def return_cum_p_d(scen_xi, Omega_up_ind, Omega_down_ind, C_B, tau, epsilon_ind):
    """
        return p sufficient realization with cumulative disorted probability
        returned object is indexed by node index j
    """
    
    C_B_prod = {}
    for j in C_B.keys():
        xi_j = scen_xi[j].to_frame().copy()
        p_d_0 = 1 / xi_j.shape[0]
        p_d_up = 1 / xi_j.shape[0] * (1 + c[j] * sum(n * tau[j, n] for n in epsilon_ind))
        p_d_down = (1 - p_d_0 * len(Omega_0_ind[j]) - p_d_up * len(Omega_up_ind[j])) / len(Omega_down_ind[j])
        xi_j['prob_d'] = p_d_0
        xi_j.loc[Omega_up_ind[j], 'prob_d'] = p_d_up 
        xi_j.loc[Omega_down_ind[j], 'prob_d'] = p_d_down
        xi_j['is_suffi'] = 0
        least_p_suffi = C_B[j].min()
        xi_j.loc[xi_j[j] >= least_p_suffi, 'is_suffi'] = 1
        cum_p_insuffi = xi_j.loc[xi_j['is_suffi'] == 0, 'prob_d'].sum()
        prob_d = xi_j.groupby(j)['prob_d'].first()
        C_B_j = pd.DataFrame(C_B[j], index = np.arange(1, len(C_B[j]) + 1))
        C_B_j.columns = ['scen']
        C_B_j_prod_d = pd.merge(C_B_j, prob_d, left_on = 'scen', right_index = True)
        
        # calculate cumulative distorted probability for each scenario
        C_B_j_prod_d_by_scen = C_B_j_prod_d.groupby('scen')['prob_d'].sum().to_frame() # probability by scenario; 
        C_B_j_prod_d_by_scen['cum_prob_d'] = C_B_j_prod_d_by_scen['prob_d'].cumsum() + cum_p_insuffi
        C_B_j_prod_d = pd.merge(C_B_j, C_B_j_prod_d_by_scen, how = 'inner', left_on = 'scen', right_index = True)
        C_B_prod[j] = C_B_j_prod_d[['scen', 'cum_prob_d']]
        
    return C_B_prod

########## import data

# data_folder = r'D:\Research\gw_ddu_mps'
data_folder = os.path.dirname(os.getcwd())
scen_xi = pd.read_csv(os.path.join(data_folder, "data", "xi_info_v6.csv"))
scen_eta = pd.read_csv(os.path.join(data_folder, "data", "eta_info_v6.csv"))
scen_sub = pd.read_csv(os.path.join(data_folder, "data", "SubNet_Info_v6.csv"))
other_params = pd.read_csv(os.path.join(data_folder, "data", "Deter_param_v2_formatted.csv"))

other_params.isnull().sum(axis = 0)
other_params.index += 1

########## re-sample scenario for computational efficiency test
random_state = [2010, 2020, 2030, 2040, 2050]
frac = 0.05

for rnd in random_state:
    print('==========================================')
    print(f'result for random state {rnd}')
    print('==========================================')
    
    # scen_xi_sampled = scen_xi.copy()
    # scen_eta_sampled = scen_eta.copy()
    
    scen_xi_sampled = scen_xi.sample(frac = frac, random_state = rnd, ignore_index = True, replace = True)
    scen_eta_sampled = scen_eta.sample(frac = frac, random_state = rnd, ignore_index = True, replace = True)
    
    if (scen_xi_sampled.index != scen_eta_sampled.index).any():
        print('sample index for node and line are not the same...')
    
    scen_xi_sampled.isnull().sum(axis = 0)
    scen_eta_sampled.isnull().sum(axis = 0)
    
    scen_xi_sampled.index += 1
    scen_eta_sampled.index += 1
    
    num_nodes = 33
    num_lines = 32
    num_scen = scen_eta_sampled.shape[0]
    
    scen_xi_sampled.columns = np.arange(1, num_nodes + 1)
    xi = {(scen, node): scen_xi_sampled.loc[scen, node] for scen, node in itertools.product(scen_xi_sampled.index, scen_xi_sampled.columns)}
    
    
    for c in scen_xi_sampled.columns:
        print('----', c, '----')
        print(scen_xi_sampled[c].value_counts(normalize = True).sort_index())
    
    
    
    
    scen_eta_sampled.columns = [int(re.findall('[0-9]+$', col)[0]) for col in scen_eta_sampled.columns]
    eta = scen_eta_sampled * -1
    eta = eta[sorted(pd.concat([eta] * 2, axis = 1).columns)]
    
# =============================================================================
#     scen_sub.columns = [int(re.findall('[0-9]+$', col)[0]) for col in scen_sub.columns]
#     xi_sub = {(scen, sub_num): scen_sub.loc[scen, sub_num] for scen, sub_num in itertools.product(scen_sub.index, scen_sub.columns)}
#     
# =============================================================================
    
# transform from dataframe to dict
    for col in other_params.columns:
        globals()[col] = other_params.loc[~other_params[col].isnull(), col].to_dict()
    
    I = np.arange(1, num_nodes + 1)
    L = np.arange(1, num_lines + 1)
    I_c = np.array([3, 6, 11, 16, 20, 25, 28, 32]) # candidate nodes
    L_hat = np.array([2, 5, 9, 14, 18, 22, 25, 30]) # vulnerable lines
    I_i = {3: [3, 4, 5],
           6: [6, 7, 8, 9],
           11: [10, 11, 12, 13, 14],
           16: [15, 16, 17, 18],
           20: [19, 20, 21, 22],
           25: [23, 24, 25],
           28: [26, 27, 28, 29, 30],
           32: [31, 32, 33]
           }
    I_mps = list(itertools.chain(*I_i.values()))
    
    sub_network = {1: [3, 4, 5],
           2: [6, 7, 8, 9],
           3: [10, 11, 12, 13, 14],
           4: [15, 16, 17, 18],
           5: [19, 20, 21, 22],
           6: [23, 24, 25],
           7: [26, 27, 28, 29, 30],
           8: [31, 32, 33]
           }
    
    ###### define parameters
    ### parameters for mobile power sources
    num_mps = 6
    b = 3 # maximal MPS pre-disposed at rural locations
    # N_ind = np.arange(1, num_mps + b + 1) # index for ???
    mps_cap_a = 500 # active power capacity for mps
    mps_cap_r = 500 # reactive power capacity for mps
    
    M = np.arange(1, num_mps + 1) # set for MPS
    Psi_a = pd.Series(mps_cap_a * np.ones(num_mps), index = np.arange(1, num_mps + 1))
    Psi_r = pd.Series(mps_cap_r * np.ones(num_mps), index = np.arange(1, num_mps + 1))
    
    
    ### parameters for nodes
    D_a = other_params['D_a']
    D_r = other_params['D_r']
    V_under = other_params['V_under']
    V_over = other_params['V_over']
    tan_theta_up = other_params['tan_theta_up']
    tan_theta_low = other_params['tan_theta_low']
    alpha = pd.Series(0.9 * np.ones(num_nodes), index = np.arange(1, num_nodes + 1))
    phi_ub = b + num_mps
    phi_lb = 0
    gamma_ub = 1
    gamma_lb = 0
    h_ub = 1
    h_lb = 0
    c = pd.Series(0.3 * np.ones(num_nodes), index = np.arange(1, num_nodes + 1)) # 1： 0.1; 
    p_b = pd.Series(np.ones(num_scen) / num_scen, index = np.arange(1, num_scen + 1)) # baseline probability for each scenario
    
    # create indicator if candidate node and node are connected
    connect_key = [key for key in itertools.product(I_c, I)]
    connect_ind = {(i, j): 0 for i, j in connect_key}
    for i in I_c:
        for j in I:
            if j in I_i[i]:
                connect_ind[i, j] = 1
    
    ### parameters for lines
    Gamma = other_params['Gamma']
    Lambda = other_params['Lambda']
    R = other_params['R']
    X = other_params['L']
    F_a = other_params['F_a']
    F_r = other_params['F_r']
    
    ### parameters for decision-dependent functions
    low_quant = 0.2 # threashold for the set Omega_down
    Omega_ind = {col: return_omega_index(scen_xi_sampled[col], low_quant) for col in scen_xi_sampled.columns} # return scenario index
    
    # creating sets: Omega_up, Omega_down
    Omega_0_ind = {j: Omega_ind[j][0] for j in Omega_ind.keys()}
    Omega_up_ind = {j: Omega_ind[j][1] for j in Omega_ind.keys()}
    Omega_down_ind = {j: Omega_ind[j][2] for j in Omega_ind.keys()}
    
    # calculate maximal "post-decision" probability
    phi_max = 2 * b + num_mps - b # maximal phi when every mps connects to j with b being from rural and M-b from urban
    gamma_max = 1
    C_B = return_p_sufficient_ind(zero_ind = Omega_0_ind, up_ind = Omega_up_ind, 
                                  down_ind = Omega_down_ind, sample = scen_xi_sampled, 
                                  phi_max = phi_max, gamma_max = gamma_max, 
                                  c = c, alpha = alpha)
    C_B = {key: pd.Series(sorted(value, reverse = True), index = np.arange(1, len(value) + 1)) for key, value in C_B.items()}
    n_B = {key: len(value) for key, value in C_B.items()}
    N_B = {key: np.arange(1, len(value) + 1) for key, value in C_B.items()}
    z_B_ind = [(j, k) for j in I_mps for k in N_B[j]]
    A_B = {(j, k): C_B[j][k] - C_B[j][k + 1] if k < N_B[j][-1] else C_B[j][k] for j in I_mps for k in N_B[j]}
    
    
    ### parameters for boolean reformulation
    # parameters for decision-independent
    alpha_L = 0.90
    
    eta_B = pd.concat((eta, eta), axis = 1) # large eta to include active and reactive pwer
    quant_eta_B = eta_B.quantile(alpha_L)
    mrg_suffi_eta_B_conf = eta_B[eta_B >= quant_eta_B]
    
    I_eta_B = np.arange(1, mrg_suffi_eta_B_conf.shape[1] + 1)
    cut_eta_B = {i: list(mrg_suffi_eta_B_conf.iloc[:, i - 1].dropna().unique()) for i in I_eta_B}
    recomb_eta_B = pd.Series(list(itertools.product(*list(cut_eta_B.values()))))
    g_eta_B = {ind: np.arange(1, len(cuts) + 1) for ind, cuts in cut_eta_B.items()}
    c_tilde = {(i, g): cut_eta_B[i][g - 1] for i in I_eta_B for g in g_eta_B[i]} # we use c_tilde in overleaf
    
    suffi_ind = recomb_eta_B.apply(is_joint_sufficient, sample = eta_B, p = alpha_L)
    K_eta_B_pos = list(suffi_ind[suffi_ind == True].index + 1) 
    K_eta_B_neg = list(suffi_ind[suffi_ind == False].index + 1) 
    suffi_recomb_eta_B = recomb_eta_B[suffi_ind]
    insuffi_recomb_eta_B = recomb_eta_B[~suffi_ind]
    
    u_ind = [(i, j) for i in I_eta_B for j in np.arange(1, len(cut_eta_B[i]) + 1)]
    beta_tilde_ind = list(itertools.product(u_ind, K_eta_B_neg))
    beta_tilde_ind = [(*each[0], each[1]) for each in beta_tilde_ind]
    
    beta_tilde = {(i, g, k): 1 if recomb_eta_B[k - 1][i -1] >= cut_eta_B[i][g - 1] else 0 for i, g, k in beta_tilde_ind}
    
    # =============================================================================
    # p_eta = pd.Series(np.ones(num_scen) / num_scen)
    # p_eta.index += 1
    # =============================================================================
    
    num_L_hat = L_hat.shape[0]
    T_B = np.zeros((4 * num_L_hat, 2* num_L_hat))
    for row in np.arange(T_B.shape[0]):
        T_B[row, row // 2] = (-1) ** row
    
    M_a = np.ones(2 * num_L_hat) * 500
    M_r = np.ones(2 * num_L_hat) * 500
    K = np.arange(1, num_scen + 1)
    loc = ['r', 'u']
    h_hat = 3 # upper limit for mps at rural locations
    ind_z = [(m, i, j) for m in M for i in I_c for j in I_i[i]]
    
    ### init model
    model_name = 'mps'
    m = gp.Model(model_name)
    m.setParam(GRB.Param.TimeLimit, 1 * 3600)
    m.setParam(GRB.Param.Threads, 1) # this is not for computational test
    m.setParam(GRB.Param.LogFile, model_name)
    
    ### define optimization variables
    # variables for deterministic constraints
    x = m.addVars(M, vtype = GRB.BINARY, name = 'x')
    beta = m.addVars(M, I_mps, loc, vtype = GRB.BINARY, name = 'beta' )
    y = m.addVars(M, I_c, vtype = GRB.BINARY, name = 'y')
    z_a = m.addVars(ind_z, vtype = GRB.CONTINUOUS, name = 'z_a')
    z_r = m.addVars(ind_z, vtype = GRB.CONTINUOUS, name = 'z_r')
    # gamma = m.addVars(I, vtype = GRB.CONTINUOUS, ub = 1, name = 'gamma')
    gamma = m.addVars(I_mps, vtype = GRB.CONTINUOUS, name = 'gamma')
    for j in gamma:
        gamma[j].ub = 1
    
    v = m.addVars(I, vtype = GRB.CONTINUOUS, name = 'v')
    q = m.addVars(I, vtype = GRB.CONTINUOUS, name = 'q')
    psi_a = m.addVars(M, I_c, vtype = GRB.CONTINUOUS, name = 'psi_a' )
    psi_r = m.addVars(M, I_c, vtype = GRB.CONTINUOUS, name = 'psi_r' )
    
    # variables for decision-independent chance constraints
    # notice that f_a[0] corresponds to the first line. gurobi 
    # matrix variable starts with index 0
    f_a = m.addMVar(num_lines, vtype = GRB.CONTINUOUS, name = 'f_a', lb = -GRB.INFINITY)
    f_a_hat = f_a[L_hat - 1] # index start with 0 for gurobipy MVar object # f_hat is vulnerable line
    f_r = m.addMVar(num_lines, vtype = GRB.CONTINUOUS, name = 'f_r', lb = -GRB.INFINITY)
    f_r_hat = f_r[L_hat - 1]
    f_hat = gp.MVar.fromlist(f_a_hat.tolist() + f_r_hat.tolist())
    u = m.addVars(u_ind, vtype = GRB.BINARY, name = 'u')
    
    
    # variables related to decision-dependent individual  chance constraints (impacted nodes)
    phi = m.addVars(I_mps, vtype = GRB.INTEGER, name = 'phi') # phi should be integer variables
    epsilon_ind = np.arange(0, b + num_mps + 1)
    epsilon = m.addVars(I, epsilon_ind, vtype = GRB.BINARY, name = 'epsilon')
    tau = m.addVars(I_mps, epsilon_ind, vtype = GRB.CONTINUOUS, name = 'tau', ub = 1, lb = 0)
    z_B = m.addVars(z_B_ind, vtype = GRB.BINARY, name = 'z_B')
    q = m.addVars(I, vtype = GRB.CONTINUOUS, name = 'q')
    
    # q[30].ub = 800
    # q[30].lb = 800
    ### define optimization constraints
    
    # deterministic constraints
    m.addConstr((sum(x[m] for m in M) <= b), name = 'x_m<b') #2d
    m.addConstrs((beta[m, j, 'r'] <= x[m] for m in M for j in I_mps), name = 'beta_mjr<x_m' )
    m.addConstrs((beta[m, j, 'u'] <= 1 - x[m] for m in M for j in I_mps), name = 'beta_mju<1-x_m' )
    m.addConstrs((beta[m, j, 'u'] + beta[m, j, 'r'] <= sum(y[m, i] for i in I_c if j in I_i[i]) for m in M for j in I_mps), name = 'beta_mju<sum_y_mi') # need a constraint to ensure beta_mju cannot be 1 if m is not connected to i where i and j are connected
    m.addConstrs((sum(y[m, i] for m in M) <= h_hat for i in I_c), name = 'y_mi<h')
    m.addConstr((sum(y[m, i] for i in I_c for m in M) <= num_mps), name = 'y_mi<M_for')
    m.addConstrs((sum(y[m, i] for i in I_c) <= 1 for m in M), name = 'one_i_per_m')
    m.addConstrs((z_a[m, i, j] <= min(Psi_a[m], D_a[j]) * y[m, i] for m in M for i in I_c for j in I_i[i]), name = 'z<min(phi,D)_a')
    m.addConstrs((z_r[m, i, j] <= min(Psi_r[m], D_r[j]) * y[m, i] for m in M for i in I_c for j in I_i[i]), name = 'z<min(phi,D)_r')
    m.addConstrs((sum(z_a[m, i, j] for i in I_c for j in I_i[i]) <= Psi_a[m] for m in M), name = 'sum_z_a<phi_a')
    m.addConstrs((sum(z_r[m, i, j] for i in I_c for j in I_i[i]) <= Psi_r[m] for m in M), name = 'sum_z_r<phi_r')
    m.addConstrs((sum(z_a[m, i, j] for m in M for i in I_c if j in I_i[i]) == D_a[j] * gamma[j] for j in I_mps), name = 'gamma_def')
    m.addConstrs((sum(2 * beta[m, j, 'r'] + beta[m, j, 'u'] for m in M) == phi[j] for j in I_mps), name = 'phi_def')
    
    m.addConstrs((sum(f_a[l - 1] for l in L if Lambda[l] == j) + sum(psi_a[m, j] for m in M)  ==  sum(f_a[l - 1] for l in L if Gamma[l] == j) + sum(z_a[m, i, j] for m in M for i in I_c if j in I_i[i]) for j in I_c), name = 'flow_bal_a_not_ic')
    m.addConstrs((sum(f_a[l - 1] for l in L if Lambda[l] == j)  ==  sum(z_a[m, i, j] for m in M for i in I_c if j in I_i[i]) + sum(f_a[l - 1] for l in L if Gamma[l] == j) for j in I if j not in I_c), name = 'flow_bal_a_ic')
    m.addConstrs((sum(f_r[l - 1] for l in L if Lambda[l] == j) + sum(psi_r[m, j] for m in M)  ==  sum(f_r[l - 1] for l in L if Gamma[l] == j) + sum(z_r[m, i, j] for m in M for i in I_c if j in I_i[i]) for j in I_c), name = 'flow_bal_r_not_ic')
    m.addConstrs((sum(f_r[l - 1] for l in L if Lambda[l] == j)  ==  sum(z_r[m, i, j] for m in M for i in I_c if j in I_i[i]) + sum(f_r[l - 1] for l in L if Gamma[l] == j) for j in I if j not in I_c), name = 'flow_bal_r_ic')
    
    m.addConstrs((psi_a[m, i] <= Psi_a[m] * y[m, i] for i in I_c for m in M), name = 'psi_a<Psi_a')
    m.addConstrs((psi_r[m, i] <= Psi_r[m] * y[m, i] for i in I_c for m in M), name = 'psi_r<Psi_r')
    
    m.addConstrs((psi_a[m, i] == sum(z_a[m, i, j] for j in I_i[i]) for i in I_c for m in M), name = 'psi_a=sum_z')
    m.addConstrs((psi_r[m, i] == sum(z_r[m, i, j] for j in I_i[i]) for i in I_c for m in M), name = 'psi_r=sum_z')
    
    m.addConstrs((psi_a[m, i] * tan_theta_low[i] <= psi_r[m, i] for i in I_c for m in M), name = 'psi<tan_theta_low')
    m.addConstrs((psi_r[m, i] <= psi_a[m, i] * tan_theta_up[i] for i in I_c for m in M), name = 'psi<tan_theta_up')
    
    m.addConstrs((v[Gamma[l]] - v[Lambda[l]] == 2 * (f_a[l - 1] * R[l] + f_r[l - 1] * X[l]) / 1000 for l in L), name = 'square_voltage')
    
    m.addConstrs((beta[m, j, 'r'] <= sum(z_a[m, i, j] for i in I_c if j in I_i[i]) for m in M for j in I_mps), name = 'beta_r_z')
    m.addConstrs((beta[m, j, 'u'] <= sum(z_a[m, i, j] for i in I_c if j in I_i[i]) for m in M for j in I_mps), name = 'beta_u_z')
    
    m.addConstrs((v[j] >= V_under[j]  for j in I), name = 'V_under<v')
    m.addConstrs((v[j] <= V_over[j] for j in I), name = 'v<V_over')
    
    m.addConstrs((-F_a[l] <= f_a[l - 1] for l in L if l not in L_hat), name = '-F_a<f_a')
    m.addConstrs((f_a[l - 1] <= F_a[l] for l in L if l not in L_hat), name = 'f_a<F_a')
    m.addConstrs((-F_r[l] <= f_r[l - 1] for l in L if l not in L_hat), name = '-F_r<f_r')
    m.addConstrs((f_r[l - 1] <= F_r[l] for l in L if l not in L_hat), name = 'f_r<F_r')
    
    ###########
    m.addConstrs((beta[m, j, 'r'] <= y[m, i] * connect_ind[i, j] for m in M for i in I_c for j in I_i[i]), name = 'beta_mjr <= y_mi')
    m.addConstrs((beta[m, j, 'u'] <= y[m, i] * connect_ind[i, j] for m in M for i in I_c for j in I_i[i]), name = 'beta_mju <= y_mi')
    
    ##############################################################
    ##### decision-independent (for vulnerable lines) constraints
    ##############################################################
    m.addConstrs((T_B[i - 1, :] @ f_hat >= sum(c_tilde[i, g] * u[i, g] for g in g_eta_B[i]) for i in I_eta_B), name = 'Tf>sum_c_u')
    m.addConstrs((sum(beta_tilde[i, g, k] * u[i, g] for i in I_eta_B for g in g_eta_B[i]) <= I_eta_B[-1] - 1 for k in K_eta_B_neg), name = 'beta_u<I_eta-1')
    m.addConstrs((sum(u[i, g] for g in g_eta_B[i]) == 1 for i in I_eta_B), name = 'sum_u_ig=1')
    
    ##############################################################
    ##### decision-dependent individual constraints
    ##############################################################
    # calculating cumulative distorted probability for potential sufficient scenarios
    C_B_cum_prob = return_cum_p_d(scen_xi = scen_xi_sampled, Omega_up_ind = Omega_up_ind, Omega_down_ind = Omega_down_ind, C_B = C_B, tau = tau, epsilon_ind = epsilon_ind)
    
    # constraints
    m.addConstrs((z_B[j, k - 1] >= z_B[j, k] for j in I_mps for k in N_B[j] if k >= 2), name = 'z_B_k-1>k')
    
    m.addConstrs((C_B_cum_prob[j].loc[k, 'cum_prob_d'] >= alpha[j] * z_B[j, k] for j in I_mps for k in np.arange(1, len(C_B[j]) + 1)), name = 'F>alpha_z')
    
    m.addConstrs((phi[j] == sum(n * epsilon[j, n] for n in epsilon_ind) for j in I_mps), name = 'phi=sum_n*epsilon')
    
    m.addConstrs((sum(epsilon[j, n] for n in epsilon_ind) == 1 for j in I_mps), name = 'sum_epsilon=1')
    
    m.addConstrs((tau[j, n] >= gamma_lb * epsilon[j, n] for j in I_mps for n in epsilon_ind), name = 'mc_tau_1')
    m.addConstrs((tau[j, n] >= gamma_ub * (epsilon[j, n] - 1) + gamma[j] for j in I_mps for n in epsilon_ind), name = 'mc_tau_2')
    m.addConstrs((tau[j, n] <= gamma_ub * epsilon[j, n] for j in I_mps for n in epsilon_ind), name = 'mc_tau_3')
    m.addConstrs((tau[j, n] <= gamma_lb * (epsilon[j, n] - 1) + gamma[j] for j in I_mps for n in epsilon_ind), name = 'mc_tau_4')
    
    m.addConstrs((q[j] >= C_B[j][1] - sum(z_B[j, k + 1] * A_B[j, k] for k in N_B[j] if k < N_B[j][-1]) for j in I_mps), name = 'q>c-z*delta')
    
    ### setting objective functions
    m.setObjective(sum(q[j] for j in I), GRB.MINIMIZE)
    
    ### testing
    # =============================================================================
    # gamma[30].lb = 1.0
    # gamma[30].ub = 1.0
    # 
    # =============================================================================
    # =============================================================================
    # q[30].lb = 800
    # q[30].ub = 800
    # 
    # gamma[30].lb = 0.9891
    # gamma[30].ub = 0.9891
    # 
    # phi[30].lb = 3
    # phi[30].ub = 3
    # 
    # =============================================================================
    ###
    ### solving the model
    m.optimize()
    
    # =============================================================================
    # m.computeIIS()
    # 
    # for constr in m.getConstrs():
    #     if constr.IISConstr:
    #         print(f"Violated Constraint: {constr.constrName}")
    # =============================================================================
    
    m.ObjVal
    
    
    ### extract optimal solutions
    # sol_df = pd.DataFrame()
    # sol_df.index = pd.Index(q.keys())
    print('=======================================')
    print(f'running time is {m.Runtime}')
    
    print('=======================================')
    print('solution')
    q_sol = pd.Series(q.values(), index = q.keys())
    print(q_sol)
    x_sol = pd.Series(x.values(), index = x.keys())
    print(x_sol)
    
    y_sol = pd.Series(y.values(), index = y.keys())
    y_nz_ind = [True if row.X != 0 else False for row in y_sol]
    print(y_sol[y_nz_ind])
    
    beta_sol = pd.Series(beta.values(), index = beta.keys())
    beta_nz_ind = [True if row.X != 0 else False for row in beta_sol]
    print(beta_sol[beta_nz_ind])
    
    z_a_sol = pd.Series(z_a.values(), index = z_a.keys())
    z_a_nz_ind = [True if row.X != 0 else False for row in z_a_sol]
    print(z_a_sol[z_a_nz_ind])
    
    
    z_r_sol = pd.Series(z_r.values(), index = z_r.keys())
    z_r_nz_ind = [True if row.X != 0 else False for row in z_r_sol]
    print(z_r_sol[z_r_nz_ind])
    
    print('this variable starts with index 0; so f_a[0] is f_a[1] in our model')
    print([var for var in f_a if var.X != 0])
    
    print('this variable starts with index 0; so f_a[0] is f_a[1] in our model')
    print([var for var in f_r if var.X != 0])
    
    # =============================================================================
    # z_tilde_sol = pd.Series(z_tilde.values(), index = z_tilde.keys())
    # z_tilde_nz_ind = [True if row.X != 0 else False for row in z_tilde_sol]
    # z_tilde_sol[z_tilde_nz_ind]
    # =============================================================================
    
    gamma_sol = pd.Series(gamma.values(), index = gamma.keys())
    gamma_nz_ind = [True if row.X != 0 else False for row in gamma_sol]
    print(gamma_sol[gamma_nz_ind])
    
    psi_a_sol = pd.Series(psi_a.values(), index = psi_a.keys())
    psi_a_nz_ind = [True if row.X != 0 else False for row in psi_a_sol]
    print(psi_a_sol[psi_a_nz_ind])
    
    psi_r_sol = pd.Series(psi_r.values(), index = psi_r.keys())
    psi_r_nz_ind = [True if row.X != 0 else False for row in psi_r_sol]
    print(psi_r_sol[psi_r_nz_ind])
    
    phi_sol = pd.Series(phi.values(), index = phi.keys())
    phi_nz_ind = [True if row.X != 0 else False for row in phi_sol]
    print(phi_sol[phi_nz_ind])
    
    v_sol = pd.Series(v.values(), index = v.keys())
    v_nz_ind = [True if row.X != 0 else False for row in v_sol]
    print(v_sol[v_nz_ind])



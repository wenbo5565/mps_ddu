"""
This script is for the paper: mobile power source 
"""

import pandas as pd
import numpy as np
import itertools
import re
import gurobipy as gp
from gurobipy import GRB
# import gurobipy_pandas as gppd
import os

### define helper functions
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

data_folder = os.path.dirname(os.getcwd())
# data_folder = r'D:\Research\gw_ddu_mps'
scen_xi = pd.read_csv(os.path.join(data_folder, "data", "xi_info_v6.csv"))
scen_eta = pd.read_csv(os.path.join(data_folder, "data", "eta_info_v6.csv"))
scen_sub = pd.read_csv(os.path.join(data_folder, "data", "SubNet_Info_v6.csv"))
other_params = pd.read_csv(os.path.join(data_folder, "data", "Deter_param_v2_formatted.csv"))

num_nodes = 33
num_lines = 32
num_scen = scen_eta.shape[0]

other_params.isnull().sum(axis = 0)
other_params.index += 1

### Adding function to resample the scenarios
random_state = [2010, 2020, 2030, 2040, 2050]
frac = 1

print('===============================')
print(f'number of scenarios is {frac * 1000}')

for rnd in random_state:
    print('==========================================')
    print(f'result for random state {rnd}')
    print('==========================================')

    
    scen_xi_sampled = scen_xi.sample(frac = frac, random_state = rnd, ignore_index = True, replace = True)
    scen_eta_sampled = scen_eta.sample(frac = frac, random_state = rnd, ignore_index = True, replace = True)
    
    if (scen_xi_sampled.index != scen_eta_sampled.index).any():
        print('sample index for node and line are not the same...')

    scen_xi_sampled.isnull().sum(axis = 0)
    scen_eta_sampled.isnull().sum(axis = 0)
    
    scen_xi_sampled.index += 1
    scen_eta_sampled.index += 1
    
    scen_xi_sampled.columns = np.arange(1, num_nodes + 1)
    xi = {(scen, node): scen_xi_sampled.loc[scen, node] for scen, node in itertools.product(scen_xi_sampled.index, scen_xi_sampled.columns)}
    
    for c in scen_xi_sampled.columns:
        print('----', c, '----')
        print(scen_xi_sampled[c].value_counts(normalize = True).sort_index())
    
    ### creating sets: Omega_up, Omega_down
    
    
    scen_eta_sampled.columns = [int(re.findall('[0-9]+$', col)[0]) for col in scen_eta_sampled.columns]
    # eta =  {(scen, node): scen_eta_sampled.loc[scen, node] for scen, node in itertools.product(scen_eta.index, scen_eta.columns)}
    eta = scen_eta_sampled * -1
    eta = eta[sorted(pd.concat([eta] * 2, axis = 1).columns)]
    
    scen_sub.columns = [int(re.findall('[0-9]+$', col)[0]) for col in scen_sub.columns]
    
    
    
    
    
    
    xi_sub = {(scen, sub_num): scen_sub.loc[scen, sub_num] for scen, sub_num in itertools.product(scen_sub.index, scen_sub.columns)}
    
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
    N_ind = np.arange(0, num_mps + b + 1) # index for ???
    mps_cap_a = 500 # active power capacity for mps
    mps_cap_r = 500 # reactive power capacity for mps
    
    M = np.arange(1, num_mps + 1) # set for MPS
    Psi_a = pd.Series(mps_cap_a * np.ones(num_mps), index = np.arange(1, num_mps + 1))
    Psi_r = pd.Series(mps_cap_r * np.ones(num_mps), index = np.arange(1, num_mps + 1))
    
    ### parameters for decision-dependent functions
    low_quant = 0.2 # threashold for the set Omega_down
    Omega_ind = {col: return_omega_index(scen_xi_sampled[col], low_quant) for col in scen_xi_sampled.columns} # return scenario index
    
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
    
    ### parameters for scenario-based reformulation
    # parameters for decision-independent
    p_eta = pd.Series(np.ones(num_scen) / num_scen)
    p_eta.index += 1
    alpha_L = 0.90
    num_L_hat = L_hat.shape[0]
    T = np.zeros((2 * num_L_hat, num_L_hat))
    for row in np.arange(T.shape[0]):
        T[row, row // 2] = (-1) ** row
    
    M_a = np.ones(2 * num_L_hat) * 500
    M_r = np.ones(2 * num_L_hat) * 500
    K = np.arange(1, num_scen + 1)
    loc = ['r', 'u']
    h_hat = 3 # upper limit for mps at rural locations
    ind_z = [(m, i, j) for m in M for i in I_c for j in I_i[i]]
    
    ### init model
    model_name = 'mps'
    m = gp.Model(model_name)
    m.setParam(GRB.Param.TimeLimit, 12 * 3600)
    m.setParam(GRB.Param.Threads, 1) 
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
    
    
    # q[33].ub = 0
    # variables related to decision-independent chance constraints (vulnerable lines)
    # f_a = gppd.add_vars(m, pd.Index(L), name = 'f_a', vtype = GRB.CONTINUOUS)
    
    # notice that f_a[0] corresponds to the first line. gurobi 
    # matrix variable starts with index 0
    f_a = m.addMVar(num_lines, vtype = GRB.CONTINUOUS, name = 'f_a', lb = -GRB.INFINITY)
    f_a_hat = f_a[L_hat - 1] # index start with 0 for gurobipy MVar object # f_hat is vulnerable line
    f_r = m.addMVar(num_lines, vtype = GRB.CONTINUOUS, name = 'f_r', lb = -GRB.INFINITY)
    f_r_hat = f_r[L_hat - 1]
    z_hat = m.addMVar(num_scen, vtype = GRB.BINARY, name = 'z_hat')
    
    # variables related to decision-dependent individual  chance constraints (impacted nodes)
    phi = m.addVars(I_mps, vtype = GRB.INTEGER, name = 'phi') # phi should be integer variables
    epsilon_ind = np.arange(0, b + num_mps + 1)
    epsilon = m.addVars(I, epsilon_ind, vtype = GRB.BINARY, name = 'epsilon')
    h_ind = [(j, k) for j in I for k in K if k in Omega_ind[j][1] or k in Omega_ind[j][2]] # index set for Mccormick variable h. Index 1, 2 corresponds to up and down scenarios?
    h = m.addVars(h_ind, vtype = GRB.CONTINUOUS, name = 'h_mccormick') # h is bounded between 0 and 1?
    pi_ind = [(j, k, ep_ind) for j, k in h_ind for ep_ind in epsilon_ind]
    pi = m.addVars(pi_ind, vtype = GRB.CONTINUOUS, name = 'pi_mccormick')
    z_tilde = m.addVars(I, K, vtype = GRB.BINARY, name = 'z_tilde')
    q = m.addVars(I, vtype = GRB.CONTINUOUS, name = 'q')
    
    
    ### define optimization constraints
    
    # deterministic constraints
    m.addConstr((sum(x[m] for m in M) <= b), name = 'x_m<b') #2d
    m.addConstrs((beta[m, j, 'r'] <= x[m] for m in M for j in I_mps), name = 'beta_mjr<x_m' )
    m.addConstrs((beta[m, j, 'u'] <= 1 - x[m] for m in M for j in I_mps), name = 'beta_mju<1-x_m' )
    m.addConstrs((beta[m, j, 'u'] + beta[m, j, 'r'] <= sum(y[m, i] for i in I_c if j in I_i[i]) for m in M for j in I_mps), name = 'beta_mju<sum_y_mi') # need a constraint to ensure beta_mju cannot be 1 if m is not connected to i where i and j are connected
    
    m.addConstrs((sum(y[m, i] for m in M) <= h_hat for i in I_c), name = 'y_mi<h')
    m.addConstr((sum(y[m, i] for i in I_c for m in M) <= num_mps), name = 'y_mi<M_for')
    m.addConstrs((sum(y[m, i] for i in I_c) <= 1 for m in M), name = 'one_i_per_m')
    
    ##### start of testing
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
    
    ##### end of testing
    m.addConstrs((sum(z_a[m, i, j] for i in I_c if j in I_i[i]) * tan_theta_low[j] <= sum(z_r[m, i, j] for i in I_c if j in I_i[i]) for j in I_mps for m in M), name = 'psi<tan_theta_low')
    m.addConstrs((sum(z_r[m, i, j] for i in I_c if j in I_i[i]) <= sum(z_a[m, i, j] for i in I_c if j in I_i[i]) * tan_theta_up[i] for j in I_mps for m in M), name = 'psi<tan_theta_up')
    
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
    m.addConstrs((T @ f_a_hat + M_a * (1 - z_hat[k - 1]) >= eta.loc[k, :].values for k in K), name = 'act-Tf+(1-z)M>=eta') # z_hat is gurobi M variable starting with index 0
    m.addConstrs((T @ f_r_hat + M_r * (1 - z_hat[k - 1]) >= eta.loc[k, :].values for k in K), name = 'react-Tf+(1-z)M>=eta') 
    m.addConstr(sum(p_eta[k] * z_hat[k - 1] for k in K) >= alpha_L, name = 'knapsack')
    
    ##############################################################
    ##### decision-dependent individual constraints
    ##############################################################
    # =============================================================================
    # m.addConstrs((sum(p_b[k] * z_tilde[j, k] for k in K) >=
    #               alpha[j] for j in I), name = 'long_dec_depen')
    # =============================================================================
    m.addConstrs((q[j] + (1 - z_tilde[j, k]) * xi[k, j] >= xi[k, j] for j in I for k in K), name = 'q+(1-z)w>w')
    
    
    # extract index for sets Omega_0, Omega_down and Omega_up
    Omega_0_ind = {j: Omega_ind[j][0] for j in Omega_ind.keys()}
    Omega_up_ind = {j: Omega_ind[j][1] for j in Omega_ind.keys()}
    Omega_down_ind = {j: Omega_ind[j][2] for j in Omega_ind.keys()}
    m.addConstrs((Omega_down_ind[j].shape[0] * (sum(p_b[k] * z_tilde[j, k] for k in K if k in Omega_0_ind[j]) + sum(p_b[k] * z_tilde[j, k] for k in K if k in Omega_up_ind[j]) + sum(p_b[k] * c[j] * sum(n * pi[j, k, n] for n in N_ind) for k in K if k in Omega_up_ind[j])) 
                  + (1 - sum(p_b[k] for k in K if k in Omega_0_ind[j])) * sum(z_tilde[j, k] for k in K if k in Omega_down_ind[j])
                  - sum(p_b[k] for k in K if k in Omega_up_ind[j]) * sum(z_tilde[j, k] for k in K if k in Omega_down_ind[j])
                  - sum(p_b[k_prime] for k_prime in K if k_prime in Omega_up_ind[j]) * sum(c[j] * sum(n * pi[j, k, n] for n in N_ind) for k in K if k in Omega_down_ind[j])
                  >= alpha[j] * Omega_down_ind[j].shape[0] for j in I), name = 'long_dec_depen')
    
    m.addConstrs((phi[j] == sum(n * epsilon[j, n] for n in epsilon_ind) for j in I_mps), name = 'phi=sum_n*epsilon')
    m.addConstrs((sum(epsilon[j, n] for n in epsilon_ind) == 1 for j in I_mps), name = 'sum_epsilon=1')
    
    m.addConstrs((h[j, k] >= gamma_lb * z_tilde[j, k] for j, k in h_ind), name = 'mc_h_1')
    m.addConstrs((h[j, k] >= gamma_ub * (z_tilde[j, k] - 1) + gamma[j] for j, k in h_ind), name = 'mc_h_2')
    m.addConstrs((h[j, k] <= gamma_ub * z_tilde[j, k] for j, k in h_ind), name = 'mc_h_3')
    m.addConstrs((h[j, k] <= gamma_lb * (z_tilde[j, k] - 1) + gamma[j] for j, k in h_ind), name = 'mc_h_4')
    
    m.addConstrs((pi[j, k, n] >= h_lb * epsilon[j, n] for j, k, n in pi_ind), name = 'mc_pi_1')
    m.addConstrs((pi[j, k, n] >= h_ub * (epsilon[j, n] - 1) + h[j, k] for j, k, n in pi_ind), name = 'mc_pi_2')
    m.addConstrs((pi[j, k, n] <= h_ub * epsilon[j, n] for j, k, n in pi_ind), name = 'mc_pi_3')
    m.addConstrs((pi[j, k, n] <= h_lb * (epsilon[j, n] - 1) + h[j, k] for j, k, n in pi_ind), name = 'mc_pi_4')
    
    # =============================================================================
    # test_node = 33
    # len(Omega_0_ind[test_node])
    # len(Omega_up_ind[test_node])
    # len(Omega_down_ind[test_node])
    # =============================================================================
    
    ### setting objective functions
    m.setObjective(sum(q[j] for j in I), GRB.MINIMIZE)
    
    # m.setObjective(sum(z_a[m, i, j] for m in M for i in I_c for j in I if j in I_i[i] ), GRB.MAXIMIZE)
    
    ### testing
# =============================================================================
#     q[1].ub =0.0
#     q[2].ub =0.0
#     q[3].ub =315.0
#     q[4].ub =420.0
#     q[5].ub =210.0
#     q[6].ub =300.0
#     q[7].ub =1000.0
#     q[8].ub =1000.0
#     q[9].ub =300.0
#     q[10].ub =300.0
#     q[11].ub =225.0
#     q[12].ub =300.0
#     q[13].ub =300.0
#     q[14].ub =600.0
#     q[15].ub =300.0
#     q[16].ub =300.0
#     q[17].ub =300.0
#     q[18].ub =450.0
#     q[19].ub =450.0
#     q[20].ub =315.0
#     q[21].ub =315.0
#     q[22].ub =450.0
#     q[23].ub =108.0
#     q[24].ub =504.0
#     q[25].ub =504.0
#     q[26].ub =300.0
#     q[27].ub =300.0
#     q[28].ub =300.0
#     q[29].ub =600.0
#     q[30].ub =1000.0
#     q[31].ub =750.0
#     q[32].ub =1050.0
#     q[33].ub =300.0
#     
#     q[1].lb =0.0
#     q[2].lb =0.0
#     q[3].lb =315.0
#     q[4].lb =420.0
#     q[5].lb =210.0
#     q[6].lb =300.0
#     q[7].lb =1000.0
#     q[8].lb =1000.0
#     q[9].lb =300.0
#     q[10].lb =300.0
#     q[11].lb =225.0
#     q[12].lb =300.0
#     q[13].lb =300.0
#     q[14].lb =600.0
#     q[15].lb =300.0
#     q[16].lb =300.0
#     q[17].lb =300.0
#     q[18].lb =450.0
#     q[19].lb =450.0
#     q[20].lb =315.0
#     q[21].lb =315.0
#     q[22].lb =450.0
#     q[23].lb =108.0
#     q[24].lb =504.0
#     q[25].lb =504.0
#     q[26].lb =300.0
#     q[27].lb =300.0
#     q[28].lb =300.0
#     q[29].lb =600.0
#     q[30].lb =1000.0
#     q[31].lb =750.0
#     q[32].lb =1050.0
#     q[33].lb =300.0
# =============================================================================
    
    ### solving the model
    m.optimize()
    m.ObjVal
    
    
    # =============================================================================
    # for c in scen_xi.columns:
    #     print('----', c, '----')
    #     print(scen_xi[c].value_counts(normalize = True).sort_index())
    # =============================================================================
    ### extract optimal solutions
    # sol_df = pd.DataFrame()
    # sol_df.index = pd.Index(q.keys())
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
    
    print(f_a)
    
    print(f_r)
    
    z_tilde_sol = pd.Series(z_tilde.values(), index = z_tilde.keys())
    z_tilde_nz_ind = [True if row.X != 0 else False for row in z_tilde_sol]
    print(z_tilde_sol[z_tilde_nz_ind])
    
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
    
    pi_sol = pd.Series(pi.values(), index = pi.keys())
    pi_nz_ind = [True if row.X != 0 else False for row in pi_sol]
    print(pi_sol[pi_nz_ind])
    # print(pi_sol[pi_nz_ind][30])
    
    v_sol = pd.Series(v.values(), index = v.keys())
    v_nz_ind = [True if row.X != 0 else False for row in v_sol]
    print(v_sol[v_nz_ind])

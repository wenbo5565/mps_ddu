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

# =============================================================================
# scen = scen_xi
# sub_network = test
# q_thresh = low_quant
# s = 7
# test_vec = (210, 300, 300, 600, 1000)  # (300, 300, 300, 600, 800)
# 
# scen_s['cum_prob'] = scen_s.apply(lambda x: scen_s.le(test_vec).all(axis = 1).sum() / scen_s.shape[0], axis = 1)
# 
# =============================================================================
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
    
def return_joint_omega_index(scen_xi, sub_network, q_thresh):
    """
        take a set of multi-dimensional scenario values of ens;
        return a dict with keys being subnetwork and values are up, down and zero index
    """
    
    omega_ind = {}
    for s in sub_network.keys():
        scen_s = scen_xi[sub_network[s]].copy()
        # index for 0 vector
        col_sum = scen_s.sum(axis = 1)
        ind_0 = np.where(col_sum == 0)[0] + 1
        # 
        scen_s_wo_0 = scen_s.loc[list(set(scen_s.index) - set(ind_0)), :].copy()
        scen_s_wo_0['cum_prob'] = scen_s_wo_0.apply(lambda x: scen_s_wo_0.le(x).all(axis = 1).sum() / scen_s_wo_0.shape[0], axis = 1) # calculate cumulative probability of each scenarios
        print(f'cumulative prob for sub network {s}')
        
        
        # print(scen_s.apply(lambda x: scen_s.le(x).all(axis = 1).sum() / scen_s.shape[0], axis = 1)) 
        scen_s['tuple'] = list(scen_s.itertuples(index = False, name = None))
        scen_s['cum_prob'] = scen_s.apply(lambda x: scen_s.le(x).all(axis = 1).sum() / scen_s.shape[0], axis = 1)
        print(scen_s.groupby('tuple')['cum_prob'].first().sort_values())
        
        up_ind = scen_s_wo_0[scen_s_wo_0['cum_prob'] <= q_thresh].index
        down_ind = scen_s_wo_0[scen_s_wo_0['cum_prob'] > q_thresh].index
        omega_ind[s] = {'up': list(up_ind), 
                     'down': list(down_ind), 
                     'zero': list(ind_0)}
       
    return omega_ind

# ??? function to calculate cut point and recombination
def return_cut_points(E, c, scen_xi, sub_network, alpha, q_thresh):
    """
        return cut points and recombination given e in E, c (sensitivity)
    """    
    S = sub_network.keys()
    omega_ind = return_joint_omega_index(scen_xi = scen_xi, sub_network = sub_network, q_thresh = q_thresh)
    cuts = {}
    recombs = {}
    p_insuff_recombs = {}
    p_suff_recombs = {}
    beta_hat = {}
    
    """
    s = 3
    q_thresh = 0.2
    e = 24
    
    """
    
    for s in S:
        for e_ind, e in enumerate(E[s]): 
            omega_0_ind = omega_ind[s]['zero']
            omega_up_ind = omega_ind[s]['up']
            omega_down_ind = omega_ind[s]['down']
            scen_s = scen_xi.loc[:, sub_network[s]].copy()
            p_0 = 1 / scen_s.shape[0]
            if 1 / scen_s.shape[0] * (1 + c[s] * e) <= (1 - p_0 * len(omega_0_ind)) / len(omega_up_ind):
                p_up = 1 / scen_s.shape[0] * (1 + c[s] * e)
                p_0_up = p_0 * len(omega_0_ind) + p_up * len(omega_up_ind)
                p_down = (1 - p_0_up) / len(omega_down_ind)
            else:
                p_up = (1 - p_0 * len(omega_0_ind)) / len(omega_up_ind)
                p_down = 0
            # get distorted probability
            scen_s['prob_d'] = p_0
            scen_s.loc[omega_up_ind, 'prob_d'] = p_up
            scen_s.loc[omega_down_ind, 'prob_d'] = p_down
            # Extract numerical columns (excluding 'prob_d')
            numeric_cols = scen_s.columns.difference(["prob_d"])
            # Convert to NumPy arrays for fast computation
            X = scen_s[numeric_cols].to_numpy()  # Feature matrix
            weights = scen_s["prob_d"].to_numpy()  # Probability column
            comparison_matrix = np.all(X[:, None, :] >= X[None, :, :], axis=2)
            scen_s['cum_prob'] = comparison_matrix @ weights
            scen_s['tuple'] = list(scen_s[numeric_cols].itertuples(index = False, name = None))
            # get cut points for each j in S
            cut = {}
            for j in sub_network[s]:
                scen_j = scen_s[[j]].copy()
                scen_j_np = scen_s[j].to_numpy()
                comp_matrix = scen_j_np[:, None] >= scen_j_np[None, :]
                scen_j['cum_prob'] = comp_matrix @ weights
                cut[j] = list(set(scen_j.loc[scen_j.cum_prob >= alpha[s], j])) 
                cuts[s, e, j] = cut[j]
            """
            temp = np.array([210, 225, 300, 300, 420])
            temp = np.array([210, 225, 300, 300, 600])
            temp = np.array([300, 225, 300, 300, 600])
            (temp[None, :] >= X).all(axis = 1) @ weights
            """
            # get recombinations
            recomb = list(itertools.product(*[cuts[s, e, j] for j in sub_network[s]]))
            recombs[(s, e)] = recomb
            # obtain which recombs are p insufficient
            recomb_np = pd.DataFrame(recomb).to_numpy()
            comp_matrix = (recomb_np[:, None, :] >= X[None, :, :]).all(axis = 2)
            recomb_cum_prob = comp_matrix @ weights
            p_insuff_recomb = np.array(recomb)[recomb_cum_prob < alpha[s]]
            p_suff_recomb = np.array(recomb)[recomb_cum_prob >= alpha[s]]
            p_insuff_recombs[(s, e)] = p_insuff_recomb
            p_suff_recombs[(s, e)] = p_suff_recomb
            for k, p_insuff_iter in enumerate(p_insuff_recomb):
                for ind, ens in enumerate(p_insuff_iter):
                    j = sub_network[s][ind]
                    for g_ind, cut_num in enumerate(cut[j]):
                            if ens >= cut_num:
                                beta_hat[s, e, j, k + 1, g_ind + 1] = 1
                            else:
                                beta_hat[s, e, j, k + 1, g_ind + 1] = 0
    return beta_hat, cuts, p_suff_recombs, p_insuff_recombs     

data_folder = os.path.dirname(os.getcwd())
# data_folder = r'D:\Research\gw_ddu_mps'
scen_xi_csv = pd.read_csv(os.path.join(data_folder, "data", "xi_info_v6.csv"))
scen_eta_csv = pd.read_csv(os.path.join(data_folder, "data", "eta_info_v6.csv"))
scen_sub_csv = pd.read_csv(os.path.join(data_folder, "data", "SubNet_Info_v6.csv"))
other_params = pd.read_csv(os.path.join(data_folder, "data", "Deter_param_v2_formatted.csv"))

other_params.isnull().sum(axis = 0)
scen_xi_csv.isnull().sum(axis = 0)
scen_eta_csv.isnull().sum(axis = 0)
scen_sub_csv.isnull().sum(axis = 0)

other_params.index += 1
scen_xi_csv.index += 1
scen_eta_csv.index += 1
scen_sub_csv.index += 1


### Adding function to resample the scenarios
random_state = [2010, 2020, 2030, 2040, 2050]
frac = 1

"""
rnd = 2010
"""
#for rnd in random_state:
#    scen_xi_sampled = scen_xi.sample(frac = frac, random_state = rnd)
### Adding function to resample the scenarios
print(f'number of scenarios: {frac * scen_xi_csv.shape[0]}')
for rnd in random_state: 
    #for rnd in random_state:
    # rnd = 2010
    scen_xi = scen_xi_csv.sample(frac = frac, random_state = rnd, ignore_index = True, replace = True)
    scen_eta = scen_eta_csv.sample(frac = frac, random_state = rnd, ignore_index = True, replace = True)
    scen_sub = scen_sub_csv.sample(frac = frac, random_state = rnd, ignore_index = True, replace = True)
    
    scen_xi.index += 1
    scen_eta.index += 1
    scen_sub.index += 1
    
    num_nodes = 33
    num_lines = 32
    num_scen = scen_eta.shape[0]
    
    print(f'sample size is {scen_xi.shape[0], scen_sub.shape[0], scen_eta.shape[0]}')
    
    # node ENS
    scen_xi.columns = np.arange(1, num_nodes + 1)
    xi = {(scen, node): scen_xi.loc[scen, node] for scen, node in itertools.product(scen_xi.index, scen_xi.columns)}
    # =============================================================================
    # for c in scen_xi.columns:
    #     print('----', c, '----')
    #     print(scen_xi[c].value_counts(normalize = True).sort_index())
    # 
    # =============================================================================
    # line ENS
    scen_eta.columns = [int(re.findall('[0-9]+$', col)[0]) for col in scen_eta.columns]
    eta = scen_eta * -1
    eta = eta[sorted(pd.concat([eta] * 2, axis = 1).columns)]
    
    # subnetwork ENS
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
    num_subs = len(sub_network)
    S = np.arange(1, num_subs + 1)
    
    # =============================================================================
    # for s in S:
    #     scen_sub_node = scen_xi[sub_network[s]].copy()
    #     scen_sub_node['joint'] = list(scen_sub_node.itertuples(index = False, name = None))
    #     print(f'distribution of subnetwork {s}')
    #     print(scen_sub_node['joint'].value_counts(normalize = True).sort_index().cumsum())
    # 
    # =============================================================================
    ###### define parameters
    ### parameters for mobile power sources
    num_mps = 6
    b = 3 # maximal MPS pre-disposed at rural locations
    # N_ind = np.arange(0, num_mps + b + 1) # index for ???
    mps_cap_a = 500 # active power capacity for mps
    mps_cap_r = 500 # reactive power capacity for mps
    
    M = np.arange(1, num_mps + 1) # set for MPS
    Psi_a = pd.Series(mps_cap_a * np.ones(num_mps), index = np.arange(1, num_mps + 1))
    Psi_r = pd.Series(mps_cap_r * np.ones(num_mps), index = np.arange(1, num_mps + 1))
    
    ### parameters for decision-dependent functions
    low_quant = 0.2 # threashold for the set Omega_down
    
    # Need to add new function to calculate subnetwork level value
    Omega_ind = return_joint_omega_index(scen_xi = scen_xi, sub_network = sub_network, q_thresh = low_quant)
    
    ### parameters for nodes
    D_a = other_params['D_a']
    D_r = other_params['D_r']
    V_under = other_params['V_under']
    V_over = other_params['V_over']
    tan_theta_up = other_params['tan_theta_up']
    tan_theta_low = other_params['tan_theta_low']
    alpha = pd.Series(0.85 * np.ones(num_subs), index = np.arange(1, num_subs + 1))
    # =============================================================================
    # phi_ub = b + num_mps
    # phi_lb = 0
    # =============================================================================
    gamma_ub = 1
    gamma_lb = 0
    # =============================================================================
    # h_ub = 1
    # h_lb = 0
    # =============================================================================
    c_scalar = 0.01
    c = pd.Series(c_scalar * np.ones(num_subs), index = np.arange(1, num_subs + 1)) # 1： 0.1; 
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
# =============================================================================
#     p_eta = pd.Series(np.ones(num_scen) / num_scen)
#     p_eta.index += 1
#     alpha_L = 0.90
#     num_L_hat = L_hat.shape[0]
#     T = np.zeros((2 * num_L_hat, num_L_hat))
#     for row in np.arange(T.shape[0]):
#         T[row, row // 2] = (-1) ** row
# =============================================================================
        
        
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
    
    # parameters for decision-dependent 
    # T_S = np.identity(num_subs)
    
    # ??? function to calculate the set of values of gamma * phi
    # s = 8
    
    
    D_a_S = {s: sum(D_a[j] for j in sub_network[s]) for s in S} # demand for each sub network s
    # possible set of values for phi and gamma
    # pre_phi = {s:list(set([len(sub_network[s]) * (2 * r_num + 1 * (num - r_num)) for num in np.arange(0, len(M) + 1) for r_num in np.arange(0, num + 1) if r_num <= b and num <= h_hat])) for s in S} 
    pre_phi = {s:list(set(np.concatenate([np.arange(0, len(sub_network[s]) + 1) * (2 * r_num + 1 * (num - r_num)) for num in np.arange(0, len(M) + 1) for r_num in np.arange(0, num + 1) if r_num <= b and num <= h_hat]))) for s in S} # enumerate possible value for beta
    
    M_combi = [list(itertools.combinations(M, num)) for num in np.arange(0, h_hat + 1)]
    pre_gamma = {s: list(set([min(1, sum(Psi_a[m] for m in m_combi) / D_a_S[s]) for m_by_num in M_combi for m_combi in m_by_num])) for s in S} 
    E = {s:list(set([phi * gamma for phi in pre_phi[s] for gamma in pre_gamma[s]])) for s in S}
    # E_ind = {s: np.arange(1, len(E[s]) + 1) for s in S}
    # Phi = {s: [phi for phi in pre_phi[s]] for s in S}
    
    # ??? function to calculate p-insufficient recombination
    beta_hat, cuts, p_suff_recombs, p_insuff_recombs = return_cut_points(E = E, c = c, scen_xi = scen_xi, 
                                                                         sub_network = sub_network, alpha = alpha, q_thresh = low_quant)
    g_ind = {(s, e, j): np.arange(1, len(cuts[s, e, j]) + 1) for s in S for e in E[s] for j in sub_network[s]}
    k_ind = {(s, e): np.arange(1, len(p_insuff_recombs[s, e]) + 1) for s in S for e in E[s]}
    theta_ind = [(s, phi_enum) for s in S for phi_enum in pre_phi[s]]
    
    
    ### init model
    model_name = 'mps'
    m = gp.Model(model_name)
    m.setParam(GRB.Param.TimeLimit, 1 * 3600)
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
    f_hat = gp.MVar.fromlist(f_a_hat.tolist() + f_r_hat.tolist())
    u = m.addVars(u_ind, vtype = GRB.BINARY, name = 'u')
    
    
    
    
    # variables related to decision-dependent joint chance constraints (impacted nodes)
    phi = m.addVars(S, vtype = GRB.INTEGER, name = 'phi') # phi should be integer variables
    gamma = m.addVars(S, vtype = GRB.CONTINUOUS, name = 'gamma')
    for s in gamma:
        gamma[s].ub = 1
    # epsilon_ind = np.arange(0, num_subs * (b + num_mps) + 1)
    # epsilon = m.addVars(S, epsilon_ind, vtype = GRB.BINARY, name = 'epsilon')
    
    # h_ind = [(s, k) for s in S for k in K if k in Omega_ind[s]['up'] or k in Omega_ind[s]['down']] # index set for Mccormick variable h. Index 1, 2 corresponds to up and down scenarios?
    # h = m.addVars(h_ind, vtype = GRB.CONTINUOUS, name = 'h_mccormick') # h is bounded between 0 and 1?
    # pi_ind = [(s, k, ep_ind) for s, k in h_ind for ep_ind in epsilon_ind]
    # pi = m.addVars(pi_ind, vtype = GRB.CONTINUOUS, name = 'pi_mccormick')
    # z_bar = m.addVars(S, K, vtype = GRB.BINARY, name = 'z_bar')
    q = m.addVars(I, vtype = GRB.CONTINUOUS, name = 'q')
    nu_ind = [(s, e) for s in S for e in E[s]]
    nu = m.addVars(nu_ind, vtype = GRB.BINARY, name = 'nu')
    mu_ind = [(s, e, j, g) for s in S for e in E[s] for j in sub_network[s] for g in g_ind[s, e, j]]
    mu = m.addVars(mu_ind, vtype = GRB.BINARY, name = 'mu')
    zeta = m.addVars(mu_ind, vtype = GRB.BINARY, name = 'zeta')
    theta = m.addVars(theta_ind, vtype = GRB.BINARY, name = 'theta')
    lambda_var = m.addVars(theta_ind, vtype = GRB.CONTINUOUS, name = 'lambda')
    # ? V_s_ind = [(s, j) for s in S for j in sub_network[s]]
    # ? v_s = m.addVars(V_s_ind, vtype = GRB.CONTINUOUS, name = 'v^S')
    
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
    
    m.addConstrs((sum(z_a[m, i, j] for m in M for i in I_c for j in I_i[i] if i in sub_network[s]) == sum(D_a[j] for j in sub_network[s]) * gamma[s] for s in S), name = 'gamma_def')
    
    """
    s = 8
    right = sum(D_a[j] for j in sub_network[s]) 
    gamma[s].getValue()
    right.getValue()
    
    
    
    left = sum(z_a[m, i, j] for m in M for i in I_c if j in I_i[i] if i in sub_network[s])
    """
    
    
    m.addConstrs((sum(2 * beta[m, j, 'r'] + beta[m, j, 'u'] for j in sub_network[s] for m in M) == phi[s] for s in S), name = 'phi_def')
    
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
    
    # m.addConstrs((beta[m, j, 'r'] <= sum(z_a[m, i, j] for i in I_c if j in I_i[i]) for m in M for j in I_mps), name = 'beta_r_z')
    # m.addConstrs((beta[m, j, 'u'] <= sum(z_a[m, i, j] for i in I_c if j in I_i[i]) for m in M for j in I_mps), name = 'beta_u_z')
    
    
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
    ##### decision-dependent joint constraints
    ##############################################################
    
    # this set is infeasible
    m.addConstrs((q[j] >= sum(cuts[s, e, j][g - 1] * zeta[s, e, j, g] for e in E[s] for g in g_ind[s, e, j]) for s in S for j in sub_network[s]), name = 'q>=sum_c_zeta') # g - 1 for position index
    
    """
    s = 8
    j = 31
    rh = q[j].X
    lh = sum(cuts[s, e, j][g - 1] * zeta[s, e, j, g] for e in E[s] for g in g_ind[s, e, j])
    lh.getValue()
    
    nu_sol = pd.Series(nu.values(), index = nu.keys())
    nu_nz_ind = [True if row.X != 0 else False for row in nu_sol]
    print(nu_sol[nu_nz_ind])
    
    mu_sol = pd.Series(mu.values(), index = mu.keys())
    mu_nz_ind = [True if row.X != 0 else False for row in mu_sol]
    print(mu_sol[mu_nz_ind])
    
    
    
    """
    
    m.addConstrs((sum(beta_hat[s, e, j, k, g] * zeta[s, e, j, g] for j in sub_network[s] for g in g_ind[s, e, j]) <= len(sub_network[s]) - 1 for s in S for e in E[s] for k in k_ind[s, e]), name = 'sum_nu_beta_mu<|J|-1') # g - 1 for position index
    
    # this set is infeasible
    m.addConstrs((sum(mu[s, e, j, g] for e in E[s] for g in g_ind[s, e, j])  == 1 for s in S for j in sub_network[s]), name = 'sum_mu=1')
    
    m.addConstrs((sum(mu[s, e, j, g] for e in E[s] for g in g_ind[s, e, j])  == 1 for s in S for j in sub_network[s]), name = 'sum_mu=1')
    m.addConstrs((sum(nu[s, e] for e in E[s]) == 1 for s in S), name = 'sum_nu=1')
    
    # do i need this constraint???
    m.addConstrs((mu[s, e, j, g] <= nu[s, e] for s in S for e in E[s] for j in sub_network[s] for g in g_ind[s, e, j]), name = 'mu<=nu')
    # end of infeasible set
    
    m.addConstrs((phi[s] == sum(phi_enum * theta[s, phi_enum] for phi_enum in pre_phi[s]) for s in S), name = 'phi=sum_theta')
    m.addConstrs((sum(theta[s, phi_enum] for phi_enum in pre_phi[s]) == 1 for s in S), name = 'sum=theta=1')
    m.addConstrs((sum(e * nu[s, e] for e in E[s]) == sum(phi_enum * lambda_var[s, phi_enum] for phi_enum in pre_phi[s]) for s in S), name = 'sum_e_nu=sum_phi_enum*lambda')
    
    m.addConstrs((zeta[s, e, j, g] >= 0 for s in S for e in E[s] for j in sub_network[s] for g in g_ind[s, e, j]), name = 'mc_zeta_1')
    m.addConstrs((zeta[s, e, j, g] >= mu[s, e, j, g] + nu[s, e] - 1 for s in S for e in E[s] for j in sub_network[s] for g in g_ind[s, e, j]), name = 'mc_zeta_2')
    m.addConstrs((zeta[s, e, j, g] <= mu[s, e, j, g] for s in S for e in E[s] for j in sub_network[s] for g in g_ind[s, e, j]), name = 'mc_zeta_3')
    m.addConstrs((zeta[s, e, j, g] <= nu[s, e] for s in S for e in E[s] for j in sub_network[s] for g in g_ind[s, e, j]), name = 'mc_zeta_4')
    
    m.addConstrs((lambda_var[s, phi_enum] >= gamma_lb * theta[s, phi_enum] for s, phi_enum in theta_ind), name = 'mc_lambda_1')
    m.addConstrs((lambda_var[s, phi_enum] >= gamma_ub * (theta[s, phi_enum] - 1) + gamma[s] for s, phi_enum in theta_ind), name = 'mc_lambda_2')
    m.addConstrs((lambda_var[s, phi_enum] <= gamma_ub * theta[s, phi_enum] for s, phi_enum in theta_ind), name = 'mc_lambda_3')
    m.addConstrs((lambda_var[s, phi_enum] <= gamma_lb * (theta[s, phi_enum] - 1) + gamma[s] for s, phi_enum in theta_ind), name = 'mc_lambda_4')
    
    ### setting objective functions
    m.setObjective(sum(q[j] for j in I), GRB.MINIMIZE)
    
    # m.setObjective(sum(z_a[m, i, j] for m in M for i in I_c for j in I if j in I_i[i] ), GRB.MAXIMIZE)
    
    ### testing
    # =============================================================================
    # q[1].ub =0.0
    # q[2].ub =0.0
    # q[3].ub =315.0
    # q[4].ub =420.0
    # q[5].ub =210.0
    # q[6].ub =300.0
    # q[7].ub =1000.0
    # q[8].ub =1000.0
    # q[9].ub =300.0
    # q[10].ub =300.0
    # q[11].ub =225.0
    # q[12].ub =300.0
    # q[13].ub =300.0
    # q[14].ub =600.0
    # q[15].ub =300.0
    # q[16].ub =300.0
    # q[17].ub =300.0
    # q[18].ub =450.0
    # q[19].ub =450.0
    # q[20].ub =315.0
    # q[21].ub =315.0
    # q[22].ub =450.0
    # q[23].ub =108.0
    # q[24].ub =504.0
    # q[25].ub =504.0
    # q[26].ub =300.0
    # q[27].ub =300.0
    # q[28].ub =300.0
    # q[29].ub =600.0
    # q[30].ub =1000.0
    # q[31].ub =750.0
    # q[32].ub =1050.0
    # q[33].ub =300.0
    # 
    # q[1].lb =0.0
    # q[2].lb =0.0
    # q[3].lb =315.0
    # q[4].lb =420.0
    # q[5].lb =210.0
    # q[6].lb =300.0
    # q[7].lb =1000.0
    # q[8].lb =1000.0
    # q[9].lb =300.0
    # q[10].lb =300.0
    # q[11].lb =225.0
    # q[12].lb =300.0
    # q[13].lb =300.0
    # q[14].lb =600.0
    # q[15].lb =300.0
    # q[16].lb =300.0
    # q[17].lb =300.0
    # q[18].lb =450.0
    # q[19].lb =450.0
    # q[20].lb =315.0
    # q[21].lb =315.0
    # q[22].lb =450.0
    # q[23].lb =108.0
    # q[24].lb =504.0
    # q[25].lb =504.0
    # q[26].lb =300.0
    # q[27].lb =300.0
    # q[28].lb =300.0
    # q[29].lb =600.0
    # q[30].lb =1000.0
    # q[31].lb =750.0
    # q[32].lb =1050.0
    # q[33].lb =300.0
    # =============================================================================
    
    
    # =============================================================================
    # j = 30
    # temp_expr_1 = Omega_down_ind[j].shape[0] * (sum(p_b[k] * z_tilde[j, k] for k in K if k in Omega_0_ind[j]) + sum(p_b[k] * z_tilde[j, k] for k in K if k in Omega_up_ind[j]) + sum(p_b[k] * c[j] * sum(n * pi[j, k, n] for n in N_ind) for k in K if k in Omega_up_ind[j])) 
    # temp_expr_1.getValue()
    # 
    # temp_expr_2 = (1 - sum(p_b[k] for k in K if k in Omega_0_ind[j])) * sum(z_tilde[j, k] for k in K if k in Omega_down_ind[j]) 
    # temp_expr_2.getValue()
    # 
    # temp_expr_3 = sum(p_b[k] for k in K if k in Omega_up_ind[j]) * sum(z_tilde[j, k] for k in K if k in Omega_down_ind[j]) 
    # temp_expr_3.getValue()
    # 
    # temp_expr_4 = sum(p_b[k_prime] for k_prime in K if k_prime in Omega_up_ind[j]) * sum(c[j] * sum(n * pi[j, k, n] for n in N_ind) for k in K if k in Omega_down_ind[j])
    # temp_expr_4.getValue()                  
    # =============================================================================
    
    ###
    # =============================================================================
    # y[1,6].lb = 1.0
    # y[2,6].lb = 1.0
    # y[3,25].lb = 1.0
    # y[4,6].lb = 1.0
    # y[5,25].lb = 1.0
    # y[6,25].lb = 1.0
    # 
    # x[3].lb = 1.0
    # x[5].lb = 1.0
    # x[6].lb = 1.0
    # 
    # # gamma[6].lb = 1
    # gamma[2].lb = 1
    # phi[2].lb = 12
    # phi[2].ub = 12
    # phi[3].lb = 24
    # phi[3].ub = 24
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
    print(f'c is {c}')
    print(f'alpha is {alpha}')
    
    q_sol = pd.Series(q.values(), index = q.keys())
    print(q_sol)
    
    for s in S:
        print('===============================')
        print(f'sub network {s} includes nodes {sub_network[s]}')
        for j in sub_network[s]:
            print(f'optimal ens for node {j} is {q[j]}')
        print('===============================')
        
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
    
    print('The index of f starts from 0 in Python result for this project')
    print('So f[0] here corresponds to f[1] in overleaf')
    print(f_a)
    
    print('The index of f starts from 0 in Python result for this project')
    print('So f[0] here corresponds to f[1] in overleaf')
    print(f_r)
    
    # =============================================================================
    # z_bar_sol = pd.Series(z_bar.values(), index = z_bar.keys())
    # z_bar_nz_ind = [True if row.X != 0 else False for row in z_bar_sol]
    # print(z_bar_sol[z_bar_nz_ind])
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
    
    # =============================================================================
    # pi_sol = pd.Series(pi.values(), index = pi.keys())
    # pi_nz_ind = [True if row.X != 0 else False for row in pi_sol]
    # print(pi_sol[pi_nz_ind])
    # =============================================================================
    # print(pi_sol[pi_nz_ind][30])
    
    v_sol = pd.Series(v.values(), index = v.keys())
    v_nz_ind = [True if row.X != 0 else False for row in v_sol]
    print(v_sol[v_nz_ind])
    
    theta_sol = pd.Series(theta.values(), index = theta.keys())
    theta_nz_ind = [True if row.X != 0 else False for row in theta_sol]
    print(theta_sol[theta_nz_ind])
    
    nu_sol = pd.Series(nu.values(), index = nu.keys())
    nu_nz_ind = [True if row.X != 0 else False for row in nu_sol]
    print(nu_sol[nu_nz_ind])



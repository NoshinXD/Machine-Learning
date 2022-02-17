import os
import numpy as np
import math
# import random
from hmmlearn import hmm

np.random.seed(10)

class global_vars:
    in_filename1 = ""
    in_filename2 = ""
    check_filename1 = ""

    out_filename1 = ""
    out_filename2 = ""
    out_filename3 = ""

    T_years = -1
    num_of_states = -1

    line_counter = 0


global_vars_obj = global_vars()

state_dict = {0: "El Nino", 1: "La Nina"}
# print("\"" + str(state_dict[0]) + "\"")

path = os.getcwd()
in_path = path + "/Input/"
global_vars_obj.in_filename1 = in_path + "data.txt"
global_vars_obj.in_filename2 = in_path + "parameters.txt.txt"

check_path = os.getcwd() + "/Output/"
global_vars_obj.check_filename1 = check_path + "states_Viterbi_wo_learning.txt"
# print(global_vars_obj.check_filename1)

# out_path = path + "/my_output/"
out_path = ""
global_vars_obj.out_filename1 = out_path + "1605071_states_Viterbi_wo_learning.txt"
global_vars_obj.out_filename2 = out_path + "1605071_states_Viterbi_after_learning.txt"
global_vars_obj.out_filename3 = out_path + "1605071_parameters_learned.txt.txt"

f2 = open(global_vars_obj.out_filename1, "a")
f2.truncate(0)

f3 = open(global_vars_obj.out_filename2, "a")
f3.truncate(0)

f4 = open(global_vars_obj.out_filename3, "a")
f4.truncate(0)

# ################################################## get_input
def get_input():
    global_vars_obj.line_counter = 0
    with open(global_vars_obj.in_filename1) as f:
        lines = f.readlines()

    observation_list = []

    for line in lines:
        temp_f = float(line)
        observation_list.append(temp_f)

    global_vars_obj.T_years = len(observation_list)

    with open(global_vars_obj.in_filename2) as f:
        lines = f.readlines()

    global_vars_obj.num_of_states = int(lines[0])

    rows, cols = (global_vars_obj.num_of_states, global_vars_obj.num_of_states)
    transition_matrix = [0] * rows
    for i in range(rows):
        transition_matrix[i] = [0] * cols

    for i in range(global_vars_obj.num_of_states):
        global_vars_obj.line_counter = global_vars_obj.line_counter + 1
        line_list = lines[i+1].rstrip().split()
        
        for j in range(global_vars_obj.num_of_states):
            transition_matrix[i][j] = float(line_list[j])


    global_vars_obj.line_counter = global_vars_obj.line_counter + 1
    mean_list = lines[global_vars_obj.line_counter].rstrip().split()
    mean_list = [float(i) for i in mean_list]

    global_vars_obj.line_counter = global_vars_obj.line_counter + 1
    variance_list = lines[global_vars_obj.line_counter].rstrip().split()
    variance_list = [float(i) for i in variance_list]

    std_list = [math.sqrt(i) for i in variance_list]

    return observation_list, transition_matrix, mean_list, std_list

def get_a_random_decimal_number():
    # U = random.uniform(0, 1)
    U = np.random.uniform(low = 0.0000001, high = 1000)
    return U

def get_random_params():
    rows, cols = (global_vars_obj.num_of_states, global_vars_obj.num_of_states)
    transition_matrix = [0] * rows
    for i in range(rows):
        transition_matrix[i] = [0] * cols

    mean_list = [0] * global_vars_obj.num_of_states
    std_list = [0] * global_vars_obj.num_of_states

    # U = random.uniform(0, 1)
    for i in range(global_vars_obj.num_of_states):
        for j in range(global_vars_obj.num_of_states):
            transition_matrix[i][j] = get_a_random_decimal_number()

    trans_sum = np.sum(transition_matrix, axis = 1)
    transition_matrix = transition_matrix / trans_sum[:, np.newaxis]

    for i in range(global_vars_obj.num_of_states):
        mean_list[i] = get_a_random_decimal_number()

    for i in range(global_vars_obj.num_of_states):
        std_list[i] = get_a_random_decimal_number()

    return transition_matrix, mean_list, std_list

# ################################################## helping function
def normal_probability_distribution(x, mean, sd):
    coeff = 1 / (sd * math.sqrt(2 * math.pi))
    power_of_e = ((x - mean) / sd) ** 2
    res = coeff * math.exp(-0.5 * power_of_e)
    return res

# 10000 -> 283, 10600 -> 260, 15600 -> 143,  1000 -> 558

def get_stationary_distribution_matmul(x): # Repeated Matrix Multiplication
    steps = 10**3
    y = x

    i=0
    while i<steps:
        y =  np.matmul(y, x)
        i = i + 1

        if np.isclose(y, y[0]).all(): # Returns a boolean array where two arrays are element-wise equal within a tolerance
            # print(i)
            break

        # pass

    # print("y = \n", y, "\n")
    # print("π = ", y[0])
    # no need to normalize still 
    y_sum = np.sum(y[0])
    y[0] = y[0] / y_sum

    return y[0]

def get_stationary_distribution_using_eqn(a, b):
    x = np.linalg.solve(a, b)
    return x

def get_stationary_distribution_using_eqn_2(p_mat):
    (rows, cols) = (len(p_mat), len(p_mat))
    a = [0] * rows
    for i in range(rows):
        a[i] = [0] * cols
    a = np.array(a).astype(float)

    b = [0] * rows
    b = np.array(b).astype(float)

    for i in range(rows):
        for j in range(cols):
            if i == rows - 1:
                a[i][j] = 1
            else:
                if j == i:
                    a[i][j] = p_mat[j][i] - 1
                else:
                    a[i][j] = p_mat[j][i]

    # print(a)
    for i in range(rows):
        if i == rows - 1:
            b[i] = 1.0
        else:
            b[i] = 0.0

    # print(b)
    x = np.linalg.solve(a, b)
    # print(x)
    return x 

# ################################################## task function
# --------------------------------------------------- viterbi
def viterbi_algo(y_list, a_list, m_list, s_list, pi_list):
    # print(y_list)
    y_list = np.array(y_list)
    a_list = np.array(a_list)
    pi_list = np.array(pi_list)
    
    rows, cols = (global_vars_obj.num_of_states, global_vars_obj.T_years)
    T1 = [0] * rows
    for i in range(rows):
        T1[i] = [0] * cols
    T1 = np.array(T1).astype(float)

    T2 = [0] * rows
    for i in range(rows):
        T2[i] = [0] * cols
    T2 = np.array(T2).astype(int)

    # for the first observation
    for i in range(global_vars_obj.num_of_states):
        # T1[i, 0] = pi_list[i] * b_list[i, 0] # proba of state i hishebe eta use hbe next e
        b_i_y0 = normal_probability_distribution(y_list[0], m_list[i], s_list[i])
        T1[i, 0] = np.log(pi_list[i]) + np.log(b_i_y0) # proba of state i hishebe eta use hbe next e
        T2[i, 0] = -1

    # for other observations
    for x in range(global_vars_obj.T_years - 1):
        j = x + 1
        for i in range(global_vars_obj.num_of_states):
            # T1[i, j] = np.max(T1[:, j-1] * a_list[:, i] * b_list[i, j])
            # T2[i, j] = np.argmax(T1[:, j-1] * a_list[:, i] * b_list[i, j])
            b_i_yj = normal_probability_distribution(y_list[j], m_list[i], s_list[i])
            T1[i,j] = np.max(T1[:, j-1] + np.log(a_list[:, i]) + np.log(b_i_yj))
            T2[i,j] = np.argmax(T1[:, j-1] + np.log(a_list[:, i]) + np.log(b_i_yj))

    z_list = [-1] * global_vars_obj.T_years
    x_list = [-1] * global_vars_obj.T_years

    z_list[global_vars_obj.T_years-1] = np.argmax(T1[:, global_vars_obj.T_years-1])
    x_list[global_vars_obj.T_years-1] = "\"" + str(state_dict[z_list[global_vars_obj.T_years-1]]) + "\"" 

    for j in range(global_vars_obj.T_years-1, 0, -1):
        z_list[j-1] = T2[z_list[j], j]
        x_list[j-1] = "\"" + str(state_dict[z_list[j-1]]) + "\"" 

    # print(x_list)
    return x_list

# --------------------------------------------------- baum-welch
def forward_procedure(y_list, a_list, m_list, s_list, pi_list):
    rows, cols = (global_vars_obj.num_of_states, global_vars_obj.T_years)
    alpha_list = [0] * rows
    for i in range(rows):
        alpha_list[i] = [0] * cols
    alpha_list = np.array(alpha_list).astype(float)

    # for the first observation
    alpha_sum = 0
    for i in range(global_vars_obj.num_of_states):
        b_i_y0 = normal_probability_distribution(y_list[0], m_list[i], s_list[i])
        alpha_list[i, 0] = pi_list[i] * b_i_y0 # proba of state i hishebe eta use hbe next e
        alpha_sum = alpha_sum + alpha_list[i, 0]

    # normalizing
    for i in range(global_vars_obj.num_of_states):
        alpha_list[i, 0] = alpha_list[i, 0] / alpha_sum

    # for the other observations 
    for x in range(global_vars_obj.T_years - 1):
        alpha_sum = 0
        t = x + 1
        for i in range(global_vars_obj.num_of_states):
            b_i_yt = normal_probability_distribution(y_list[t], m_list[i], s_list[i])
            alpha_list[i, t] = np.sum(alpha_list[:, t-1] * a_list[:, i]) * b_i_yt
            alpha_sum = alpha_sum + alpha_list[i, t]
        
        # normalizing
        for i in range(global_vars_obj.num_of_states):
            alpha_list[i, t] = alpha_list[i, t] / alpha_sum

    # print("alpha: ")
    # print(alpha_list)

    return alpha_list

def backward_procedure(y_list, a_list, m_list, s_list,):
    # print(m_list)
    # print(s_list)
    rows, cols = (global_vars_obj.num_of_states, global_vars_obj.T_years)
    beta_list = [0] * rows
    for i in range(rows):
        beta_list[i] = [0] * cols
    beta_list = np.array(beta_list).astype(float)

    # for the last observation
    beta_sum = 0
    for i in range(global_vars_obj.num_of_states):
        beta_list[i, global_vars_obj.T_years - 1] = 1.0
        beta_sum = beta_sum + beta_list[i, global_vars_obj.T_years - 1]

    # normalizing
    for i in range(global_vars_obj.num_of_states):
        beta_list[i, global_vars_obj.T_years - 1] = beta_list[i, global_vars_obj.T_years - 1] / beta_sum


    for x in range(global_vars_obj.T_years - 1, 0, -1):
        beta_sum = 0
        t = x
        for i in range(global_vars_obj.num_of_states):
            for j in range(global_vars_obj.num_of_states):
                b_j_yt = normal_probability_distribution(y_list[t], m_list[j], s_list[j])
                beta_list[i, t-1] = beta_list[i, t-1] + beta_list[j, t] * a_list[i, j] * b_j_yt
                beta_sum = beta_sum + beta_list[i, t-1]

        for i in range(global_vars_obj.num_of_states):
            beta_list[i, t-1] = beta_list[i, t-1] / beta_sum


    # print("beta:")
    # print(beta_list)
    # if global_vars_obj.while_counter == 1:
    #     print(beta_list)

    return beta_list

# def get_gamma_list(alpha_list, beta_list):
#     denominator_list = np.sum(np.multiply(alpha_list, beta_list), axis = 0)

#     gamma_list = np.multiply(np.multiply(alpha_list, beta_list), 1/denominator_list)
    
#     # print(gamma_list)
#     # if global_vars_obj.while_counter == 2:
#     #     print(gamma_list)

#     return gamma_list


def get_gamma_list(alpha_list, beta_list):
    gamma_list = np.multiply(alpha_list, beta_list)
    gamma_sum = np.sum(gamma_list, axis = 0)

    gamma_list = np.multiply(gamma_list, 1/gamma_sum)
    
    # print(gamma_list)
    # if global_vars_obj.while_counter == 2:
    #     print(gamma_list)

    return gamma_list



def get_epsilon_list(y_list, a_list, m_list, s_list, alpha_list, beta_list):
    epsilon_list = np.zeros((global_vars_obj.T_years - 1, global_vars_obj.num_of_states, global_vars_obj.num_of_states)) 

    for t in range(global_vars_obj.T_years - 1):
        for i in range(global_vars_obj.num_of_states):
            for j in range(global_vars_obj.num_of_states):
                b_j = normal_probability_distribution(y_list[t+1], m_list[j], s_list[j])
                epsilon_list[t, i, j] = alpha_list[i, t] * a_list[i, j] * beta_list[j, t+1] * b_j

    for t in range(global_vars_obj.T_years - 1):
        epsilon_sum = 0
        for i in range(global_vars_obj.num_of_states):
            for j in range(global_vars_obj.num_of_states):
                epsilon_sum = epsilon_sum + epsilon_list[t, i, j]

        for i in range(global_vars_obj.num_of_states):
            for j in range(global_vars_obj.num_of_states):
                epsilon_list[t, i, j] = epsilon_list[t, i, j] / epsilon_sum
    
    # if global_vars_obj.while_counter == 2:
    #     print(epsilon_list)

    # print(epsilon_list)
    return epsilon_list

def update_parameters(y_list, gamma_list, epsilon_list):
    # ---------------------------------------updating initial probability
    # next_pi_list = gamma_list[:, 0]
    # pi_sum = np.sum(next_pi_list)
    # next_pi_list = np.multiply(next_pi_list, 1/pi_sum)

    # print(next_pi_list)

    rows, cols = (global_vars_obj.num_of_states, global_vars_obj.num_of_states)

    next_a_list = [0] * rows
    for i in range(rows):
        next_a_list[i] = [0] * cols
    next_a_list = np.array(next_a_list).astype(float)

    next_mean_list = [0] * global_vars_obj.num_of_states
    next_std_list =  [0] * global_vars_obj.num_of_states

    # ---------------------------------------updating transition matrix
    for i in range(global_vars_obj.num_of_states):
        for j in range(global_vars_obj.num_of_states):
            numerator = 0
            denominator = 0
            for t in range(global_vars_obj.T_years - 1):
                numerator = numerator + epsilon_list[t, i, j]
                denominator = denominator + gamma_list[i, t]
            # next_a_list[i, j] = numerator / denominator
            next_a_list[i, j] = numerator

    a_sum = np.sum(next_a_list, axis = 1)
    next_a_list = next_a_list / a_sum[:, np.newaxis] # np.newaxis increase the dimension of the existing array by one more dimension, when used once

    # print(next_a_list)

    ni = np.sum(gamma_list, axis = 1)

    # ---------------------------------------updating mean
    for i in range(global_vars_obj.num_of_states):
        for j in range(global_vars_obj.T_years):
            next_mean_list[i] = next_mean_list[i] + gamma_list[i, j] * y_list[j]

    next_mean_list = next_mean_list / ni

    # print(next_mean_list)
    # ---------------------------------------updating variance
    for i in range(global_vars_obj.num_of_states):
        for j in range(global_vars_obj.T_years):
            next_std_list[i] = next_std_list[i] + gamma_list[i, j] * ((y_list[j] - next_mean_list[i]) ** 2)

    next_std_list = np.sqrt(next_std_list / ni)
    # print(next_std_list)
    # next_variance_list = np.square(next_std_list)
    # print(next_variance_list)

    return next_a_list, next_mean_list, next_std_list

# def baum_welch_algo_2(y_list, a_list, m_list, s_list, pi_list):
#     # print("hello from baum-welch")
#     y_list = np.array(y_list)
#     a_list = np.array(a_list)
#     pi_list = np.array(pi_list)

#     # print(y_list)
#     # print(a_list)
#     # print(b_list)
#     # print(pi_list)
#     global_vars_obj.while_counter = 0
#     while True:
#         global_vars_obj.while_counter = global_vars_obj.while_counter + 1

#         # print("iter: " + str(global_vars_obj.while_counter - 1))
#         # print(pi_list)
#         # print(a_list)
#         # print(m_list)
#         # print(s_list)
#         # print("--------------")

#         alpha_list = forward_procedure(y_list, a_list, m_list, s_list, pi_list)
#         beta_list = backward_procedure(y_list, a_list, m_list, s_list) 
#         gamma_list = get_gamma_list(alpha_list, beta_list)
#         # print(gamma_list)
#         epsilon_list = get_epsilon_list(y_list, a_list, m_list, s_list, alpha_list, beta_list)

#         next_a_list, next_mean_list, next_std_list = update_parameters(y_list, gamma_list, epsilon_list)

#         next_pi_list = get_stationary_distribution_matmul(next_a_list)
        
#         pi_list = next_pi_list
#         a_list = next_a_list
#         m_list = next_mean_list
#         s_list = next_std_list

#         if global_vars_obj.while_counter == 5:
#             print("here2")
#             # print(pi_list)
#             # print(a_list)
#             # print(m_list)
#             # # print(s_list)
#             # v_list = np.square(s_list)
#             # print(v_list)
#             break

#     v_list = np.square(s_list)
#     return pi_list, a_list, m_list, v_list


def baum_welch_algo(y_list, a_list, m_list, s_list, pi_list):
    # print("hello from baum-welch")
    # print(y_list)
    # print(a_list)
    # print(b_list)
    # print(pi_list)
    global_vars_obj.while_counter = 0
    y_list = np.array(y_list)
    while True:
        a_list = np.array(a_list)
        pi_list = np.array(pi_list)

        global_vars_obj.while_counter = global_vars_obj.while_counter + 1
        
        alpha_list = forward_procedure(y_list, a_list, m_list, s_list, pi_list)
        beta_list = backward_procedure(y_list, a_list, m_list, s_list) 
        gamma_list = get_gamma_list(alpha_list, beta_list)
        epsilon_list = get_epsilon_list(y_list, a_list, m_list, s_list, alpha_list, beta_list)

        # if global_vars_obj.while_counter == 1:
        #     print("for itertion: " + str(global_vars_obj.while_counter-1))
        #     print("aplha")
        #     print(alpha_list)
        #     print()
        #     print("beta")
        #     print(beta_list)
        #     print()
        #     print("gamma")
        #     print(gamma_list)
        #     print()
        #     print("epsilon")
        #     print(epsilon_list)

        next_a_list, next_mean_list, next_std_list = update_parameters(y_list, gamma_list, epsilon_list)

        next_pi_list = get_stationary_distribution_matmul(next_a_list)
        # next_pi_list = get_stationary_distribution_using_eqn_2(next_a_list)
        # p = np.array([[(next_a_list[0,0]) - 1, next_a_list[1,0]], [1, 1]])
        # q = np.array([0, 1])
        # next_pi_list = get_stationary_distribution_using_eqn(p, q)

        # equal_check_pi = np.isclose(pi_list, next_pi_list)
        equal_check_a = np.isclose(a_list, next_a_list)
        equal_check_m = np.isclose(m_list, next_mean_list)
        equal_check_s = np.isclose(s_list, next_std_list)

        # converged_pi = np.all(equal_check_pi)
        converged_a = np.all(equal_check_a)
        converged_m = np.all(equal_check_m)
        converged_s = np.all(equal_check_s)

        if converged_a == True and converged_m == True and converged_s == True:
            print("converged at iter: " + str(global_vars_obj.while_counter - 1))
            # print(pi_list)
            # print(a_list)
            # print(m_list)
            # # print(s_list)
            # v_list = np.square(s_list)
            # print(v_list)
            break
        else:
            pi_list = next_pi_list
            a_list = next_a_list
            m_list = next_mean_list
            s_list = next_std_list

        if global_vars_obj.while_counter == 1000:
            print("not converged:")
            # print(pi_list)
            # print(a_list)
            # print(m_list)
            # # print(s_list)
            # v_list = np.square(s_list)
            # print(v_list)
            break

    v_list = np.square(s_list)
    return pi_list, a_list, m_list, v_list


def write_params(a_list, m_list, v_list, pi_list):
    f4.write(str(global_vars_obj.num_of_states) + "\n")

    for i in range(global_vars_obj.num_of_states):
        for j in range(global_vars_obj.num_of_states):
            f4.write(str(a_list[i, j]) + "\t")
        f4.write("\n")
    
    for i in range(global_vars_obj.num_of_states):
        f4.write(str(m_list[i]) + "\t")
    f4.write("\n")

    for i in range(global_vars_obj.num_of_states):
        f4.write(str(v_list[i]) + "\t")
    f4.write("\n")

    for i in range(global_vars_obj.num_of_states):
        f4.write(str(pi_list[i]) + "\t")
    f4.write("\n")


def get_hmm_result():
    observation_list, transition_matrix, mean_list, std_list = get_input()
    # get_input()
    variance_list = np.square(std_list)
    initial_probability_list = get_stationary_distribution_matmul(transition_matrix)

    # print(transition_matrix)
    # print(mean_list)
    # variance_list = np.square(std_list)
    # print(variance_list)
    observation_list = np.reshape(observation_list, (-1, 1))

    model = hmm.GaussianHMM(n_components=global_vars_obj.num_of_states, covariance_type="full", init_params='')

    model.startprob_ = np.array(initial_probability_list)
    model.transmat_ = np.array(transition_matrix)
    model.means_ = np.array(mean_list).reshape(global_vars_obj.num_of_states, 1)
    model.covars_ = np.array(variance_list).reshape(global_vars_obj.num_of_states, 1, 1)

    model.fit(observation_list)

    print("params from hmm:")
    print(model.transmat_)
    print(model.means_)
    print(model.covars_)
    print(model.startprob_)

    Z = model.predict(observation_list)

    # comment out the rest portion of wants to write in a file
    # check_path = path + "/check_hmm/"
    # check_filename1 = check_path + "from_hmm.txt"
    
    # f_c = open(check_filename1, "a")
    # f_c.truncate(0)

    # for i in range(len(Z)):
    #     f_c.write(str(Z[i]) + "\n")
    #     # if Z[i] == 0:
    #     #     f_c.write("\"El Nino\"\n")
    #     # else:
    #     #     f_c.write("\"La Nina\"\n")
    #     # pass

def my_main_func():
    # ---------------------- viterbi
    observation_list, transition_matrix, mean_list, std_list = get_input()
    initial_probability_list = get_stationary_distribution_matmul(transition_matrix)

    x_list = viterbi_algo(observation_list, transition_matrix, mean_list, std_list, initial_probability_list)

    for i in range(len(x_list)):
        f2.write(x_list[i] + "\n")
        pass

    # ---------------------- baum-welch
    transition_matrix, mean_list, std_list = get_random_params()
    initial_probability_list = get_stationary_distribution_matmul(transition_matrix)

    # print("--------")
    # print(transition_matrix)
    # print(initial_probability_list)
    # print(mean_list)
    # print(std_list)
    # print("--------")

    new_initial_probability_list, new_transition_matrix, new_mean_list, new_variance_list = baum_welch_algo(observation_list, transition_matrix, mean_list, std_list, initial_probability_list)
    print(new_transition_matrix)
    print(new_mean_list)
    print(new_variance_list)
    print(new_initial_probability_list)
    
    write_params(new_transition_matrix, new_mean_list, new_variance_list, new_initial_probability_list)

    new_std_list = [math.sqrt(i) for i in new_variance_list]
    new_x_list = viterbi_algo(observation_list, new_transition_matrix, new_mean_list, new_std_list, new_initial_probability_list)

    for i in range(len(new_x_list)):
        f3.write(new_x_list[i] + "\n")


my_main_func()
# get_hmm_result()
import numpy as np
from matplotlib import pyplot as plt
import math as m
import random as random

#DEFINICIÓN DE FUNCIONES

def get_design_mtx(input_data_vec, target_data_vec, param_nmbr, func_type):
    
    Phi_mtx = []
    
    if func_type == 'pol':
        input_data_min = np.min(input_data_vec)
        input_data_range = np.max(input_data_vec)-np.min(input_data_vec)    
        input_data_vec_resc = []
        for i in range(len(input_data_vec)):
            input_data_resc = (input_data_vec[i]-input_data_min)/\
                              (input_data_range)
            input_data_vec_resc.append(input_data_resc)
            
    for i in range(len(input_data_vec)):
        Phi_mtx.append([])
        for j in range(param_nmbr):
            if func_type == 'pol':
                Phi_ij = input_data_vec_resc[i]**j
            if func_type == 'gauss':
                x_min = np.min(input_data_vec)
                x_max = np.max(input_data_vec)
                delta_x = (x_max - x_min)/param_nmbr
                mu_j = x_min + 0.5*delta_x + j*delta_x
                s = delta_x/2
                Phi_ij = m.exp(-(input_data_vec[i]-mu_j)**2/(2*s**2))
            Phi_mtx[i].append(Phi_ij)

    return Phi_mtx

def get_evidence(input_data_vec, target_data_vec, param_nmbr, func_type, alpha, beta, it):

    Phi_mtx = get_design_mtx(input_data_vec, target_data_vec, param_nmbr, func_type)
    count = 0
    lambda_over_beta_vec = np.linalg.eig(np.transpose(Phi_mtx)@Phi_mtx)[0]

    while count <= it:

        count += 1
        SN_mtx = np.linalg.inv(alpha*np.eye(param_nmbr) + beta*np.transpose(Phi_mtx)@Phi_mtx)
        mN_vec = beta*SN_mtx@np.transpose(Phi_mtx)@target_data_vec
        lambda_vec = beta*lambda_over_beta_vec

        gamma = 0
        for i in range(len(lambda_vec)):
            gamma += lambda_vec[i]/(lambda_vec[i] + alpha)
            
        alpha = gamma/np.dot(mN_vec, mN_vec)

        for i in range(len(input_data_vec)):
            phi_vec = Phi_mtx[i]
            inv_beta = (target_data_vec[i] - np.dot(mN_vec, phi_vec))**2
        inv_beta = 1/(len(input_data_vec)-gamma)
        beta = 1/inv_beta
    
    A_mtx = alpha*np.eye(param_nmbr) + beta*np.transpose(Phi_mtx)@Phi_mtx
    mN_vec = beta*np.linalg.inv(A_mtx)@np.transpose(Phi_mtx)@target_data_vec

    E = 0
    
    for i in range(len(input_data_vec)):
        E += 0.5*beta*(target_data_vec[i] - (Phi_mtx@mN_vec)[i])**2
        
    E += 0.5*alpha*np.dot(mN_vec, mN_vec)

    evidence = 0.5*param_nmbr*m.log(alpha) +\
               0.5*len(input_data_vec)*m.log(beta) -E -\
               0.5*m.log(np.linalg.det(A_mtx)) -\
               0.5*len(input_data_vec)*m.log(2*m.pi)

    return evidence

def get_param_vec(input_data_vec, target_data_vec, param_nmbr, func_type):

    Phi_mtx = get_design_mtx(input_data_vec, target_data_vec, param_nmbr, func_type)
    
    if func_type == 'pol':
        input_data_min = np.min(input_data_vec)
        input_data_range = (np.max(input_data_vec)-np.min(input_data_vec))
    
        param_vec_resc = np.linalg.pinv(np.transpose(Phi_mtx)@Phi_mtx)@\
                         np.transpose(Phi_mtx)@target_data_vec

        param_vec = []
        for i in range(param_nmbr):
            param_i = 0
            for j in range(i,param_nmbr):
                param_i += param_vec_resc[j]*input_data_range**(-j)*\
                           m.factorial(j)/(m.factorial(i)*m.factorial(j-i))*\
                           (-input_data_min)**(j-i)
            param_vec.append(param_i)    
    else:
        param_vec = np.linalg.pinv(np.transpose(Phi_mtx)@Phi_mtx)@\
                    np.transpose(Phi_mtx)@target_data_vec

    return param_vec

def eval_regression(input_vec, param_vec, input_data_vec, func_type):

    output_vec = []

    if func_type == 'pol':
        for i in range(len(input_vec)):
            output = 0
            for j in range(len(param_vec)):
                output += param_vec[j]*input_vec[i]**j
            output_vec.append(output)

    if func_type == 'gauss':
        x_min = np.min(input_data_vec)
        x_max = np.max(input_data_vec)
        delta_x = (x_max - x_min)/len(param_vec)
        s = delta_x/2
        for i in range(len(input_vec)):
            output = 0
            for j in range(len(param_vec)):
                mu_j = x_min + 0.5*delta_x + j*delta_x        
                output += param_vec[j]*m.exp(-(input_vec[i]-mu_j)**2/(2*s**2))
            output_vec.append(output)

    return output_vec

def RMSE(pred_data_vec, true_data_vec):
    
    RMSE = 0
    
    for i in range(len(pred_data_vec)):
        RMSE += (pred_data_vec[i]-true_data_vec[i])**2

    RMSE = m.sqrt(RMSE/len(pred_data_vec))

    return RMSE

def bootstrap(input_data_vec, target_data_vec, set_nmbr, rep_nmbr):

    bootstrap_input_data_mtx = []
    bootstrap_target_data_mtx = []
    
    for i in range(set_nmbr):
        bootstrap_input_data_mtx.append([])
        bootstrap_target_data_mtx.append([])

    for i in range(set_nmbr):
        rep_bootstrap_idx_vec = random.sample(range(0,len(input_data_vec)), rep_nmbr)
        norep_bootstrap_idx_vec = []

        for j in range(len(input_data_vec)):
            if j not in rep_bootstrap_idx_vec:
                norep_bootstrap_idx_vec.append(j)
        bootstrap_idx_vec = norep_bootstrap_idx_vec

        for j in range(rep_nmbr):
            bootstrap_idx_vec.append(norep_bootstrap_idx_vec[j])

        for j in range(len(input_data_vec)):
            bootstrap_input_data_mtx[i].append(input_data_vec[bootstrap_idx_vec[j]])
            bootstrap_target_data_mtx[i].append(target_data_vec[bootstrap_idx_vec[j]])

    return bootstrap_input_data_mtx, bootstrap_target_data_mtx

def get_bias2_var(input_data_vec, target_data_vec, param_nmbr, func_type):
    
    alternative_input_data_mtx = bootstrap(input_data_vec, target_data_vec, 100, 100)[0]
    alternative_target_data_mtx = bootstrap(input_data_vec, target_data_vec, 100, 100)[1]

    param_vec_mtx = []
    
    for i in range(len(alternative_input_data_mtx)):
        input_vec = alternative_input_data_mtx[i]
        target_vec = alternative_target_data_mtx[i]
        param_vec_mtx.append(get_param_vec(input_vec, target_vec, param_nmbr, func_type))

    pred_vec_mtx = []
    
    for i in range(len(param_vec_mtx)):
        pred_vec_mtx.append(eval_regression(input_data_vec, param_vec_mtx[i], alternative_input_data_mtx[i], func_type))
        
    avg_pred_vec = []

    for i in range(len(input_data_vec)):
        pred = 0
        for j in range(len(param_vec_mtx)):
            pred += pred_vec_mtx[j][i]
        avg_pred_vec.append(pred/len(param_vec_mtx))

    bias2 = 0
    
    for i in range(len(input_data_vec)):
        bias2 += (avg_pred_vec[i]-target_data_vec[i])**2

    bias2 = bias2/len(input_data_vec)

    var = 0

    for i in range(len(param_vec_mtx)):
        for j in range(len(input_data_vec)):
            var += (pred_vec_mtx[i][j]-avg_pred_vec[j])**2

    var = var/len(param_vec_mtx)/len(input_data_vec)
    
    return bias2, var

#ENTRADAS

M = 3 #número de parámetros
ftype = 'pol' #tipo de funciones base: 'pol' polinómicas, 'gauss' gaussianas

'''
Índices aleatorios de las filas en el .txt de datos que serán utilizadas para entrenamiento y de
aquellas que serán utilizadas para pruebas. Estos índices fueron generados con un código independiente
para evitar que cambien cada vez que se ejecute este programa.
'''

random_train_idx = [7, 81, 73, 68, 28, 39, 98, 37, 77, 56, 3, 30, 76, 89, 51, 72, 10, 45, 80, 42,\
                    43, 66, 85, 46, 47, 60, 14, 22, 69, 35, 5, 32, 49, 99, 57, 24, 84, 83, 23, 44,\
                    53, 0, 59, 63, 91, 58, 93, 78, 96, 87, 12, 67, 50, 79, 64, 25, 19, 65, 29, 4,\
                    27, 55, 6, 92, 71, 95, 21, 26, 74, 52, 54, 13, 2, 40, 8, 38, 15, 34, 94, 9]

random_test_idx = [1, 11, 16, 17, 18, 20, 31, 33, 36, 41, 48, 61, 62, 70, 75, 82, 86, 88, 90, 97]

input_train_data_vec = []
input_test_data_vec = []

with open('dataset-1-training-Entradas.txt','r') as input_data:
    input_data = input_data.readlines()

for i in range(len(random_train_idx)):
    row = input_data[random_train_idx[i]]
    elements = row.split(' ')
    for j in range(len(elements)):
        input_train_data_vec.append(float(elements[j]))

for i in range(len(random_test_idx)):
    row = input_data[random_test_idx[i]]
    elements = row.split(' ')
    for j in range(len(elements)):
        input_test_data_vec.append(float(elements[j]))

target_train_data_vec = []
target_test_data_vec = []

with open('dataset-1-training-targets.txt','r') as target_data:
    target_data = target_data.readlines()

for i in range(len(random_train_idx)):
    row = target_data[random_train_idx[i]]
    elements = row.split(' ')
    for j in range(len(elements)):
        target_train_data_vec.append(float(elements[j]))

for i in range(len(random_test_idx)):
    row = target_data[random_test_idx[i]]
    elements = row.split(' ')
    for j in range(len(elements)):
        target_test_data_vec.append(float(elements[j]))

input_new_data_vec = []

with open('dataset-1-Entradas.txt','r') as input_data:
    input_data = input_data.readlines()

for i in range(len(random_train_idx)):
    row = input_data[random_train_idx[i]]
    elements = row.split(' ')
    for j in range(len(elements)):
        input_new_data_vec.append(float(elements[j]))

w_vec = get_param_vec(input_train_data_vec, target_train_data_vec, M, ftype)

pred_train_data_vec = eval_regression(input_train_data_vec, w_vec, input_train_data_vec, ftype)
pred_test_data_vec = eval_regression(input_test_data_vec, w_vec, input_train_data_vec, ftype)
pred_new_data_vec = eval_regression(input_new_data_vec, w_vec, input_train_data_vec, ftype)

RMSE_train_data = RMSE(pred_train_data_vec, target_train_data_vec)
RMSE_test_data = RMSE(pred_test_data_vec, target_test_data_vec)

bias2_train_data = get_bias2_var(input_train_data_vec, target_train_data_vec, M, ftype)[0]
var_train_data = get_bias2_var(input_train_data_vec, target_train_data_vec, M, ftype)[1]

evidence_train_data = get_evidence(input_train_data_vec, target_train_data_vec, M, ftype, 0.1, 10, 100)

print('RMSE training:', RMSE_train_data)
print('RMSE test:', RMSE_test_data)
print('Var:', var_train_data)
print('Bias2:', bias2_train_data)
print('log evidence:', evidence_train_data)

plt.plot(input_test_data_vec, target_test_data_vec, 'go')
plt.plot(input_test_data_vec, pred_test_data_vec)
plt.show()

plt.plot(input_new_data_vec, pred_new_data_vec, 'go')
plt.show()

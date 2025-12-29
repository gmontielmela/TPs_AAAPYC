from scipy.io import loadmat
import numpy as np
import math as m
from matplotlib import pyplot as plt
import matplotlib as mpl

#inputs
n_iter = 1000
n_train_data = 500
calculate_W_mtx = False

data = loadmat("Datos_03.mat")

T_mtx = data["t_Conjunto_1"]

Phi_mtx = data["phi_t"]

n_data = len(Phi_mtx)
n_knowndata = len(T_mtx)
n_classes = len(T_mtx[0])

if calculate_W_mtx:
    cond = False
    cond_vec = np.zeros(n_classes)
    
    '''
    indexes are suffled until there is at least one specimen for each class
    '''
    
    while not cond:
        perm = np.random.permutation(len(T_mtx))
        perm = perm[:n_train_data]
        T_train_mtx = T_mtx[perm]

        for i in range(n_classes):
            if 1 in T_mtx[:,i]:
                cond_vec[i] = 1
                
        if 0 in cond_vec:
            cond = False
        else:
            cond = True
            
    Phi_train_mtx = Phi_mtx[perm]

    RMSE_vec = np.zeros(n_iter+1)
    iter_vec = np.zeros(n_iter+1)

    W_mtx = np.linalg.inv(np.transpose(Phi_train_mtx)@Phi_train_mtx)@np.transpose(Phi_train_mtx)@T_train_mtx
    Y_mtx = np.zeros((n_train_data, n_classes))

    for i in range(n_train_data):
        Y_mtx[i] = np.tanh(np.transpose(W_mtx)@Phi_train_mtx[i])
        E = 0

        for j in range(n_classes):
            ynk = Y_mtx[i,j]
            tnk = T_train_mtx[i,j]
                
            E += 0.5*(ynk - tnk)**2

        RMSE = np.sqrt(2*E/n_train_data)
        RMSE_vec[0] = RMSE
        iter_vec[0] = 0

    for i in range(n_iter):
        for j in range(n_classes):
            wk_vec = W_mtx[:,j]
            yk_vec = Y_mtx[:,j]
            tk_vec = T_train_mtx[:,j]

            Rk_mtx = np.diag(yk_vec**2)

            zk_vec = Phi_train_mtx@wk_vec - np.linalg.inv(Rk_mtx)@(yk_vec - tk_vec)

            W_mtx[:,j] = np.linalg.inv(np.transpose(Phi_train_mtx)@Rk_mtx@Phi_train_mtx)@np.transpose(Phi_train_mtx)@Rk_mtx@zk_vec
            
        for j in range(n_train_data):
            Y_mtx[j] = np.tanh(np.transpose(W_mtx)@Phi_train_mtx[j])
            E = 0

            for k in range(n_classes):
                ynk = Y_mtx[j,k]
                tnk = T_train_mtx[j,k]
                
                E += 0.5*(ynk - tnk)**2

        RMSE = np.sqrt(2*E/n_train_data)
        RMSE_vec[i+1] = RMSE
        iter_vec[i+1] = i

    np.savetxt('W_mtx.txt', W_mtx)

    plt.plot(iter_vec, RMSE_vec)
    plt.show()

W_mtx = np.loadtxt('W_mtx.txt')

print('Parameter Matrix:\n', W_mtx)

conf_mtx = np.zeros((n_classes, n_classes))
Phi_test_mtx = Phi_mtx[:n_knowndata, :]
Y_test_mtx = np.zeros((n_knowndata, n_classes))
T_test_mtx = T_mtx

tn_vec = np.zeros(n_knowndata)
yn_vec = np.zeros(n_knowndata)
n_vec = np.zeros(n_knowndata)

for i in range(n_knowndata):
    Y_test_mtx[i] = np.tanh(np.transpose(W_mtx)@Phi_test_mtx[i])

    tn_class = np.argmax(T_test_mtx[i])
    yn_class = np.argmax(Y_test_mtx[i])

    tn_vec[i] = tn_class
    yn_vec[i] = yn_class
    
    n_vec[i] = i
    
    conf_mtx[tn_class, yn_class] += 1

print('Confusion Matrix:\n', conf_mtx)

for i in range(n_classes):
    print('----------------------------')
    print('Metrics for class',i+1)
    print('Precision:', conf_mtx[i,i]/np.sum(conf_mtx[:,i]))
    print('Recall (true positive rate):', conf_mtx[i,i]/np.sum(conf_mtx[i]))
    print('False positive rate:', (np.sum(conf_mtx[:,i])-conf_mtx[i,i])/(np.sum(conf_mtx)-np.sum(conf_mtx[i])))
    print('False discovery rate:', (np.sum(conf_mtx[:,i])-conf_mtx[i,i])/np.sum(conf_mtx[:,i]))

mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})

plt.figure(figsize=(6, 3))
plt.yticks(np.arange(0, 4, 1))
plt.xlabel('$n$')
plt.ylabel('$k$')
plt.tight_layout()

plt.plot(n_vec, tn_vec+1, 'bo', markersize=3)

n_tn_coor = np.zeros((n_knowndata,2))
n_tn_coor[:,0] = n_vec
n_tn_coor[:,1] = tn_vec+1

np.savetxt("n_tn_coor.txt", n_tn_coor) 

for i in range(n_knowndata, n_data):
    y_vec = np.tanh(np.transpose(W_mtx)@Phi_mtx[i])

    yn_class = np.argmax(y_vec)
    yn_vec = np.append(yn_vec, yn_class)
    n_vec = np.append(n_vec,i)

n_yn_coor = np.zeros((n_data,2))
n_yn_coor[:,0] = np.rint(n_vec).astype(int)
n_yn_coor[:,1] = np.rint(yn_vec+1).astype(int)

np.savetxt("n_yn_coor.txt", n_yn_coor, fmt='%d') 

plt.plot(n_vec, yn_vec+1,'r.', markersize=1)
plt.show()

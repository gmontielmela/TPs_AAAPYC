import numpy as np
from matplotlib import pyplot as plt

n_iter = 50000
lay1_activation = 'tanh'
lay2_activation = 'tanh'
eta = 0.1 #learning rate
n1 = 3 #excluding biases
n2 = 3 #excluding biases

random_train_idx = [7, 81, 73, 68, 28, 39, 98, 37, 77, 56, 3, 30, 76, 89, 51, 72, 10, 45, 80, 42,\
                    43, 66, 85, 46, 47, 60, 14, 22, 69, 35, 5, 32, 49, 99, 57, 24, 84, 83, 23, 44,\
                    53, 0, 59, 63, 91, 58, 93, 78, 96, 87, 12, 67, 50, 79, 64, 25, 19, 65, 29, 4,\
                    27, 55, 6, 92, 71, 95, 21, 26, 74, 52, 54, 13, 2, 40, 8, 38, 15, 34, 94, 9]

random_test_idx = [1, 11, 16, 17, 18, 20, 31, 33, 36, 41, 48, 61, 62, 70, 75, 82, 86, 88, 90, 97]

input_train_data_vec = []
input_test_data_vec = []

with open('dataset-2-training-Entradas.txt','r') as input_data:
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

with open('dataset-2-training-targets.txt','r') as target_data:
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

with open('dataset-2-Entradas.txt','r') as input_data:
    input_data = input_data.readlines()

for i in range(len(input_data)):
    row = input_data[i]
    elements = row.split(' ')
    for j in range(len(elements)):
        input_new_data_vec.append(float(elements[j]))

n_data = len(input_train_data_vec)
n_test_data = len(input_test_data_vec)
n_new_data = len(input_new_data_vec)

xn_vec = input_train_data_vec
tn_vec = target_train_data_vec

np.random.seed(0)
w_lay1_vec = np.random.randn(2*n1)
w_lay2_mtx = np.random.randn(n2,n1+1)
w_lay3_vec = np.random.randn(n2+1)

RMSE_vec = np.array([])
iter_vec = np.array([])

for i in range(n_iter):
    sum_dEdw_lay1_vec = np.zeros(2*n1)
    sum_dEdw_lay2_mtx = np.zeros((n2,n1+1))
    sum_dEdw_lay3_vec = np.zeros(n2+1)

    E = 0
    
    for j in range(n_data):
        a1_vec = np.zeros(n1)
        a2_vec = np.zeros(n2)

        dEdw_lay1_vec = np.zeros(2*n1)
        dEdw_lay2_mtx = np.zeros((n2,n1+1))
        dEdw_lay3_vec = np.zeros(n2+1)
        
        x = xn_vec[j]
        t = tn_vec[j]
        
        for k in range(n1):
            a1_vec[k] = w_lay1_vec[k]*x + w_lay1_vec[k+n1]

        if lay1_activation == 'tanh':
            z1_vec = np.tanh(a1_vec)
        if lay1_activation == 'sin':
            z1_vec = np.sin(a1_vec)

        for k in range(n2):
            a2_vec[k] = np.dot(w_lay2_mtx[k,0:-1],z1_vec) + w_lay2_mtx[k,-1]

        if lay2_activation == 'tanh':
            z2_vec = np.tanh(a2_vec)
        if lay2_activation == 'sin':
            z2_vec = np.sin(a2_vec)

        y = np.dot(z2_vec,w_lay3_vec[0:-1]) + w_lay3_vec[-1] 

        E += 0.5*(y - t)**2

        if lay1_activation == 'tanh':
            dzda_lay1_vec = 1 - z1_vec**2
        if lay1_activation == 'sin':
            dzda_lay1_vec = np.cos(a1_vec)
            
        if lay2_activation == 'tanh':
            dzda_lay2_vec = 1 - z2_vec**2
        if lay2_activation == 'sin':
            dzda_lay2_vec = np.cos(a2_vec)

        for k in range(n1):
            for l in range(n2):
                dEdw_lay1_vec[k] += w_lay3_vec[l]*dzda_lay2_vec[l]*w_lay2_mtx[l,k]*dzda_lay1_vec[k]*(y-t)*x
                dEdw_lay1_vec[k+n1] += w_lay3_vec[l]*dzda_lay2_vec[l]*w_lay2_mtx[l,k]*dzda_lay1_vec[k]*(y-t) 

        for k in range(n2):
            for l in range(n1+1):
                if l < n1:
                    dEdw_lay2_mtx[k,l] = w_lay3_vec[k]*dzda_lay2_vec[k]*(y-t)*z1_vec[l]
                if l == n1:
                    dEdw_lay2_mtx[k,l] = w_lay3_vec[k]*dzda_lay2_vec[k]*(y-t)

        for k in range(n2):
            dEdw_lay3_vec[k] = (y-t)*z2_vec[k]

        dEdw_lay3_vec[-1] = y-t

        sum_dEdw_lay1_vec += dEdw_lay1_vec
        sum_dEdw_lay2_mtx += dEdw_lay2_mtx
        sum_dEdw_lay3_vec += dEdw_lay3_vec

    w_lay1_vec -= eta*sum_dEdw_lay1_vec/n_data
    w_lay2_mtx -= eta*sum_dEdw_lay2_mtx/n_data
    w_lay3_vec -= eta*sum_dEdw_lay3_vec/n_data

    RMSE = np.sqrt(2*E/n_data)

    RMSE_vec = np.append(RMSE_vec, RMSE)
    iter_vec = np.append(iter_vec, i)
    print('iter:', i)

plt.plot(iter_vec, RMSE_vec) 
plt.show()

print('final training RMSE:', RMSE_vec[-1])

xt_vec = input_test_data_vec
tt_vec = target_test_data_vec
yt_vec = np.ones(len(xt_vec))
a1_vec = np.zeros(n1)
a2_vec = np.zeros(n2)

E = 0

for j in range(n_test_data):
        x = xt_vec[j]
        t = tt_vec[j]
        
        for k in range(n1):
            a1_vec[k] = w_lay1_vec[k]*x + w_lay1_vec[k+n1]

        if lay1_activation == 'tanh':
            z1_vec = np.tanh(a1_vec)
        if lay1_activation == 'sin':
            z1_vec = np.sin(a1_vec)

        for k in range(n2):
            a2_vec[k] = np.dot(w_lay2_mtx[k,0:-1],z1_vec) + w_lay2_mtx[k,-1]

        if lay2_activation == 'tanh':
            z2_vec = np.tanh(a2_vec)
        if lay2_activation == 'sin':
            z2_vec = np.sin(a2_vec)
            
        y = np.dot(z2_vec,w_lay3_vec[0:-1]) + w_lay3_vec[-1] 
        yt_vec[j] = y

        E += 0.5*(y-t)**2

RMSE_test = np.sqrt(2*E/n_test_data)

print('Test: Input - Target - Prediction')
for i in range(n_test_data):
    print(xt_vec[i], tt_vec[i], yt_vec[i])

print('test RMSE:', RMSE_test)
plt.plot(xt_vec, tt_vec, 'bo', label = 'Data')
plt.plot(xt_vec, yt_vec, label = 'Predicted')
plt.legend()
plt.show()

xnew_vec = input_new_data_vec
ynew_vec = np.zeros(n_new_data)
a1_vec = np.zeros(n1)
a2_vec = np.zeros(n2)

for j in range(n_new_data):
        x = xnew_vec[j]
        
        for k in range(n1):
            a1_vec[k] = w_lay1_vec[k]*x + w_lay1_vec[k+n1]

        if lay1_activation == 'tanh':
            z1_vec = np.tanh(a1_vec)
        if lay1_activation == 'sin':
            z1_vec = np.sin(a1_vec)

        for k in range(n2):
            a2_vec[k] = np.dot(w_lay2_mtx[k,0:-1],z1_vec) + w_lay2_mtx[k,-1]

        if lay2_activation == 'tanh':
            z2_vec = np.tanh(a2_vec)
        if lay2_activation == 'sin':
            z2_vec = np.sin(a2_vec)
            
        y = np.dot(z2_vec,w_lay3_vec[0:-1]) + w_lay3_vec[-1] 
        ynew_vec[j] = y        

print('New: Input - Prediction')
for i in range(n_new_data):
    print(xnew_vec[i], ynew_vec[i])

plt.plot(xnew_vec, ynew_vec, label = 'Predicted')
plt.legend()
plt.show()    


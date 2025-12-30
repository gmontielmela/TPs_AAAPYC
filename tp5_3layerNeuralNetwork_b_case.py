import numpy as np
from matplotlib import pyplot as plt

n_data = 80
n_iter = 10000

x_train_min = 0
x_train_max = 0.7

n_test_data = 20

x_test_min = 0
x_test_max = 0.7

n_new_data = 101
x_new_min = 500
x_new_max = 600

eta = 1 #learning rate
n1 = 1 #excluding biases
n2 = 2 #excluding biases

lay1_activation = 'sin'
lay2_activation = 'sin'

xn_vec = np.linspace(x_train_min, x_train_max, n_data)
tn_vec = np.sin(xn_vec + np.sin(xn_vec**2))

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
            dzda_lay1_vec = (1 - z1_vec**2)
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

plt.plot(iter_vec, RMSE_vec) 
plt.show()

print('final training RMSE:', RMSE_vec[-1])

xt_vec = np.linspace(x_test_min, x_test_max, n_test_data)
tt_vec = np.sin(xt_vec + np.sin(xt_vec**2))
yt_vec = np.ones(len(xt_vec))
a1_vec = np.zeros(n1)
a2_vec = np.zeros(n2)
E=0

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
    print(xt_vec[i],tt_vec[i],yt_vec[i])

print('test RMSE:', RMSE_test)
plt.plot(xt_vec, tt_vec, 'bo', label = 'Data')
plt.plot(xt_vec, yt_vec, label = 'Predicted')
plt.legend()
plt.show()

xnew_vec = np.linspace(x_new_min, x_new_max, n_new_data)
tnew_vec = np.sin(xnew_vec + np.sin(xnew_vec**2))
ynew_vec = np.ones(len(xnew_vec))
a1_vec = np.zeros(n1)
a2_vec = np.zeros(n2)
E=0

for j in range(n_new_data):
        x = xnew_vec[j]
        t = tnew_vec[j]
        
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

        E += 0.5*(y-t)**2

RMSE_new = np.sqrt(2*E/n_new_data)

print('New: Input - Target - Prediction')
for i in range(n_new_data):
    print(xnew_vec[i],tnew_vec[i],ynew_vec[i])

print('new RMSE:', RMSE_new)
plt.plot(xnew_vec, tnew_vec, 'bo', label = 'Data')
plt.plot(xnew_vec, ynew_vec, label = 'Predicted')
plt.legend()
plt.show()
    



## MODULES AND SETTINGS
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import confusion_matrix
from scipy.io import loadmat
import numpy as np

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": "16",
})

## FUNCTIONS
def get_metrics(conf_2darr):
    n_classes = len(conf_2darr)
    
    for i in range(n_classes):
        
        Precision = conf_2darr[i,i]/np.sum(conf_2darr[:,i])
        TPR = conf_2darr[i,i]/np.sum(conf_2darr[i])
        FPR = (np.sum(conf_2darr[:,i])-conf_2darr[i,i])/\
                     (np.sum(conf_2darr)-np.sum(conf_2darr[i]))
        FDR = (np.sum(conf_2darr[:,i])-conf_2darr[i,i])/\
                      np.sum(conf_2darr[:,i])

        print('Metrics for class', i+1)
        print('-Precision:', Precision)
        print('-TPR:', TPR)
        print('-FPR:', FPR)
        print('-FDR:', FDR)

## DATA
data = loadmat("Datos_03.mat")
X_data = data["phi_t"]
y_data = data["t_Conjunto_1"]

n_inputs = len(X_data) #labelled + unlabelled inputs
n_samples = len(y_data) #labelled inputs
n_train_data = 500 #labelled data used for training
n_features = len(X_data[0])-1 #excluding bias
n_classes = len(y_data[0])

#indexes are shuffled until there is at least one specimen for each class

cond = False

while not cond:
    cond_vec = np.zeros(n_classes)
    perm = np.random.permutation(n_samples)
    perm = perm[:n_train_data]
    y_train_data = y_data[perm]

    for i in range(n_classes):
        if 1 in y_train_data[:,i]:
            cond_vec[i] = 1
            
    if 0 in cond_vec:
        cond = False
    else:
        cond = True

X_train_data = X_data[perm]

X = np.zeros((n_train_data, n_features))
y = np.zeros(n_train_data)

for i in range(n_train_data):
    X[i,:] = X_train_data[i,1:]
    y[i] = np.argmax(y_train_data[i])

X_full = np.zeros((n_inputs, n_features))

for i in range(n_inputs):
    X_full[i,:] = X_data[i,1:]
 
y_full = np.zeros(n_samples)

for i in range(n_samples):
    y_full[i] = np.argmax(y_data[i])

data_idx_1darr = np.linspace(0,n_samples-1,n_samples)
inputs_idx_1darr = np.linspace(0,n_inputs-1,n_inputs)

## SVC WITH LINEAR KERNEL
svc_linear = svm.SVC(kernel="linear")
svc_linear.fit(X,y)
svc_linear_y_pred = svc_linear.predict(X_full)

plt.figure(figsize=(10, 5))
plt.title('SVC with linear kernel')
plt.xlabel('Sample index')
plt.ylabel('Label')
plt.yticks([1,2,3])
plt.scatter(data_idx_1darr, y_full+1, c='r', s=50, label = 'Data')
plt.scatter(inputs_idx_1darr, svc_linear_y_pred+1, c='k', s=10, label = 'Prediction')
plt.legend(loc='lower right')
plt.show()

print('\n--- SVC with linear kernel ---')
conf_2darr = confusion_matrix(y_full, svc_linear_y_pred[:n_samples])
print('Confusion matrix:')
print(conf_2darr)
get_metrics(conf_2darr)

## SVC WITH RBF KERNEL
svc_rbf = svm.SVC(kernel = "rbf")
svc_rbf.fit(X,y)
svc_rbf_y_pred = svc_rbf.predict(X_full)

plt.figure(figsize=(10, 5))
plt.title('SVC with RBF kernel')
plt.xlabel('Sample index')
plt.ylabel('Label')
plt.yticks([1,2,3])
plt.scatter(data_idx_1darr, y_full+1, c='r', s=50, label = 'Data')
plt.scatter(inputs_idx_1darr, svc_rbf_y_pred+1, c='k', s=10, label = 'Prediction')
plt.legend(loc='lower right')
plt.show()

print('\n--- SVC with RBF kernel ---')
conf_2darr = confusion_matrix(y_full, svc_rbf_y_pred[:n_samples])
print('Confusion matrix:')
print(conf_2darr)
get_metrics(conf_2darr)

## SVC WITH POLYNOMIAL (DEGREE 3) KERNEL
svc_poly = svm.SVC(kernel = "poly")
svc_poly.fit(X,y)
svc_poly_y_pred = svc_poly.predict(X_full)

plt.figure(figsize=(10, 5))
plt.title('SVC with polynomial (degree 3) kernel')
plt.xlabel('Sample index')
plt.ylabel('Label')
plt.yticks([1,2,3])
plt.scatter(data_idx_1darr, y_full+1, c='r', s=50, label = 'Data')
plt.scatter(inputs_idx_1darr, svc_poly_y_pred+1, c='k', s=10, label = 'Prediction')
plt.legend(loc='lower right')
plt.show()

print('\n--- SVC with polynomial (degree 3) kernel ---')
conf_2darr = confusion_matrix(y_full, svc_poly_y_pred[:n_samples])
print(conf_2darr)
get_metrics(conf_2darr)

## LINEARSVC (LINEAR KERNEL)
linear_svc = svm.LinearSVC()
linear_svc.fit(X,y)
linear_svc_y_pred = linear_svc.predict(X_full)

plt.figure(figsize=(10, 5))
plt.title('LinearSVC (linear kernel)')
plt.xlabel('Sample index')
plt.ylabel('Label')
plt.yticks([1,2,3])
plt.scatter(data_idx_1darr, y_full+1, c='r', s=50, label = 'Data')
plt.scatter(inputs_idx_1darr, linear_svc_y_pred+1, c='k', s=10, label = 'Prediction')
plt.legend()
plt.show()

print('\n--- LinearSVC (linear kernel) ---')
conf_2darr = confusion_matrix(y_full, linear_svc_y_pred[:n_samples])
print('Confusion matrix:')
print(conf_2darr)
get_metrics(conf_2darr)

## MODULES AND SETTINGS

import numpy as np
import time
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": "16",
})

## DATA

random_train_idx = [7, 81, 73, 68, 28, 39, 98, 37, 77, 56, 3, 30, 76, 89, 51,\
                    72, 10, 45, 80, 42,43, 66, 85, 46, 47, 60, 14, 22, 69, 35,\
                    5, 32, 49, 99, 57, 24, 84, 83, 23, 44, 53, 0, 59, 63, 91,\
                    58, 93, 78, 96, 87, 12, 67, 50, 79, 64, 25, 19, 65, 29, 4,\
                    27, 55, 6, 92, 71, 95, 21, 26, 74, 52, 54, 13, 2, 40, 8, 38,\
                    15, 34, 94, 9]

random_test_idx = [1, 11, 16, 17, 18, 20, 31, 33, 36, 41, 48, 61, 62, 70, 75, 82,\
                   86, 88, 90, 97]

inputs = np.loadtxt("tp3_training_inputs.txt")
targets = np.loadtxt("tp3_training_targets.txt")
new_inputs = np.loadtxt("tp3_new_inputs.txt")

X = inputs[random_train_idx]       
y = targets[random_train_idx]      
X_plot = inputs[random_test_idx]
y_plot = targets[random_test_idx]
X_new = new_inputs

X = X.reshape(-1, 1)        
y = y.reshape(-1,)          
X_plot = X_plot.reshape(-1, 1)
y_plot = y_plot.reshape(-1,)
X_new = X_new.reshape(-1, 1)

n_train = len(X)
n_test = len(X_plot)

## MODEL SETTINGS

svr = GridSearchCV(
    SVR(kernel="rbf"),
    param_grid = {'kernel': ['rbf'],\
                  'C': [1, 10, 10, 1000],\
                  'gamma': [1/1, 1/10, 1/100, 1/1000],\
                  'epsilon': [0.01, 0.1, 1, 10]}
)

kr = GridSearchCV(
    KernelRidge(kernel="rbf"),
    param_grid = {'kernel': ['rbf'],\
                  'alpha': [1/1, 1/10, 1/100, 1/1000],\
                  'gamma': [1/1, 1/10, 1/100, 1/1000]}
)

## TRAIN AND TEST MODELS, COMPARISON OF TRAINING AND TESTING TIMES

t0 = time.time()
svr.fit(X, y)
svr_fit = time.time() - t0
print(f"Best SVR with params: {svr.best_params_} and R2 score:\
      {svr.best_score_:.3f}")
print("SVR complexity and bandwidth selected and model fitted in %.3f s"\
      % svr_fit)

t0 = time.time()
kr.fit(X, y)
kr_fit = time.time() - t0
print(f"Best KRR with params: {kr.best_params_} and R2 score:\
      {kr.best_score_:.3f}")
print("KRR complexity and bandwidth selected and model fitted in %.3f s"\
      % kr_fit)

sv_ratio = svr.best_estimator_.support_.shape[0] / n_train
print("Support vector ratio: %.3f" % sv_ratio)

t0 = time.time()
y_svr = svr.predict(X_plot)
svr_predict = time.time() - t0
print("SVR prediction for %d inputs in %.3f s" % (X_plot.shape[0], svr_predict))

print("SVR test set prediction score:", round(svr.score(X_plot, y_plot),3))

t0 = time.time()
y_kr = kr.predict(X_plot)
kr_predict = time.time() - t0
print("KRR prediction for %d inputs in %.3f s" % (X_plot.shape[0], kr_predict))

print("KRR test set prediction score:", round(kr.score(X_plot, y_plot),3))

plt.figure(figsize=(6, 6))

sv_ind = svr.best_estimator_.support_
plt.scatter(
    X[sv_ind],
    y[sv_ind],
    c="r",
    s=50,
    label="SVR support vectors",
    zorder=2,
    edgecolors=(0, 0, 0),
)

plt.scatter(X, y, c="k", label="data", zorder=1,\
            edgecolors=(0, 0, 0))

plt.plot(
    X_plot,
    y_svr,
    c="r",
    label="SVR (fit: %.3fs, predict: %.3fs)" % (svr_fit, svr_predict),
)
plt.plot(
    X_plot, y_kr, c="g", label="KRR (fit: %.3fs, predict: %.3fs)" %\
    (kr_fit, kr_predict)
)

plt.xlabel("data")
plt.ylabel("target")
plt.title("SVR versus Kernel Ridge")
plt.legend(
    loc='upper center',         
    bbox_to_anchor=(0.5, -0.2),  
)
plt.tight_layout()
plt.show()


## MAKE PREDICTIONS FOR UNKNOWN ENTRIES

plt.figure(figsize=(6, 6))

y_svr = svr.predict(X_new)

plt.plot(
    X_new,
    y_svr,
    c="r",
    label="SVR",
    )

y_kr = kr.predict(X_new)

plt.plot(
    X_new,
    y_kr,
    c="g",
    label="KRR",
    )

plt.xlabel("data")
plt.ylabel("target")
plt.title("SVR versus Kernel Ridge")
plt.legend(
    loc='upper center',         
    bbox_to_anchor=(0.5, -0.2),  
)
plt.tight_layout()
plt.show()

## TEST DIFFERENT DATA SIZES AND COMPARE FITTING AND TESTING TIMES

sizes = np.linspace(n_test, n_train, 7).astype(int)

kr_train_time = []
kr_test_time = []
svr_train_time = []
svr_test_time = []

for train_test_size in sizes:
    t0 = time.time()
    kr.fit(X[:train_test_size], y[:train_test_size])
    kr_train_time.append(time.time()-t0)

    t0 = time.time()
    kr.predict(X_plot)
    kr_test_time.append(time.time()-t0)

    t0 = time.time()
    svr.fit(X[:train_test_size], y[:train_test_size])
    svr_train_time.append(time.time()-t0)

    t0 = time.time()
    svr.predict(X_plot)
    svr_test_time.append(time.time()-t0)

plt.figure(figsize=(6, 6))

plt.plot(
    sizes,
    kr_train_time,
    "o-",
    color="g",
    label="KRR (train) ",
)

plt.plot(
    sizes,
    kr_test_time,
    "o--",
    color="g",
    label="KRR (test)",
)

plt.plot(
    sizes,
    svr_train_time,
    "o-",
    color="r",
    label="SVR (train)",
)

plt.plot(
    sizes,
    svr_test_time,
    "o--",
    color="r",
    label="SVR (test)",
)

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Train size")
plt.ylabel("Time (seconds)")
plt.title("Execution Time")
plt.legend(
    loc='upper center',         
    bbox_to_anchor=(0.5, -0.2), 
    ncol=2,                     
)

plt.tight_layout()
plt.show()

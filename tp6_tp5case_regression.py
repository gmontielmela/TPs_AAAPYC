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

n_train = 80
n_test = 20
n_new = 100

x_train_min = 0
x_train_max = 0.7

x_test_min = 0
x_test_max = 0.7

x_new_min = 500
x_new_max = 600

X = np.linspace(x_train_min, x_train_max, n_train)    
y = np.sin(X + np.sin(X**2))
X_plot = np.linspace(x_test_min, x_test_max, n_test)
y_plot = np.sin(X_plot + np.sin(X_plot**2))
X_new = np.linspace(x_new_min, x_new_max, n_new)
y_new = np.sin(X_new + np.sin(X_new**2))

X = X.reshape(-1, 1)        
y = y.reshape(-1,)          
X_plot = X_plot.reshape(-1, 1)
y_plot = y_plot.reshape(-1,)
X_new = X_new.reshape(-1, 1)
y_new = y_new.reshape(-1,)

## MODEL SETTINGS

svr = GridSearchCV(
    SVR(kernel="rbf"),
    param_grid = {'kernel': ['rbf'],\
                  'C': [1, 10, 100, 1000],\
                  'gamma': [1/1, 1/10, 1/100, 1/1000],\
                  'epsilon': [0.01, 0.1, 1, 10]}
)

kr = GridSearchCV(
    KernelRidge(kernel="rbf"),
    param_grid = {'kernel': ['rbf'],\
                  'alpha': [1/1, 1/10, 1/100, 1/1000],\
                  'gamma': [1/1, 1/10, 1/100, 1/1000]}
)

## COMPARE FITTING AND TESTING TIMES

t0 = time.time()
svr.fit(X, y)
svr_fit = time.time() - t0
print(f"Best SVR with params: {svr.best_params_} and R2 score:\
      {svr.best_score_:.3f}")
print("SVR complexity and bandwidth selected and model fitted in %.4f s"\
      % svr_fit)

t0 = time.time()
kr.fit(X, y)
kr_fit = time.time() - t0
print(f"Best KRR with params: {kr.best_params_} and R2 score:\
      {kr.best_score_:.3f}")
print("KRR complexity and bandwidth selected and model fitted in %.4f s"\
      % kr_fit)

sv_ratio = svr.best_estimator_.support_.shape[0] / n_train
print("Support vector ratio: %.4f" % sv_ratio)

t0 = time.time()
y_svr = svr.predict(X_plot)
svr_predict = time.time() - t0
print("SVR prediction for %d inputs in %.4f s" % (X_plot.shape[0], svr_predict))

print("SVR test set prediction score:", round(svr.score(X_plot, y_plot),3))

t0 = time.time()
y_kr = kr.predict(X_plot)
kr_predict = time.time() - t0
print("KRR prediction for %d inputs in %.4f s" % (X_plot.shape[0], kr_predict))

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
    label="SVR (fit: %.4fs, predict: %.4fs)" % (svr_fit, svr_predict),
)
plt.plot(
    X_plot, y_kr, c="g", label="KRR (fit: %.4fs, predict: %.4fs)" %\
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

## MAKE PREDICTIONS FOR THE UNKNOWN ENTRIES

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

plt.scatter(X_new, y_new, c="k", label="data", zorder=1,\
            edgecolors=(0, 0, 0))

plt.xlabel("data")
plt.ylabel("target")
plt.title("SVR versus Kernel Ridge")
plt.legend(
    loc='upper center',         
    bbox_to_anchor=(0.5, -0.2),  
)
plt.tight_layout()
plt.show()

## TEST DIFFERENT TRAIN SET SIZES AND COMPARE FITTING AND TESTING TIMES

n_train = 400
n_test = 100

x_train_min = 0
x_train_max = 0.7

x_test_min = 0
x_test_max = 0.7

X = np.linspace(x_train_min, x_train_max, n_train)    
y = np.sin(X + np.sin(X**2))
X_plot = np.linspace(x_test_min, x_test_max, n_test)
y_plot = np.sin(X_plot + np.sin(X_plot**2))

X = X.reshape(-1, 1)        
y = y.reshape(-1,)          
X_plot = X_plot.reshape(-1, 1)
y_plot = y_plot.reshape(-1,)

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
    svr_test_time,
    "o-",
    color="r",
    label="SVR (train)",
)

plt.plot(
    sizes,
    svr_train_time,
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

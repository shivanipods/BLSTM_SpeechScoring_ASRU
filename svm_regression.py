#!/usr/bin/env python
import sys
import os
import pickle as pkl
import numpy as np
from scikits.learn.svm import SVR
from multiprocessing import Process, Queue
from scipy.stats.stats import pearsonr
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

with open('data_clean/train.pkl') as f:
    train_key  = pkl.load(f)
    train_data = pkl.load(f)
    train_r1 = pkl.load(f)

with open('data_clean/val.pkl') as f:
    val_key  = pkl.load(f)
    val_data = pkl.load(f)
    val_r1 = pkl.load(f)

with open('data_clean/test.pkl') as f:
    test_key  = pkl.load(f)
    test_data = pkl.load(f)
    test_r1 = pkl.load(f)
    test_r2 = pkl.load(f)
print len(val_key)
gamma = [10**i for i in range(-3, 4, 1)]
C = [10**i for i in range(-3, 4, 1)]
val_mse = []


param_queue = Queue()
mse_queue = Queue()

def validate(param_queue, mse_queue, train_data, train_r1, val_data, val_r1):
    while True:
        param = param_queue.get()
        if param is None:
            break
        index = param[0]
        g = param[1]
        c = param[2]
        svr = SVR(kernel='rbf', C = c, gamma = g)
        svr.fit(train_data, train_r1)
        val_predict = svr.predict(val_data)
        mse = np.mean((val_predict - val_r1)**2)
        mse_queue.put((index, mse))

n_process = 8
process = [None] * n_process
for idx in xrange(n_process):
    process[idx] = Process(target=validate,
                           args=(param_queue, mse_queue, train_data,
                                 train_r1, val_data, val_r1))
    process[idx].start()

index = 0
for g in gamma:
    for c in C:
        param_queue.put((index, g, c))
        index += 1

for idx in xrange(n_process):
    param_queue.put(None)

val_mse = [None] * len(gamma) * len(C)
for i in xrange(len(gamma) * len(C)):
    mse_entry = mse_queue.get()
    val_mse[mse_entry[0]] = mse_entry[1]
    if i % 1 == 0:
        print 'processed %d' %(i)

best_idx = val_mse.index(min(val_mse))
best_gamma = gamma[best_idx / len(C)]
best_c = C[best_idx % len(C)]
print best_idx, best_gamma, best_c

svr = SVR(kernel='rbf', C = best_c, gamma = best_gamma)
svr.fit(train_data, train_r1)
test_predict = svr.predict(test_data)
mse_r1 = np.mean((test_r1 - test_predict)**2)
mse_r2 = np.mean((test_r2 - test_predict)**2)
print pearsonr(test_r1,test_predict)
print 'mse_r1: %.3f mse_r2: %.3f' %(mse_r1, mse_r2)

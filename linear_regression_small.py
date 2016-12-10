#!/usr/bin/env python

import pickle as pkl
import numpy as np
from scikits.learn import linear_model
from scipy.stats.stats import pearsonr
with open('data/train_small.pkl') as f:
    train_key  = pkl.load(f)
    train_data = pkl.load(f)
    train_r1 = pkl.load(f)

with open('data/val_small.pkl') as f:
    val_key  = pkl.load(f)
    val_data = pkl.load(f)
    val_r1 = pkl.load(f)

# with open('data/test.pkl') as f:
#     test_key  = pkl.load(f)
#     test_data = pkl.load(f)
#     test_r1 = pkl.load(f)
#     test_r2 = pkl.load(f)

alpha = [2**i for i in range(-20, 10, 1)]
val_rmse = []
val_corr = []
for a in alpha:
    clf = linear_model.Ridge(alpha = a)
    clf.fit(train_data, train_r1)
    val_predict = clf.predict(val_data)
    val_rmse.append(np.sqrt(np.mean((val_predict - val_r1)**2)))
    val_corr.append(pearsonr(val_predict, val_r1))
best_alpha = alpha[val_rmse.index(min(val_rmse))]
print val_rmse
print val_corr
# clf = linear_model.Ridge(alpha = best_alpha)
# clf.fit(train_data, train_r1)
# test_predict = clf.predict(test_data)
# rmse_r1 = np.sqrt(np.mean((test_r1 - test_predict)**2))
# rmse_r2 = np.sqrt(np.mean((test_r2 - test_predict)**2))
# #coef_r1 = np.corrcoef(test_predict,test_r1)
# print pearsonr(test_r1,test_predict)
# print 'rmse_r1: %.2f rmse_r2: %.2f' %(rmse_r1, rmse_r2)
# #print coef_r1
# #print 'coef_r1: %.2f ' %coef_r1

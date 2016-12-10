#!/usr/bin/env python

import pickle as pkl
import numpy as np
from scikits.learn import linear_model
from scipy.stats.stats import pearsonr
with open('data_no_asr_orinial/train.pkl') as f:
    train_key  = pkl.load(f)
    train_data = pkl.load(f)
    train_r1 = pkl.load(f)

with open('data_no_asr_orinial/val.pkl') as f:
    val_key  = pkl.load(f)
    val_data = pkl.load(f)
    val_r1 = pkl.load(f)

with open('data_no_asr_orinial/test.pkl') as f:
    test_key  = pkl.load(f)
    test_data = pkl.load(f)
    test_r1 = pkl.load(f)
    test_r2 = pkl.load(f)

#some stats on r1 and r2
print len(test_r1)
print pearsonr(test_r1,test_r2)
print len(val_r1)
mse_r1_r2 = np.mean((test_r1 - test_r2)**2)
print mse_r1_r2
alpha = [2**i for i in range(-20, 10, 1)]
val_rmse = []

for a in alpha:
    clf = linear_model.Ridge(alpha = a)
    clf.fit(train_data, train_r1)
    val_predict = clf.predict(val_data)
    val_rmse.append(np.mean((val_predict - val_r1)**2))

best_alpha = alpha[val_rmse.index(min(val_rmse))]

clf = linear_model.Ridge(alpha = best_alpha)
clf.fit(train_data, train_r1)
test_predict = clf.predict(test_data)
test_av = (test_r1 + test_r2)/2
mse_r1 = np.mean((test_r1 - test_predict)**2)
mse_r2 = np.mean((test_r2 - test_predict)**2)
mse_av = np.mean((test_av - test_predict)**2)
#coef_r1 = np.corrcoef(test_predict,test_r1)
print pearsonr(test_r1,test_predict)
print 'mse_r1: %.3f mse_r2: %.3f mse_av: %.3f' %(mse_r1, mse_r2,mse_av)
print 'correaltion of mean r1 r2'
print pearsonr(test_av,test_predict)

#print coef_r1
#print 'coef_r1: %.2f ' %coef_r1

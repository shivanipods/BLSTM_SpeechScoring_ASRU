#!/usr/bin/python

import datetime
import os
import sys

from lstm_theano import *
from data_provision import *
from batch_processing import *

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

model = sys.argv[1]
split = sys.argv[2]

data_path = 'data'
data_provision = DataProvision(data_path)

# load model
options, params, shared_params = load_model(model)

feat, seq_feat, seq_feat_reverse, input_mask, y,\
    dropout, cost, y_pred = build_model(shared_params, options)

# validation function no gradient updates
f_test = theano.function(inputs = [feat, seq_feat, seq_feat_reverse,
                                  input_mask],
                        outputs = [y_pred])


if not split == 'test':
    pred_y_list = []
    y_list = []
    for batch_feat, batch_seq_feat, batch_r1 in \
        data_provision.iterate_batch(split, batch_size):
        batch_seq_feat, batch_seq_feat_reverse, mask = process_batch(batch_seq_feat)
        dropout.set_value(numpy.float32(0.))
        pred_y = f_test(batch_feat, batch_seq_feat, batch_seq_feat_reverse, mask)
        pred_y_list.append(pred_y.tolist())
        y_list.append(batch_r1.tolist())
    mse = np.mean((np.array(pred_y_list) - np.array(y_list))**2)
    rmse = sqrt(mse)
    print '%s mse: %f rmse: %f' %(split, mse, rmse)
else:
    pred_y_list = []
    r1_list = []
    r2_list = []
    for batch_feat, batch_seq_feat, batch_r1, batch_r2 in \
        data_provision.iterate_batch(split, batch_size):
        batch_seq_feat, batch_seq_feat_reverse, mask = process_batch(batch_seq_feat)
        dropout.set_value(numpy.float32(0.))
        pred_y = f_test(batch_feat, batch_seq_feat, batch_seq_feat_reverse, mask)
        pred_y_list.append(pred_y.tolist())
        r1_list.append(batch_r1.tolist())
        r2_list.append(batch_r2.tolist())

    mse_r1 = np.mean((np.array(pred_y_list) - np.array(r1_list))**2)
    rmse_r1 = sqrt(mse_r1)
    mse_r2 = np.mean((np.array(pred_y_list) - np.array(r2_list))**2)
    rmse_r2 = sqrt(mse_r2)
    print '%s r1 mse: %f rmse: %f' %(split, mse_r1, rmse_r1)
    print '%s r2 mse: %f rmse: %f' %(split, mse_r2, rmse_r2)
#!/usr/bin/env python

import datetime
import os
import sys

from lstm_theano import *
from data_provision import *
from batch_processing import *
import pickle as pkl
def lstm_regression(options):
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

    data_path = 'data'
    data_provision = DataProvision(data_path)

    ##################
    # initialization #
    ##################
    # dimensions
    options['n_seq_feat'] = data_provision.get_n_seq_feat()
    options['n_feat'] = data_provision.get_n_feat()

    batch_size = options['batch_size']
    max_epochs = options['max_epochs']


    ###############
    # build model #
    ###############
    params = init_params(options)
    # initialize with bias vectors
    # params['b_o'] = bias_init_vector.astype('float32')
    shared_params = init_shared_params(params)

    feat, seq_feat, seq_feat_reverse, input_mask, y,\
        dropout, cost, y_pred = build_model(shared_params, options)

    ####################
    # add weight decay #
    ####################
    weight_decay = theano.shared(numpy.float32(options['weight_decay']),\
                                 name = 'weight_decay')
    reg_cost = 0

    for k in shared_params.iterkeys():
        # for all the weights
        if 'w' in k:
            # print k
            reg_cost += (shared_params[k]**2).sum()
        reg_cost *= weight_decay

    reg_cost = cost + reg_cost;

    ###############
    # # gradients #
    ###############
    grads = T.grad(reg_cost, wrt = shared_params.values())
    grad_buf = [theano.shared(p.get_value() * 0, name='%s_grad_buf' % k )
                for k, p in shared_params.iteritems()]
    # accumulate the gradients within one batch
    update_grad = [(g_b, g) for g_b, g in zip(grad_buf, grads)]
    # need to declare a share variable ??
    grad_clip = options['grad_clip']
    update_clip = [(g_b, T.clip(g_b, -grad_clip, grad_clip)) for g_b in grad_buf]

    # corresponding update function
    f_grad_clip = theano.function(inputs = [],
                                  updates = update_clip)
    f_train = theano.function(inputs = [feat, seq_feat, seq_feat_reverse,
                                        input_mask, y],
                              outputs = [cost, y_pred],
                              updates = update_grad)
    # validation function no gradient updates
    f_val = theano.function(inputs = [feat, seq_feat, seq_feat_reverse,
                                      input_mask, y],
                            outputs = [cost, y_pred])

    # prediction function no y as input
    f_predict = theano.function(inputs = [feat, seq_feat, seq_feat_reverse,
                                          input_mask],
                                outputs = y_pred)

    grad_cache, f_grad_cache_update,\
        f_param_update = rmsprop(shared_params, grad_buf, options)

    # calculate how many iterations we need
    num_iters_one_epoch =  data_provision.get_size('train') / batch_size
    max_iters = max_epochs * num_iters_one_epoch
    eval_period_in_epochs = options['eval_period']
    eval_interval_in_iters = options['eval_interval']
    save_interval_in_iters = options['save_interval']

    # check_point_basename = sys.argv[1]
    check_point_basename = 'toefl'
    check_point_folder = 'model'

    for itr in xrange(max_iters):
    # for itr in xrange(1):
        batch_feat, batch_seq_feat, batch_r1 = data_provision.next_batch('train', batch_size)
        batch_seq_feat, batch_seq_feat_reverse, mask = process_batch(batch_seq_feat)
        dropout.set_value(numpy.float32(1.))
        cost, pred_y = f_train(batch_feat, batch_seq_feat, batch_seq_feat_reverse, mask,
                               batch_r1)
        f_grad_clip()
        f_grad_cache_update()
        f_param_update()
        print '%s] iteration %d/%d cost %f' \
            % (datetime.datetime.now().isoformat(), \
               itr, max_iters, cost, )

        is_last_iter = (itr + 1) == max_iters
        if itr == 0 or  (((itr + 1) % eval_interval_in_iters) == 0  and \
                        itr < max_iters - 5) or is_last_iter:
            # perform evaluation on val set
            # close dropout first
            dropout.set_value(numpy.float32(0.))
            print '%s]' %(datetime.datetime.now().isoformat())
            val_cost_list = []
            for batch_feat, batch_seq_feat, batch_r1 in \
                data_provision.iterate_batch('val', batch_size):
                batch_seq_feat, batch_seq_feat_reverse, mask = process_batch(batch_seq_feat)
                dropout.set_value(numpy.float32(0.))
                cost, pred_y = f_val(batch_feat, batch_seq_feat, batch_seq_feat_reverse, mask,
                                     batch_r1)
                val_cost_list.append(cost)
            print 'validatation mse: %f' %(sum(val_cost_list)/len(val_cost_list))

        if ((itr + 1) % save_interval_in_iters) == 0 or is_last_iter:
            # copy parameters from gpu to gpu
            print '%s]' %(datetime.datetime.now().isoformat())
            file_name = check_point_basename + '_' \
                        + str(itr + 1) + '_' +  str(options['lr']*numpy.float32(1e5)) + '_' +'%.2f' %(sum(val_cost_list)/len(val_cost_list))+'.model'
            print 'saving model to %s' % (file_name)
            save_model(os.path.join(check_point_folder, file_name),
                       options, params, shared_params);
    return sum(val_cost_list)/len(val_cost_list)

if __name__ == '__main__':
    best = numpy.float32(1000)
    leanring_rate = [numpy.float32(5e-5),numpy.float32(1e-4),numpy.float32(1e-3)]
    validation_mse =[]
    for lr in leanring_rate:
        options = dict()
        options['n_dim'] = 512
        options['n_output'] = 1

        options['use_tanh'] = False

        options['init_type'] = 'uniform'
        options['range'] = 0.01

        # learning parameters
        options['batch_size'] = 32
        options['lr'] = lr
        options['momentum'] = numpy.float32(0.9)
        options['max_epochs'] = 50
        options['weight_decay'] = 0.0
        options['decay_rate'] = numpy.float32(0.999)
        options['drop_ratio'] = numpy.float32(0.5)
        options['smooth'] = numpy.float32(1e-8)
        options['grad_clip'] = numpy.float32(5)

        # interval to evaluate model
        # interval to save model
        options['eval_period'] = 1
        # eval interval in terms iterations
        options['eval_interval'] = 100
        options['save_interval'] = 500
        mse = lstm_regression(options)
        if mse < best:
            param = options
            best = mse
            validation_mse.append(mse)
    print best
    with open('model/param.pkl', 'wb') as f:
            pkl.dump(param, f)
            pkl.dump(best)
    print "one param done"
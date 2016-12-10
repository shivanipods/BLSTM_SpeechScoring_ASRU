#!/usr/bin/env python

import theano
import theano.tensor as T
import numpy
from collections import OrderedDict
from theano import config
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import cPickle as pickle

floatX = config.floatX

def shared_to_cpu(shared_params, params):
    for k, v in shared_params.iteritems():
        params[k] = v.get_value()

def cpu_to_shared(params, shared_params):
    for k, v in params.iteritems():
        shared_params[k].set_value(v)

def save_model(filename, options, params, shared_params):
    shared_to_cpu(shared_params, params);
    model = OrderedDict()
    model['options'] = options
    model['params'] = params
    pickle.dump(model, open(filename, 'w'))

def load_model(filename):
    model = pickle.load(open(filename, 'rb'))
    options = model['options']
    params = model['params']
    shared_params = init_shared_params(params)
    return options, params, shared_params

def init_weight(n, d, options):
    if options['init_type'] == 'gaussian':
        return (numpy.random.randn(n, d).astype(floatX)) * options['std']
    elif options['init_type'] == 'uniform':
        # [-range, range]
        return ((numpy.random.rand(n, d) * 2 - 1) * \
                options['range']).astype(floatX)

# initialize the parmaters
def init_params(options):
    '''
    Initialize all the parameters
    '''
    params = OrderedDict()
    # dimension of sequence feature
    n_seq_feat = options['n_seq_feat']
    # dimension of non sequence feature
    n_feat = options['n_feat']
    n_dim = options['n_dim']
    n_output = options['n_output']

    # lstm encoder weights and bias
    params['encoder_w_x'] = init_weight(n_seq_feat, 4 * n_dim, options)
    params['encoder_w_h'] = init_weight(n_dim, 4 * n_dim, options)
    params['encoder_b_h'] = numpy.zeros(4 * n_dim, dtype=floatX)

    params['encoder_reverse_w_x'] = init_weight(n_seq_feat, 4 * n_dim, options)
    params['encoder_reverse_w_h'] = init_weight(n_dim, 4 * n_dim, options)
    params['encoder_reverse_b_h'] = numpy.zeros(4 * n_dim, dtype=floatX)


    # outputs weights and bais
    # this is the weight to multiple the combined lstm feat and non_seq feat
    params['w_o'] = init_weight(2*n_dim + n_feat, n_output, options)
    params['b_o'] = numpy.zeros(n_output, dtype=floatX)

    return params

# return a shared version of all parameters
def init_shared_params(params):
    shared_params = OrderedDict()
    for k, p in params.iteritems():
        shared_params[k] = theano.shared(params[k], name = k)

    return shared_params


def lstm_layer(shared_params, x, mask, coder, h_0, c_0, options):
    ''' this is the lstm layer: either encoder or decoder
    :param shared_params: shared parameters
    :param x: input, T x batch_size x n_seq_feat
    :param mask: mask for x, T x batch_size
    '''
    # batch_size = optins['batch_size']
    n_dim = options['n_dim']
    # weight matrix for x, n_seq_feat x 4*n_dim (ifoc)
    lstm_w_x = shared_params[coder + '_w_x']
    # weight matrix for h, n_dim x 4*n_dim
    lstm_w_h = shared_params[coder + '_w_h']
    lstm_b_h = shared_params[coder + '_b_h']

    def recurrent(x_t, mask_t, h_tm1, c_tm1):
        ifoc = T.dot(x_t, lstm_w_x) + T.dot(h_tm1, lstm_w_h) + lstm_b_h
        # 0:3*n_dim: input forget and output gate
        i_gate = T.nnet.sigmoid(ifoc[:, 0 : n_dim])
        f_gate = T.nnet.sigmoid(ifoc[:, n_dim : 2*n_dim])
        o_gate = T.nnet.sigmoid(ifoc[:, 2*n_dim : 3*n_dim])
        # 3*n_dim : 4*n_dim c_temp
        c_temp = T.tanh(ifoc[:, 3*n_dim : 4*n_dim])
        # c_t = input_gate * c_temp + forget_gate * c_tm1
        c_t = i_gate * c_temp + f_gate * c_tm1

        if options['use_tanh']:
            h_t = o_gate * T.tanh(c_t)
        else:
            h_t = o_gate * c_t

        # if mask = 0, then keep the previous c and h
        h_t = mask_t[:, None] * h_t + \
              (numpy.float32(1.0) - mask_t[:, None]) * h_tm1
        c_t = mask_t[:, None] * c_t + \
              (numpy.float32(1.0) - mask_t[:, None]) * c_tm1

        return h_t, c_t

    [h, c], updates = theano.scan(fn = recurrent,
                                  sequences = [x, mask],
                                  outputs_info = [h_0[:x.shape[1]],
                                                  c_0[:x.shape[1]]],
                                  n_steps = x.shape[0])
    return h, c

def build_model(shared_params, options):
    trng = RandomStreams(1234)
    drop_ratio = options['drop_ratio']
    batch_size = options['batch_size']
    n_dim = options['n_dim']

    w_o = shared_params['w_o']
    b_o = shared_params['b_o']

    # sequence features
    # batch_size x n_feat
    feat = T.matrix('feat')
    # T x batch_size x n_seq_feat
    seq_feat = T.tensor3('seq_feat')
    seq_feat_reverse = T.tensor3('seq_feat_reverse')
    # seq_feat and seq_feat_reverse shared the same mask
    input_mask = T.matrix('input_mask')
    y = T.vector('y')
    dropout = theano.shared(numpy.float32(0.))

    h_0 = theano.shared(numpy.zeros((batch_size, n_dim), dtype='float32'))
    c_0 = theano.shared(numpy.zeros((batch_size, n_dim), dtype='float32'))

    h_encode, c_encode = lstm_layer(shared_params, seq_feat, input_mask,
                                    'encoder', h_0, c_0, options)
    h_encode_reverse, c_encode_reverse = lstm_layer(shared_params, seq_feat_reverse, input_mask,
                                                    'encoder_reverse', h_0, c_0, options)
    # pick the last one as encoder
    h_encode = h_encode[-1]
    h_encode_reverse = h_encode_reverse[-1]
    h_encode_combined = T.concatenate([h_encode, h_encode_reverse], axis=1)

    # use dropout after decoding
    h_encode_combined_drop = T.switch(dropout,
                                      (h_encode_combined*
                                       trng.binomial(h_encode_combined.shape,
                                                     p = 1 - drop_ratio,
                                                     n = 1,
                                                     dtype = h_encode_combined.dtype) \
                                       / (numpy.float32(1.0) - drop_ratio)),
                                      h_encode_combined)

    combined_feat = T.concatenate([h_encode_combined_drop, feat], axis=1)


    y_pred = T.dot(combined_feat, w_o) + b_o
    # turn matrix into vector
    y_pred = y_pred.flatten()
    cost = T.mean((y_pred - y)**2)

    return feat, seq_feat, seq_feat_reverse, input_mask, y, \
        dropout, cost, y_pred

def sgd(shared_params, grads, options):
    '''
    grads is already the shared variable containing the gradients, we only
    need to do a accumulation and then do an updated
    '''
    lr = options['lr']
    momentum = options['momentum']
    # the cache here can not be reseach by outside function
    grad_cache = [theano.shared(p.get_value() * numpy.float32(0.),
                                name='%s_grad_cache' % k )
                  for k, p in shared_params.iteritems()]
    # update the caches
    grad_cache_update = [(g_c, g_c * momentum + g)
                         for g_c, g in zip (grad_cache, grads)]
    param_update = [(p, p - lr * g_c )
                    for p, g_c in zip(shared_params.values(),
                                      grad_cache)]

    # two functions: do the grad cache updates and param_update
    f_grad_cache_update = theano.function([], [],
                                          updates = grad_cache_update,
                                          name = 'f_grad_cache_update')
    f_param_update = theano.function([], [],
                                     updates = param_update,
                                     name = 'f_param_update')

    return grad_cache, f_grad_cache_update, f_param_update


def rmsprop(shared_params, grads, options):
    lr = options['lr']
    decay_rate = options['decay_rate']
    smooth = options['smooth']
    # the cache here can not be reseach by outside function
    grad_cache = [theano.shared(p.get_value() * numpy.float32(0.),
                                name='%s_grad_cache' % k)
                  for k, p in shared_params.iteritems()]
    # update the caches
    grad_cache_update = [(g_c, g_c * decay_rate +
                          (numpy.float32(1.) - decay_rate) * g**2)
                         for g_c, g in zip (grad_cache, grads)]
    param_update = [(p, p - lr * g / T.sqrt(g_c + smooth))
                    for p, g_c, g in zip(shared_params.values(), grad_cache, grads)]

    # two functions: do the grad cache updates and param_update
    f_grad_cache_update = theano.function([], [],
                                          updates = grad_cache_update,
                                          name = 'f_grad_cache_update')
    f_param_update = theano.function([], [],
                                     updates = param_update,
                                     name = 'f_param_update')

    return grad_cache, f_grad_cache_update, f_param_update
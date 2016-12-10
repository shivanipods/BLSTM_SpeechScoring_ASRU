#!/usr/bin/env python

import numpy as np

def process_batch(batch_seq_feat, down_sample_rate = 10):
    batch_size = len(batch_seq_feat)
    num_seq_feat = batch_seq_feat[0].shape[1]
    # downsample
    max_len = 0
    for i in range(len(batch_seq_feat)):
        batch_seq_feat[i] = batch_seq_feat[i][0:batch_seq_feat[i].shape[0]:down_sample_rate]
        max_len = max(max_len, batch_seq_feat[i].shape[0])
    mask = np.zeros((max_len, batch_size), dtype='float32')
    batch_seq_feat_fill = np.zeros((batch_size, max_len, num_seq_feat), dtype='float32')
    batch_seq_feat_reverse_fill = np.zeros((batch_size, max_len, num_seq_feat),
                                           dtype='float32')
    for i in range(len(batch_seq_feat)):
        mask[0:batch_seq_feat[i].shape[0], i] = 1
        batch_seq_feat_fill[i, 0:batch_seq_feat[i].shape[0], :] = batch_seq_feat[i]
        batch_seq_feat_reverse_fill[i, 0:batch_seq_feat[i].shape[0], :] = batch_seq_feat[i][::-1]
    batch_seq_feat_fill = np.swapaxes(batch_seq_feat_fill, 0, 1)
    batch_seq_feat_reverse_fill = np.swapaxes(batch_seq_feat_reverse_fill, 0, 1)
    return batch_seq_feat_fill, batch_seq_feat_reverse_fill, mask
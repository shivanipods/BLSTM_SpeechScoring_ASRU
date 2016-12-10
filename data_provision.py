#!/usr/bin/env python

import numpy as np
import pickle
from collections import OrderedDict
import os

class DataProvision:
    def __init__(self, folder):
        self._feat = OrderedDict()
        self._key = OrderedDict()
        self._seq_feat = OrderedDict()
        self._count = OrderedDict()
        self._r1 = OrderedDict()
        self._r2 = OrderedDict()
        self._pointer = OrderedDict()
        self._splits = ['train', 'val', 'test']
        for split in self._splits:
            with open(os.path.join(folder, split) + '.pkl') as f:
                self._key[split] = pickle.load(f)
                self._feat[split] = pickle.load(f)
                self._r1[split] = pickle.load(f)
                if split == 'test':
                    self._r2[split] = pickle.load(f)
            with open(os.path.join(folder, split) + '_seq.pkl') as f:
                # some will override
                self._key[split] = pickle.load(f)
                self._count[split] = pickle.load(f)
                self._seq_feat[split] = pickle.load(f)
                self._r1[split] = pickle.load(f)
                if split == 'test':
                    self._r2[split] = pickle.load(f)
                self._pointer[split] = 0

    def get_size(self, partition):
        return len(self._feat[partition])

    def get_n_feat(self):
        return self._feat['train'].shape[1]

    def get_n_seq_feat(self):
        return self._seq_feat['train'].shape[1]

    def reset_pointer(self, partition):
        self._pointer[partition] = 0

    def iterate_batch(self, partition, batch_size):
        batch = []
        current = 0
        while current + batch_size <= len(self._feat[partition]):
            batch_feat = self._feat[partition][current : current + batch_size]
            batch_r1 = self._r1[partition][current : current + batch_size]
            if partition == 'test':
                batch_r2 = self._r2[partition][current : current + batch_size]

            batch_count = self._count[partition][current :
                                                 current + batch_size]
            batch_count = [0] + batch_count
            curr_count = sum(self._count[partition][0 : current])
            accu_count = np.cumsum(batch_count)
            batch_seq_feat = []
            for i in xrange(len(batch_count) - 1):
                batch_seq_feat.append(self._seq_feat[partition][curr_count +
                                                                accu_count[i] :
                                                                curr_count +
                                                                accu_count[i + 1]])

            if partition == 'test':
                yield batch_feat, batch_seq_feat, batch_r1, batch_r2
            else:
                yield batch_feat, batch_seq_feat, batch_r1
            current = current + batch_size
        if current != len(self._feat[partition]):
            batch_feat = self._feat[partition][current : ]
            batch_r1 = self._r1[partition][current : ]
            if partition == 'test':
                batch_r2 = self._r2[partition][current : ]

            batch_count = self._count[partition][current : ]
            batch_count = [0] + batch_count
            curr_count = sum(self._count[partition][0 : current])
            accu_count = np.cumsum(batch_count)
            batch_seq_feat = []
            for i in xrange(len(batch_count) - 1):
                batch_seq_feat.append(self._seq_feat[partition][curr_count +
                                                                accu_count[i] :
                                                                curr_count +
                                                                accu_count[i + 1]])

            if partition == 'test':
                yield batch_feat, batch_seq_feat, batch_r1, batch_r2
            else:
                yield batch_feat, batch_seq_feat, batch_r1

    def next_batch(self, partition, batch_size):
        if self._pointer[partition] + batch_size <= len(self._feat[partition]):
            batch_feat = self._feat[partition][self._pointer[partition]:
                                               self._pointer[partition] + batch_size]
            batch_count = self._count[partition][self._pointer[partition]:
                                                 self._pointer[partition] + batch_size]
            batch_count = [0] + batch_count
            curr_count = sum(self._count[partition][0 : self._pointer[partition]])
            accu_count = np.cumsum(batch_count)
            batch_seq_feat = []
            for i in xrange(len(batch_count) - 1):
                batch_seq_feat.append(self._seq_feat[partition][curr_count +
                                                                accu_count[i] :
                                                                curr_count +
                                                                accu_count[i + 1]])

            batch_r1 = self._r1[partition][self._pointer[partition] :
                                           self._pointer[partition] + batch_size]
            if partition == 'test':
                batch_r2 = self._r2[partition][self._pointer[partition] :
                                               self._pointer[partition] + batch_size]

            # update pointer
            self._pointer[partition] = (self._pointer[partition] + batch_size) \
                                       % len(self._feat[partition])
            if partition == 'test':
                return batch_feat, batch_seq_feat, batch_r1, batch_r2
            else:
                return batch_feat, batch_seq_feat, batch_r1
        else:
            next_pointer = (self._pointer[partition] + batch_size) \
                           % len(self._feat[partition])
            batch_feat = self._feat[partition][self._pointer[partition]:]
            batch_feat = np.append(batch_feat,
                                   self._feat[partition][:next_pointer],
                                   axis = 0)
            batch_r1 = self._r1[partition][self._pointer[partition]:]
            batch_r1 = np.append(batch_r1, self._r1[partition][:next_pointer])
            if partition == 'test':
                batch_r2 = self._r2[partition][self._pointer[partition]:]
                batch_r2 = np.append(batch_r2, self._r2[partition][:next_pointer])

            batch_count = self._count[partition][self._pointer[partition]:]
            batch_count = [0] + batch_count
            accu_count = np.cumsum(batch_count)
            curr_count = sum(self._count[partition][0 : self._pointer[partition]])
            batch_seq_feat = []
            for i in xrange(len(batch_count) - 1):
                batch_seq_feat.append(self._seq_feat[partition][curr_count +
                                                                accu_count[i] :
                                                                curr_count +
                                                                accu_count[i+1]])
            batch_count = self._count[partition][:next_pointer]
            batch_count = [0] + batch_count
            curr_count = 0
            for i in xrange(len(batch_count) - 1):
                batch_seq_feat.append(self._seq_feat[partition][curr_count +
                                                                batch_count[i] :
                                                                curr_count +
                                                                batch_count[i+1]])

            self._pointer[partition] = next_pointer
            if partition == 'test':
                return batch_feat, batch_seq_feat, batch_r1, batch_r2
            else:
                return batch_feat, batch_seq_feat, batch_r1
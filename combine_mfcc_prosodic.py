import pickle as pkl
import numpy as np
import os
path_a = 'data_rand'
path_b = 'data_rand_MFCC'
path_c = 'data_rand_cmb'
for split in ['train_seq','val_seq','test_seq']:
    with open(os.path.join(path_a,split)+'.pkl') as f:
        key_a = pkl.load(f)
        seq_a = pkl.load(f)
        data_a = pkl.load(f)
        r1_a = pkl.load(f)
        r2_a = pkl.load(f)

    with open(os.path.join(path_b,split)+'.pkl') as f:
        key_b = pkl.load(f)
        seq_b = pkl.load(f)
        data_b = pkl.load(f)
        r1_b = pkl.load(f)
        r2_b = pkl.load(f)

    data = np.concatenate((data_a,data_a), axis =1)
    with open(os.path.join(path_c,split) +'.pkl','wb') as f:
        pkl.dump(key_a,f)
        pkl.dump(seq_a,f)
        pkl.dump(data,f)
        pkl.dump(r1_a,f)
        pkl.dump(r2_a,f)

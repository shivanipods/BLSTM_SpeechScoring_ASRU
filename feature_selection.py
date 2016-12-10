__author__ = 'zhouyu'
#!/usr/bin/env python

import pickle as pkl
import os
for split in ['train','val','test']:
    with open(os.path.join('data_rand', split) + '.pkl') as f:
        key = pkl.load(f)
        data = pkl.load(f)
        r1 = pkl.load(f)
        r2 = pkl.load(f)
    data_select = data[:,42:53]
    with open(os.path.join('data_rand_select',split) +'.pkl','wb') as f:
        pkl.dump(key,f)
        pkl.dump(data_select,f)
        pkl.dump(r1,f)
        pkl.dump(r2,f)


#!/usr/bin/env python

import pickle as pkl
import os
import csv
featfile = 'data/features_cleaned.csv'
featfile_old = 'data/features.csv'
feature = csv.DictReader(open(featfile))
feature_old = csv.DictReader(open(featfile_old))

clean = feature.fieldnames
old = feature_old.fieldnames
clean.pop(0)
old.pop(0)
id = []
for clean_i in clean:
    id.append(old.index(clean_i))

for i in range(6):
    id.append(112-5+i)
print id
for split in ['train','val','test']:
    with open(os.path.join('data', split) + '.pkl') as f:
        key = pkl.load(f)
        data = pkl.load(f)
        r1 = pkl.load(f)
        if split == 'test':
            r2 = pkl.load(f)


    with open(os.path.join('data_clean',split)+ '.pkl','w') as f:
        pkl.dump(key,f)
        pkl.dump(data[:,id],f)
        pkl.dump(r1,f)
        if split == 'test':
            pkl.dump(r2,f)

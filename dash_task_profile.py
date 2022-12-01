import sys
sys.path.append("..")
import pathlib
import pickle
import numpy as np
from util import *
import parcel_hierarchy as ph
# import nbformat

with open('data.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    Prop, atlas, cmap, labels, info, profiles, conditions = pickle.load(f)

# Collect parcel profile for each task
label_profile = {}
n_highest = 3
for l,label in enumerate(labels):
    if l != 0:
        parcel_no = labels.tolist().index(label)-1
        profile = ph.show_parcel_profile(parcel_no, profiles, conditions, datasets, show_ds='all', ncond=1, print=False)
        highest_conditions = ['{}:{}'.format(datasets[p][:2], ' & '.join(prof[:n_highest])) for p,prof in enumerate(profile)]
        label_profile[label]=highest_conditions

labels_alpha = sorted(label_profile.keys())

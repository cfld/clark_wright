#!/usr/bin/env python

"""
    main.py
"""

import argparse
import numpy as np
import pandas as pd
from time import time
from tqdm import tqdm
from tsplib95 import parser
from joblib import Parallel, delayed
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist, pdist, squareform

from rsub import *
from seaborn import heatmap
from matplotlib import pyplot as plt

from clark_wright import CW

# --
# IO

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath',       type=str, default='instances/VRPXXL/Antwerp2.txt')
    parser.add_argument('--cw-neighbors', type=int, default=100)
    parser.add_argument('--seed',         type=int, default=123)
    return parser.parse_args()

args = parse_args()

prob = parser.parse(open(args.inpath).read())

cap          = prob['CAPACITY']
n_customers  = prob['DIMENSION'] - 1
xy           = np.array(list(prob['NODE_COORD_SECTION'].values()))
demand       = np.array(list(prob['DEMAND_SECTION'].values())).astype(int)

depot_xy     = xy[0]
customers_xy = xy[1:]
demand       = demand[1:]

# --
# Compute distance matrix

eps = np.random.normal(0, 1e-10, (customers_xy.shape[0], 2))

nn = NearestNeighbors(n_neighbors=args.cw_neighbors + 1).fit(customers_xy + eps)

D, I = nn.kneighbors(customers_xy + eps)
D, I = D[:,1:], I[:,1:]

D_depot = cdist(depot_xy.reshape(1, -1), customers_xy).squeeze()

# --
# Compute savings

def f(alpha, beta):
    def compute_savs(D, D_depot):
        depot2a = D_depot.reshape(-1, 1)
        depot2b = D_depot[I]
        a2b     = D
        return (depot2a / beta) + (depot2b / beta) - a2b ** alpha
        
    savs = compute_savs(D, D_depot).ravel()
    
    srcs  = np.repeat(np.arange(n_customers), args.cw_neighbors)
    dsts  = I.ravel()
    dists = D.ravel()
    
    sel = srcs < dsts
    srcs, dsts, savs, dists = srcs[sel], dsts[sel], savs[sel], dists[sel]
    
    p = np.argsort(-savs, kind='stable')
    srcs, dsts, savs, dists = srcs[p], dsts[p], savs[p], dists[p]
    
    srcs = srcs.astype(int)
    dsts = dsts.astype(int)
    
    edges = {}
    for src, dst, d in zip(srcs, dsts, dists):
        edges[(src, dst)] = d
        edges[(dst, src)] = d
    
    routes = CW(edges, D_depot, demand, cap).run()
    
    return {
        "alpha"      : alpha,
        "beta"       : beta,
        "total_cost" : sum([r['cost'] for r in routes.values()]),
    }

baseline = f(1, 1)['total_cost']

jobs = []
for alpha in np.arange(0.9, 1.1, 0.01):
    for beta in np.logspace(-1, 1, 20):
        job = delayed(f)(alpha=alpha, beta=beta)
        jobs.append(job)

res = Parallel(backend='multiprocessing', n_jobs=40, verbose=10)(jobs)

df = pd.DataFrame(res).round(2)
df.sort_values('total_cost').head(20)

z = df.pivot('alpha', 'beta', 'total_cost')
z[z > np.percentile(z, 50)] = np.percentile(z, 50)
_ = heatmap(z)
show_plot()

_ = plt.plot(z.min(axis=0))
_ = plt.plot(z.min(axis=1))
_ = plt.axhline(baseline, c='red')
show_plot()
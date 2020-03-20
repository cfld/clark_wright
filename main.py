#!/usr/bin/env python

"""
    main.py
"""

import argparse
import numpy as np
from time import time
from tqdm import tqdm
from tsplib95 import parser
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist, pdist, squareform

from clark_wright import CW

# --
# IO

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath',       type=str, default='instances/VRPXXL/Ghent1.txt')
    parser.add_argument('--cw-neighbors', type=int, default=100)
    parser.add_argument('--seed',         type=int, default=123)
    return parser.parse_args()

args = parse_args()

prob = parser.parse(open(args.inpath).read())

cap          = prob['CAPACITY']
n_customers  = prob['DIMENSION'] - 1
xy           = np.array(list(prob['NODE_COORD_SECTION'].values()))
demand       = np.array(list(prob['DEMAND_SECTION'].values())).astype(int)

# >>
xy -= xy[0]
# <<

depot_xy     = xy[0]
customers_xy = xy[1:]
demand       = demand[1:]

print(customers_xy.mean(axis=0))

# --
# Compute distance matrix

from numba import njit
from tqdm import trange

@njit()
def _radial_distance(a, b):
    a = a.reshape(1, -1)
    u = a / np.sqrt((a ** 2).sum(axis=-1)).reshape(1, -1)
    
    proj = (b @ u.T) * u
    
    rad = np.sqrt(((a - proj) ** 2).sum(axis=-1))
    ort = np.sqrt(((b - proj) ** 2).sum(axis=-1))
    
    return np.column_stack((rad, ort))

def radial_pdist(a):
    d = np.stack([_radial_distance(a[i], a) for i in trange(a.shape[0])])
    return d[...,0], d[...,1]


eps      = np.random.normal(0, 1e-10, (customers_xy.shape[0], 2))
rad, ort = radial_pdist(customers_xy + eps)

euc = squareform(pdist(customers_xy + eps))

for alpha in np.linspace(0, 1, 11):
    dist = np.sqrt(2 * (alpha * rad ** 2 + (1 - alpha) * ort ** 2))
    
    I = np.argsort(dist, axis=-1)[:,:args.cw_neighbors + 1]
    assert (I[:,0] == np.arange(I.shape[0])).all()
    I = I[:,1:]

    srcs = np.repeat(np.arange(n_customers), args.cw_neighbors)
    D    = dist[(srcs, I.ravel())].reshape(n_customers, -1)

    D_depot = cdist(depot_xy.reshape(1, -1), customers_xy).squeeze()

    # --
    # Compute savings

    srcs = np.repeat(np.arange(n_customers), args.cw_neighbors)
    dsts = I.ravel()

    savs = D_depot.reshape(-1, 1) + D_depot[I] - D
    savs = savs.ravel()

    dists = D.ravel()

    sel = srcs < dsts
    srcs, dsts, savs, dists = srcs[sel], dsts[sel], savs[sel], dists[sel]

    p = np.argsort(-savs, kind='stable')
    srcs, dsts, savs, dists = srcs[p], dsts[p], savs[p], dists[p]

    srcs = list(srcs.astype(int))
    dsts = list(dsts.astype(int))

    edges = {}
    for src, dst, d in zip(srcs, dsts, dists):
        edges[(src, dst)] = d
        edges[(dst, src)] = d

    # --
    # Run

    routes = CW(edges, D_depot, demand, cap).run()

    total_cost = sum([r['cost'] for r in routes.values()])

    nodes    = [r['nodes'] for r in routes.values()]
    act_cost = sum([D_depot[n[0]] + euc[(n[:-1], n[1:])].sum() + D_depot[n[-1]]
        for n in nodes])

    print(alpha, int(total_cost), int(act_cost))





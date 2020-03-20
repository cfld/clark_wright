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
    parser.add_argument('--inpath',       type=str, default='instances/VRPXXL/Antwerp1.txt')
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
print('total_cost', total_cost)




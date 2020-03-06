#!/usr/bin/env python

"""
    main.py
    
    !! Runs pretty quickly, even on problems w/ 60k nodes
    !! For larger problems, could avoid explicitly computing the distance matrix
"""

import numpy as np
from time import time
from tqdm import tqdm
from tsplib95 import parser
from scipy.spatial.distance import cdist, pdist, squareform

from clark_wright import CW

# --
# IO

inpath = '/Users/bjohnson/projects/routing/RoutingSolver/instances/VRPXXL/Antwerp1.txt'

prob = parser.parse(open(inpath).read())

cap  = prob['CAPACITY']

n_customers  = prob['DIMENSION'] - 1
cw_neighbors = 100

MAX_TOUR_LENGTH = np.inf

coords = list(prob['NODE_COORD_SECTION'].values())
depot, customers = coords[0], coords[1:]

demand = list(prob['DEMAND_SECTION'].values())
demand = demand[1:]
demand = np.array(demand).astype(np.int)

# --
# Compute distance matrix

dist = squareform(pdist(np.array(customers)))
np.fill_diagonal(dist, np.inf)

dist_to_depot = cdist(np.array(depot).reshape(1, -1), customers).squeeze()

# --
# Compute savings

sdist_idx = np.argsort(dist, axis=-1, kind='stable')[:,:cw_neighbors]
sdist_val = np.sort(dist, axis=-1, kind='stable')[:,:cw_neighbors]

saving = dist_to_depot.reshape(-1, 1) + dist_to_depot[sdist_idx] - sdist_val
# saving[saving < 0.1] = 0.1

srcs = np.repeat(np.arange(n_customers), cw_neighbors)
dsts = sdist_idx.ravel()
vals = saving.ravel()

p = np.argsort(-vals, kind='stable')
srcs, dsts, vals = srcs[p], dsts[p], vals[p]

sel = srcs < dsts
srcs, dsts, vals = srcs[sel], dsts[sel], vals[sel]

srcs = list(srcs.astype(int))
dsts = list(dsts.astype(int))
vals = list(vals)

candidates = {}
for src, dst, d in zip(srcs, dsts, dist[(srcs, dsts)]):
    candidates[(src, dst)] = d
    candidates[(dst, src)] = d

# --
# Run

routes = CW(candidates, dist_to_depot, demand, cap).run()

total_cost = sum([r['cost'] for r in routes.values()])
print('total_cost', total_cost)
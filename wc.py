#!/usr/bin/env python

"""
    wc.py
    
    !! Runs pretty quickly, even on problems w/ 60k nodes
    !! For larger problems, could avoid explicitly computing the distance matrix
"""

import numpy as np
from time import time
from tqdm import tqdm
from tsplib95 import parser
from scipy.spatial.distance import cdist, pdist, squareform

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

sdist_idx = np.argsort(dist, axis=-1)[:,:cw_neighbors]
sdist_val = np.sort(dist, axis=-1)[:,:cw_neighbors]

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

dist_lookup = {}
for src, dst, d in zip(srcs, dsts, dist[(srcs, dsts)]):
    dist_lookup[(src, dst)] = d
    dist_lookup[(dst, src)] = d

del dist
del sdist_idx
del sdist_val
del saving

# --
# Clark-Wright functions

def new_route(src, dst):
    global route_idx
    
    load = demand[src] + demand[dst]
    cost = dist_to_depot[src] + dist_lookup[(src, dst)] + dist_to_depot[dst]
    
    if load > cap:
        return
    
    r = {
        "idx"   : route_idx,
        "nodes" : [src, dst],
        "load"  : load,
        "cost"  : cost,
    }
    
    visited.add(src)
    visited.add(dst)
    boundary.add(src)
    boundary.add(dst)
    
    node2route[src] = route_idx
    node2route[dst] = route_idx
    
    routes[route_idx] = r
    route_idx += 1


def extend_route(a, b):
    r = routes[node2route[a]]
    
    new_load = r['load'] + demand[b]
    new_cost = r['cost'] + dist_lookup[(a, b)] + dist_to_depot[b] - dist_to_depot[a]
    
    if new_load > cap:
        return
    
    if r['nodes'][0] == a:
        r['nodes'].insert(0, b)
    elif r['nodes'][-1] == a:
        r['nodes'].append(b)
    else:
        raise Exception
        
    r['load'] = new_load
    r['cost'] = new_cost
    
    node2route[b] = r['idx']
    
    visited.add(b)
    boundary.remove(a)
    boundary.add(b)

def merge_route(src, dst):
    global route_idx
    
    if node2route[src] == node2route[dst]:
        return
    
    r_src = routes[node2route[src]]
    r_dst = routes[node2route[dst]]
    
    new_load = r_src['load'] + r_dst['load']
    new_cost = r_src['cost'] + r_dst['cost'] + dist_lookup[(src, dst)] - dist_to_depot[src] - dist_to_depot[dst]
    
    if new_load > cap:
        return
    
    # reverse direction to fit
    if r_src['nodes'][-1] != src:
        r_src['nodes'] = r_src['nodes'][::-1]
    
    if r_dst['nodes'][0] != dst:
        r_dst['nodes'] = r_dst['nodes'][::-1]
    
    r = {
        "idx"   : route_idx,
        "nodes" : r_src['nodes'] + r_dst['nodes'],
        "load"  : new_load,
        "cost"  : new_cost,
    }
    
    del routes[node2route[src]]
    del routes[node2route[dst]]
    for n in r['nodes']:
        node2route[n] = route_idx
    
    boundary.remove(src)
    boundary.remove(dst)
    
    routes[route_idx] = r
    route_idx += 1


routes   = {}
visited  = set([])
boundary = set([])

node2route = {}

route_idx = 0

t = time()
for (src, dst, val) in zip(srcs, dsts, vals):
    if (src in visited) and (src not in boundary):
        pass
    
    elif (dst in visited) and (dst not in boundary):
        pass
    
    elif (src not in visited) and (dst not in visited):
        new_route(src, dst)
    
    elif (src in boundary) and (dst not in visited):
        extend_route(src, dst)
    
    elif (dst in boundary) and (src not in visited):
        extend_route(dst, src)
    
    elif (src in boundary) and (dst in boundary):
        merge_route(src, dst)
    
    else:
        raise Exception

print(time() - t)

# fix customers that haven't been visited
if len(visited) != n_customers:
    for n in range(n_customers):
        if n not in visited:
            routes[route_idx] = {
                "idx"    : route_idx,
                "nodes"  : [n],
                "load"   : demand[n],
                "cost"   : 2 * dist_to_depot[n],
            }
            route_idx += 1

total_cost = sum([r['cost'] for r in routes.values()])
print(total_cost)

# 498791.9977090355
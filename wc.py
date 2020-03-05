#!/usr/bin/env python

"""
    wc.py
"""

import numpy as np
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
dist_to_depot = dist_to_depot


# --
# Compute savings

sdist_idx = np.argsort(dist, axis=-1)[:,:cw_neighbors]
sdist_val = np.sort(dist, axis=-1)[:,:cw_neighbors]

saving = dist_to_depot.reshape(-1, 1) + dist_to_depot[sdist_idx] - sdist_val
# saving[saving < 0.1] = 0.1

row = np.repeat(np.arange(n_customers), cw_neighbors)
col = sdist_idx.ravel()
val = saving.ravel()

p = np.argsort(-val, kind='stable')
row, col, val = row[p], col[p], val[p]
sav = np.column_stack([row, col, val])

sav = sav[sav[:,0] <= sav[:,1]]

sav[:10]

# --

def new_route(src, dst):
    global route_idx
    
    load = demand[src] + demand[dst]
    if load > cap:
        return
    
    r = {
        "idx"    : route_idx,
        "nodes"  : [0, src, dst, 0],
        "length" : 2,
        "load"   : load,
        "cost"   : dist_to_depot[src] + dist[src, dst] + dist_to_depot[dst],
    }
    
    # countVisits += 2
    
    visited.add(src)
    visited.add(dst)
    
    extension.add(src)
    extension.add(dst)
    
    node2route[src] = route_idx
    node2route[dst] = route_idx
    
    routes[route_idx] = r
    route_idx += 1


def extend_route(a, b):
    r = routes[node2route[a]]
    
    if r['load'] + demand[b] > cap:
        return
    
    costs_add = dist[a, b] + dist_to_depot[b] - dist_to_depot[a]
    if r['cost'] + costs_add > MAX_TOUR_LENGTH:
        return
    
    if r['nodes'][1] == a:
        r['nodes'].insert(1, b)
    elif r['nodes'][-2] == a:
        r['nodes'].insert(-1, b)
    else:
        raise Exception
        
    r['length'] += 1
    r['load']   += demand[b]
    r['cost']   += costs_add
    
    node2route[b] = r['idx']
    
    extension.remove(a)
    extension.add(b)
    interior.add(a)
    visited.add(b)


def merge_route(src, dst):
    global route_idx
    
    if node2route[src] == node2route[dst]:
        return
    
    r_src = routes[node2route[src]]
    r_dst = routes[node2route[dst]]
    
    if r_src['load'] + r_dst['load'] > cap:
        return
    
    cost_merge = r_src['cost'] + r_dst['cost'] + dist[src, dst] - dist_to_depot[src] - dist_to_depot[dst]
    if cost_merge > MAX_TOUR_LENGTH:
        return
    
    if r_src['nodes'][1] == src:
        r_src['nodes'] = r_src['nodes'][::-1]
    if r_dst['nodes'][1] != dst:
        r_dst['nodes'] = r_dst['nodes'][::-1]
    
    r = {
        "idx"    : route_idx,
        "nodes"  : r_src['nodes'][:-1] + r_dst['nodes'][1:],
        "length" : r_src['length'] + r_dst['length'],
        "load"   : r_src['load'] + r_dst['load'],
        "cost"   : cost_merge,
    }
    
    del routes[node2route[src]]
    del routes[node2route[dst]]
    for n in r['nodes'][1:-1]:
        node2route[n] = route_idx
    
    extension.remove(src)
    extension.remove(dst)
    interior.add(src)
    interior.add(dst)
    
    routes[route_idx] = r
    route_idx += 1


def check_cost(r):
    z = r['nodes'][1:-1]
    c = sum([dist[a, b] for a,b in zip(z[:-1], z[1:])]) + dist_to_depot[z[0]] + dist_to_depot[z[-1]]
    assert np.allclose(r['cost'], c)


routes    = {}
visited   = set([])
extension = set([])
interior  = set([])

node2route = {}

route_idx = 0

for (src, dst, val) in tqdm(sav):
    src = int(src)
    dst = int(dst)
    
    if (src in interior) or (dst in interior):
        continue
    
    src_ext = src in extension
    dst_ext = dst in extension
    
    if (not src_ext) and (not dst_ext):
        # create a new route
        new_route(src, dst)
    elif (src_ext) and (not dst_ext):
        # add point to route
        extend_route(src, dst)
    elif (not src_ext) and (dst_ext):
        # add point to route
        extend_route(dst, src)
    else:
        # merge routes
        assert src_ext and dst_ext
        merge_route(src, dst)

for r in routes.values():
    check_cost(r)


if len(visited) != n_customers:
    for n in range(n_customers):
        if n not in visited:
            routes[route_idx] = {
                "idx"    : route_idx,
                "nodes"  : [0, n, 0],
                "length" : 1,
                "load"   : demand[n],
                "cost"   : 2 * dist_to_depot[n],
            }
            route_idx += 1

498791.99770903547

class CW:
    def __init__(self, edges, D_depot, demand, cap):
        self.edges = edges
        self.D_depot     = D_depot
        self.demand      = demand
        self.cap         = cap
        
        self.visited    = set([])
        self.boundary   = set([])
        self.routes     = {}
        self.node2route = {}
        self.route_idx  = 0
        
    def _new_route(self, src, dst):
        load = self.demand[src] + self.demand[dst]
        cost = self.D_depot[src] + self.edges[(src, dst)] + self.D_depot[dst]
        
        if load > self.cap:
            return
        
        self.visited.add(src)
        self.visited.add(dst)
        self.boundary.add(src)
        self.boundary.add(dst)
        
        self.node2route[src] = self.route_idx
        self.node2route[dst] = self.route_idx
        
        self.routes[self.route_idx] = {
            "idx"   : self.route_idx,
            "nodes" : [src, dst],
            "load"  : load,
            "cost"  : cost,
        }
        
        self.route_idx += 1
        
    def _extend_route(self, a, b):
        r = self.routes[self.node2route[a]]
        
        new_load = r['load'] + self.demand[b]
        new_cost = r['cost'] + self.edges[(a, b)] + self.D_depot[b] - self.D_depot[a]
        
        if new_load > self.cap:
            return
            
        self.visited.add(b)
        self.boundary.remove(a)
        self.boundary.add(b)
        
        if r['nodes'][0] == a:
            r['nodes'].insert(0, b)
        elif r['nodes'][-1] == a:
            r['nodes'].append(b)
        else:
            raise Exception('not in right position')
            
        r['load'] = new_load
        r['cost'] = new_cost
        
        self.node2route[b] = r['idx']
        
    def _merge_route(self, src, dst):
        r_src = self.routes[self.node2route[src]]
        r_dst = self.routes[self.node2route[dst]]
        
        new_load = r_src['load'] + r_dst['load']
        new_cost = r_src['cost'] + r_dst['cost'] + self.edges[(src, dst)] - self.D_depot[src] - self.D_depot[dst]
        
        if new_load > self.cap:
            return
            
        self.boundary.remove(src)
        self.boundary.remove(dst)
        
        # reverse direction to fit
        if r_src['nodes'][-1] != src:
            r_src['nodes'] = r_src['nodes'][::-1]
        
        if r_dst['nodes'][0] != dst:
            r_dst['nodes'] = r_dst['nodes'][::-1]
        
        del self.routes[self.node2route[src]]
        del self.routes[self.node2route[dst]]
        
        r = {
            "idx"   : self.route_idx,
            "nodes" : r_src['nodes'] + r_dst['nodes'],
            "load"  : new_load,
            "cost"  : new_cost,
        }
        for n in r['nodes']:
            self.node2route[n] = self.route_idx
        
        self.routes[self.route_idx] = r
        self.route_idx += 1
    
    def _fix_unvisited(self):
        # fix customers that haven't been visited
        for n in range(self.demand.shape[0]):
            if n not in self.visited:
                self.routes[self.route_idx] = {
                    "idx"    : self.route_idx,
                    "nodes"  : [n],
                    "load"   : self.demand[n],
                    "cost"   : 2 * self.D_depot[n],
                }
                self.route_idx += 1
    
    def run(self):
        for (src, dst) in self.edges.keys():
            
            src_visited  = src in self.visited
            dst_visited  = dst in self.visited
            src_boundary = src in self.boundary
            dst_boundary = dst in self.boundary
            
            if src > dst:
                pass
            
            if src_visited and not src_boundary:
                pass
            
            elif dst_visited and not dst_boundary:
                pass
            
            elif not src_visited and not dst_visited:
                self._new_route(src, dst)
            
            elif src_boundary and not dst_visited:
                self._extend_route(src, dst)
            
            elif dst_boundary and not src_visited:
                self._extend_route(dst, src)
            
            elif src_boundary and dst_boundary and (self.node2route[src] != self.node2route[dst]):
                self._merge_route(src, dst)
            
            else:
                pass
        
        self._fix_unvisited()
        
        return self.routes



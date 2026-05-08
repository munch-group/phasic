from phasic import Graph, StateIndexer, Property, with_ipv
import numpy as np                                                                         
from itertools import combinations_with_replacement

def clone_with_ipv(graph, ipv):
    """         
    clone but with new IPV. IPV must be an array/list of same length as number of graph vertices.
    """
    # ipv = _validate_ipv(ipv, allow_single_state=False)
    assert len(ipv) == graph.vertices_length()
    assert ipv[graph.starting_vertex().index()] == 0, "IPV cannot specify transition to itself"

    new = Graph(graph.state_length())                                                              
    if graph.parameterized():                                                                      
        new.set_param_length(graph.param_length())
                                                                                                
    # vertices
    starting_vertex = graph.starting_vertex()
    vmap = {starting_vertex.index(): new.starting_vertex()}                                   
    for v in graph.vertices():
        if v.index() == starting_vertex.index():
            continue                                                                            
        vmap[v.index()] = new.create_vertex(list(v.state()))

    # IPV edges
    for i in range(graph.vertices_length()):
        if ipv[i]:
            new.starting_vertex().add_edge(vmap[i], ipv[i])

    # remaining edges
    for v in graph.vertices():                                                                     
        if not v.edges():
            assert ipv[v.index()] == 0, "IPV cannot specify transitions to absorbing states."
            continue
        if v.index() == starting_vertex.index():
            continue                                                                            
        nv = vmap[v.index()]
        if graph.parameterized():
            for e in v.parameterized_edges():
                nv.add_edge(vmap[e.to().index()], list(e.edge_state(graph.param_length())))
        else:
            for e in v.edges():                                                                 
                nv.add_edge(vmap[e.to().index()], e.weight())
                                                                                                
    new.is_discrete = graph.is_discrete
    new._cache_trace = graph._cache_trace
    new._trace = None                                                                           
    new._trace_dirty = True
    new._last_theta = None        
    new._joint_prob_base_graph_indexer = graph._joint_prob_base_graph_indexer                                                              
    new._rewarded_props = graph._rewarded_props    
    
    return new  


def epoch_joint_prob_graph(jg, t, theta):  
    """
    Joint prob graph IPV at time t. Made special so that t-states in the join graph are never left once entered.
    """
                                                                                       
    new = Graph(jg.state_length())                                                              
    if jg.parameterized():                                                                      
        new.set_param_length(jg.param_length())
                                                                                                
    start_old = jg.starting_vertex()
    vmap = {start_old.index(): new.starting_vertex()}                                           

    def is_trash(v):
        if not v.state().sum() and v.edges_length() == 1:
            child = v.edges()[0].to()
            if not v.state().sum() and child.edges_length() == 1:
                return child.edges()[0].to().index() == v.index()
        return False

    t_vertex_indices = []
    trash_indices = []
    for v in jg.vertices():
        if v.index() == start_old.index():
            continue
        if not v.edges():
            abs_index = v.index()  

        for edge in v.edges():
            if len(edge.to().edges()) == 0:
                t_vertex_indices.append(v.index())
                break

        if is_trash(v):
            trash_indices.append(v.index())
        vmap[v.index()] = new.create_vertex(list(v.state()))

    t_vertex_indices = np.unique(t_vertex_indices)
    assert len(trash_indices) == 2

    t_vertex_sisters = dict()

    # add edges 
    for v in jg.vertices():                                                                     
        if not v.edges():
            continue
        if v.index() in trash_indices:
            continue 

        nv = vmap[v.index()]

        if v.index() in t_vertex_indices:
            t_vertex_sister = new.create_vertex([0]*jg.state_length())
            nv.add_edge(t_vertex_sister, [1]*jg.param_length())
            t_vertex_sister.add_edge(nv, [1]*jg.param_length())
            t_vertex_sisters[v.index()] = t_vertex_sister.index()
            continue

        for e in v.parameterized_edges():
            to_index = e.to().index()
            if to_index in trash_indices:
                to_index = abs_index
            if jg.parameterized():
                nv.add_edge(vmap[to_index], list(e.edge_state(jg.param_length())))
            else:
                nv.add_edge(vmap[to_index], e.weight())
                                                                                                
    new.is_discrete = jg.is_discrete
    new._cache_trace = jg._cache_trace
    new._trace = None                                                                           
    new._trace_dirty = True
    new._last_theta = None    
    new._joint_prob_base_graph_indexer = jg._joint_prob_base_graph_indexer   
    new._rewarded_props = jg._rewarded_props                                                           

    new.update_weights(theta)

    st_with_sisters = np.array(new.stop_probability(t))

    idx = np.array(sorted(v.index() for v in vmap.values()))

    st = st_with_sisters[idx]
    for i in t_vertex_sisters:
        st[i] += st_with_sisters[t_vertex_sisters[i]]
        # st[i] += st_with_sisters[i]

    st = np.array(st)

    return st


def coalescent(state, indexer=None):
    transitions = []
    for i, j in combinations_with_replacement(range(indexer.lineages.state_length), 2):
        same = int(i == j)
        if same and state[i] < 2:
            continue
        if not same and (state[i] < 1 or state[j] < 1):
            continue
        new = state.copy()
        new[i] -= 1
        new[j] -= 1
        new[min(i + j + 1, state.size - 1)] += 1
        pair_count = state[i] * (state[j] - same) / (1 + same)
        transitions.append([new, [pair_count]])
    return transitions


N_SAMPLES = 4
EPOCH_BOUNDARY = 1
MUTATION_RATE = 1
REWARD_LIMIT = 5 
TRUE_THETA = 7

indexer = StateIndexer(
    lineages=[Property('descendants', min_value=1, max_value=N_SAMPLES)],
)
ipv = [0] * indexer.state_length
ipv[indexer.lineages.props_to_index(descendants=1)] = N_SAMPLES  # start with N_SAMPLES singletons
graph = Graph(coalescent, ipv=ipv, indexer=indexer)

# VANILLA
_disc_joint_graph = graph.joint_prob_graph(mutation_rate=MUTATION_RATE, reward_limit=REWARD_LIMIT, discrete=True) 
_disc_joint_graph.update_weights([TRUE_THETA, 1])
df = _disc_joint_graph.joint_prob_table()
t_indices = df.index.to_numpy()
print('VANILLA:', df.prob.to_list())

# DAISY SINGLE EPOCH
_cont_joint_graph = graph.joint_prob_graph(mutation_rate=MUTATION_RATE, reward_limit=REWARD_LIMIT, discrete=False) 
_cont_joint_graph.update_weights([TRUE_THETA, 1])
epoch_ipv = epoch_joint_prob_graph(_cont_joint_graph, t=10, theta=[TRUE_THETA, 1]).tolist()  
print('DAISY:', sorted(epoch_ipv, reverse=True)[:len(df.prob)])

# DAISY TWO EPOCHS SAME THETA
# first:
_cont_joint_graph = graph.joint_prob_graph(mutation_rate=MUTATION_RATE, reward_limit=REWARD_LIMIT, discrete=False) 
epoch_ipv = epoch_joint_prob_graph(_cont_joint_graph, t=EPOCH_BOUNDARY, theta=[TRUE_THETA, 1]).tolist()  
# final:
_cont_joint_graph_epoch = clone_with_ipv(_cont_joint_graph, epoch_ipv)
stop_probs = epoch_joint_prob_graph(_cont_joint_graph_epoch, t=20, theta=[TRUE_THETA, 1]).tolist()  
print('DAISY TWO EPOCHS:', sorted(stop_probs, reverse=True)[:len(df.prob)])

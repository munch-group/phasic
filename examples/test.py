
from phasic import Graph, with_ipv, ExpStepSize # ALWAYS import phasic first to set jax backend correctly
import numpy as np
import jax.numpy as jnp
import sys
from tqdm.auto import tqdm
import pandas as pd
from typing import Optional

def joint_prob_reward_callback(state, current_rewards=None, 
                               mutation_rate=1, reward_limit=10, 
                               tot_reward_limit=np.inf):

    reward_limits = np.repeat(reward_limit, len(state))
    
    reward_dims = len(reward_limits)
    if current_rewards is None:
        current_rewards = np.zeros(reward_dims)

    reward_rates = np.zeros(reward_dims)
    trash_rate = 0
    
    for i in range(reward_dims):
        rate = state[i] * mutation_rate 
        r = np.zeros(reward_dims)
        r[i] = 1
        if np.all(current_rewards + r <= reward_limits) and np.sum(current_rewards + r) <= tot_reward_limit:
            reward_rates[i] = rate
        else:
            trash_rate = trash_rate + rate

    return np.append(reward_rates, trash_rate)


def joint_prob_graph(graph, reward_rates_callback, mutation_rate:Optional[float]=None,
                     reward_limit:Optional[int]=0, tot_reward_limit:Optional[float]=np.inf):

    starting_vertex = graph.starting_vertex()
    reward_dims = len(reward_rates_callback(starting_vertex.state(), mutation_rate=mutation_rate,
                                            reward_limit=reward_limit, tot_reward_limit=tot_reward_limit )) - 1 # a bit of a hack. -1 to not count trash rate...

    orig_state_vector_length = len(graph.vertex_at(1).state())
    state_vector_length = orig_state_vector_length + reward_dims

    state_indices = np.arange(orig_state_vector_length)
    joint_reward_state_indices = np.arange(orig_state_vector_length,
                                           state_vector_length)

    new_graph = Graph(state_vector_length)
    new_starting_vertex = new_graph.starting_vertex()

    null_rewards = np.zeros(reward_dims)

    index = 0

    # Get param_length for extracting parameterized edge coefficients
    param_length = graph.param_length()

    # Use parameterized_edges() to preserve parameterization
    for edge in starting_vertex.parameterized_edges():
        coeffs = list(edge.edge_state(param_length))
        # Pad with 0.0 for mutation dimension: [c0] → [c0, 0.0]
        coeffs.append(0.0)
        new_starting_vertex.add_edge(
          new_graph.find_or_create_vertex(
              np.append(edge.to().state(), null_rewards).astype(int)),
          coeffs)

    prev_completion = 0
    pbar = tqdm(position=0, total=1, miniters=0, desc='visited/created', bar_format='{l_bar}{bar}')

    index = index + 1
    
    trash_rates = {}
    t_vertex_indices = np.array([], dtype=int)
    while index < new_graph.vertices_length():

        new_vertex = new_graph.vertex_at(index)
        new_state = new_vertex.state()
        state = new_vertex.state()[state_indices]
        vertex = graph.find_vertex(state)

        for edge in vertex.parameterized_edges():
            new_child_state = np.append(
                edge.to().state(),
                new_state[joint_reward_state_indices]
                )

            if np.all(new_state == new_child_state):
                continue

            new_child_vertex = new_graph.find_or_create_vertex(
                new_child_state)
            coeffs = list(edge.edge_state(param_length))
            # Pad with 0.0 for mutation dimension
            coeffs.append(0.0)
            new_vertex.add_edge(new_child_vertex, coeffs)

            if not graph.find_vertex(new_child_state[state_indices]).edges():
                t_vertex_indices = np.append(t_vertex_indices, new_child_vertex.index()) 

        current_state = new_state[state_indices]
        current_rewards = new_state[joint_reward_state_indices]
        rates = reward_rates_callback(current_state, current_rewards, 
                                    mutation_rate=mutation_rate, 
                                    reward_limit=reward_limit, 
                                    tot_reward_limit=tot_reward_limit) 


        trash_rates[index] = rates[reward_dims]
        for i in range(reward_dims):
            rate = rates[i]
            if rate > 0:
                new_rewards = current_rewards.copy()
                new_rewards[i] = new_rewards[i] + 1
                new_child_state = np.append(current_state, new_rewards)

                if not graph.find_vertex(new_child_state[state_indices]).edges():
                    continue

                new_child_vertex = new_graph.find_or_create_vertex(new_child_state)
                # Mutation edges: [0.0, rate] → weight = 0.0*theta[0] + rate*theta[1]
                # When theta[1]=1.0 (fixed), weight = rate (constant!)
                new_vertex.add_edge(new_child_vertex, [0.0, rate])
                
        index = index + 1 

        completion = index/new_graph.vertices_length()
        pbar.update(completion - prev_completion)
        prev_completion = completion

    pbar.close()

    trash_vertex = new_graph.find_or_create_vertex(np.repeat(0, state_vector_length))
    trash_loop_vertex = new_graph.create_vertex(np.repeat(0, state_vector_length))
    # Constant edges: [0.0, 1.0] → weight = 0.0*theta[0] + 1.0*theta[1] = 1.0
    trash_vertex.add_edge(trash_loop_vertex, [0.0, 1.0])
    trash_loop_vertex.add_edge(trash_vertex, [0.0, 1.0])

    for i, rate in trash_rates.items():
        if rate > 0:
            new_graph.vertex_at(i).add_edge(trash_vertex, [0.0, rate])

    new_absorbing = new_graph.create_vertex(np.repeat(0, state_vector_length))
    t_vertex_indices = np.unique(t_vertex_indices)
    for i in t_vertex_indices:
        new_graph.vertex_at(i).add_edge(new_absorbing, [0.0, 1.0])

    weights_were_multiplied_with = new_graph.normalize()

    return new_graph


def joint_prob_table(joint_graph, obs2idx):

    idx2obs = {v: k for k, v in obs2idx.items()}
    assert len(idx2obs) == len(obs2idx)

    t_indices = list(idx2obs.keys())
    sojourn_times = joint_graph.expected_sojourn_time(t_indices)
    assert len(sojourn_times) == len(t_indices)
    records = []
    for idx, prob in zip(t_indices, sojourn_times):
        obs = idx2obs[idx]
        records.append([*obs, prob])
    joint_probs = pd.DataFrame(records, columns=list(range(1, nr_samples+1)) + ['prob'])
    return joint_probs


nr_samples = 4

@with_ipv([nr_samples]+[0]*(nr_samples-1))
def coalescent_1param(state):
    transitions = []
    for i in range(state.size):
        for j in range(i, state.size):            
            same = int(i == j)
            if same and state[i] < 2:
                continue
            if not same and (state[i] < 1 or state[j] < 1):
                continue 
            new = state.copy()
            new[i] -= 1
            new[j] -= 1
            new[i+j+1] += 1
            transitions.append([new, [state[i]*(state[j]-same)/(1+same)]])
    return transitions

true_theta = [7]
base_graph = Graph(coalescent_1param)
base_graph.update_weights(true_theta)
base_graph.plot()
joint_graph = joint_prob_graph(base_graph, 
                               joint_prob_reward_callback, 
                               mutation_rate=0.1, 
                               reward_limit=3)


def coalescent_obs2idx_map(graph, base_graph_state_length):
    t_vertex_indices = []
    for vertex in graph.vertices():
        for edge in vertex.edges():
            if len(edge.to().edges()) == 0:
                t_vertex_indices.append(vertex.index())
                break
    t_vertex_indices = np.unique(t_vertex_indices)
    states = graph.states()
    joint_reward_state_indices = np.arange(base_graph_state_length, graph.state_length())
    state_reward_matrix = states[t_vertex_indices, :][:, joint_reward_state_indices]
    mapping = {}
    for rewards, idx in zip(state_reward_matrix, t_vertex_indices):
        mapping[tuple(rewards.tolist())] = int(idx)
    return mapping


obs2idx = coalescent_obs2idx_map(joint_graph, base_graph.state_length())
joint = joint_prob_table(joint_graph, obs2idx)

probs = joint.loc[1:, 'prob'].to_numpy()
modelled_obs = joint.iloc[1:, :-1].to_numpy()
rng = np.random.default_rng()
observations = rng.choice(modelled_obs, 10000, axis=0, replace=True, p=probs/probs.sum()).tolist()

obs_indices = [obs2idx[tuple(o)] for o in observations]

def uninformative_prior(phi):
    """Uninformative prior: φ ~ N(0, 10^2) - very wide"""
    mu = 0.0
    sigma = 10.0
    return -0.5 * jnp.sum(((phi - mu) / sigma)**2)

step_schedule = ExpStepSize(first_step=0.1, last_step=0.01, tau=50.0)

params = dict(
    bandwidth = 'median',             
    theta_dim = len(true_theta),      # number of model parameters
    prior = uninformative_prior,      # prior on parameters
    n_particles = 50,                 # number of particles
    n_iterations = 200,               # number of optimization steps
    learning_rate = step_schedule,    # step size schedule
    seed = 42,                        # random seed
    verbose = False,                  # print what it is doing
    progress = True,                  # show progress bar
    discrete = False,                
)
svgd = joint_graph.svgd(
    obs_indices,
    joint_index=True,
    theta_dim=2,           # Full parameter space: [coalescent_rate, mutation_rate]
    fixed=[0, 1],          # 0=optimize coalescent, 1=fix mutation at 1.0
    progress=True,
    n_particles=100,
    n_iterations=200
)

svgd.summary()
print("True theta:", true_theta)
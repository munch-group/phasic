"""
Tests XXXX
"""

from phasic import Graph, with_ipv
import numpy as np
from pytest import approx, raises

nr_samples = 4

np.random.seed(42)
@with_ipv([nr_samples]+[0]*(nr_samples-1))
def coalescent(state):
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
            transitions.append((new, [state[i]*(state[j]-same)/(1+same)]))
    return transitions


class TestRewardTransform:
    """Test reward transformation"""

    def test_singleton_transform(self):

        graph = Graph(coalescent)
        rt_graph = graph.reward_transform([0, 4, 2, 0, 1, 0])
        assert rt_graph.expectation() == approx(2)

    def test_doubleton_transform(self):

        graph = Graph(coalescent)
        rt_graph = graph.reward_transform([0, 0, 1, 2, 0, 0])
        assert rt_graph.expectation() == approx(1)

    def test_tripleton_transform(self):

        graph = Graph(coalescent)
        rt_graph = graph.reward_transform([0, 0, 0, 0, 1, 0])
        assert rt_graph.expectation() == approx(2/3)        

class TestRewardedExpectation:
    """Test rewards"""

    def test_rewarded_expectation(self):

        graph = Graph(coalescent)
        assert graph.expectation([0, 4, 2, 0, 1, 0]) == approx(2)
        assert graph.expectation([0, 0, 1, 2, 0, 0]) == approx(1)
        assert graph.expectation([0, 0, 0, 0, 1, 0]) == approx(2/3)



# ## Forward computation in steps:

# - **accumulated_occupancy(...)** calls **accumulated_visits** for discrete  graphs, **accumulated_visiting_time** for continuous graphs
# - **state_probability(t)**. The probability vector p(t) where p(t)[i] = probability of being in transient state i at time t (continuous) or after n jumps (discrete). 

# ## Graph elimination

# - **expected_waiting_time(rewards)** Computes E[total accumulated reward before absorption] via graph elimination. With the default reward vector (all 1s), this gives E[T], the expected total time until absorption. The result is a vector of length n_vertices where result[0] (the starting vertex) holds the final answer. The other entries are intermediate quantities from the elimination. This is the workhorse behind moments(). Higher moments are computed by iterating reward transformations.
# - **expected_sojourn_time()**. Computes E[time in state i before absorption] for every state, starting from the initial distribution. Implemented by running expected_waiting_time with n different unit reward vectors (one-hot for each state) in a single batched pass. So expected_sojourn_time()[i] = the expected total time the chain spends in state i over its entire lifetime. You can think of it as accumulated_occupancy for infinite t.



class TestRewardedExpectedWaitingTime:
    """Test expected waiting time"""

    def test_expected_waiting_time(self):
        ...


class TestRewardedStateProbability:
    """Test state probability"""

    def test_state_probability_XXXX(self):
        ...



class TestRewardedExpectedSojournTime:
    """Test expected sojourn time"""

    def test_expected_sojourn_time_XXXX(self):
        ...

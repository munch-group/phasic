"""
Tests XXXX
"""

class TestRewards:
    """Test rewards"""

    def test_expectation_rewards(self):
        ...



# ## Forward computation in steps:

# - **accumulated_occupancy(...)** calls **accumulated_visits** for discrete  graphs, **accumulated_visiting_time** for continuous graphs
# - **state_probability(t)**. The probability vector p(t) where p(t)[i] = probability of being in transient state i at time t (continuous) or after n jumps (discrete). 

# ## Graph elimination

# - **expected_waiting_time(rewards)** Computes E[total accumulated reward before absorption] via graph elimination. With the default reward vector (all 1s), this gives E[T], the expected total time until absorption. The result is a vector of length n_vertices where result[0] (the starting vertex) holds the final answer. The other entries are intermediate quantities from the elimination. This is the workhorse behind moments(). Higher moments are computed by iterating reward transformations.
# - **expected_sojourn_time()**. Computes E[time in state i before absorption] for every state, starting from the initial distribution. Implemented by running expected_waiting_time with n different unit reward vectors (one-hot for each state) in a single batched pass. So expected_sojourn_time()[i] = the expected total time the chain spends in state i over its entire lifetime. You can think of it as accumulated_occupancy for infinite t.



class TestExpectedWaitingTime:
    """Test expected waiting time"""

    def test_expected_waiting_time(self):
        ...


class TestStateProbability:
    """Test state probability"""

    def test_state_probability_XXXX(self):
        ...


class TestExpectedWaitingTime:
    """Test expected waiting time"""

    def test_expected_waiting_time_XXXX(self):
        ...


class TestExpectedSojournTime:
    """Test expected sojourn time"""

    def test_expected_sojourn_time_XXXX(self):
        ...

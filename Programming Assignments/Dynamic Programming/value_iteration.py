import numpy as np
import pprint
import sys
if "../" not in sys.path:
    sys.path.append("../")
from lib.envs.gridworld import GridworldEnv


pp = pprint.PrettyPrinter(indent=2)
env = GridworldEnv()

def value_iteration(env, theta=0.0001, discount_factor=1.0):
    """
    Value Iteration Algorithm

    Args:
        env: OpenAI env. env.P represents the transition probabilities of the environment
        env.P[s][a] is a list of transition tuples (prob, next_state, reward, done)
        env.nS is the number of states in the environment
        env.nA is the number of actions in the environment
    theta: We stop evaluation once our value function change is less than theta for all states
    discount_factor: Gamma discount factor

    Returns:
        A tuple (policy, V) of the optimal policy and the optimal value function
    """

    def one_step_lookahead(state, V):
        """
        Helper function to calculate the value for all actions in a given state

        Args:
            state: The state to consider (int)
            V: The value to use as an estimator
        """
        A = np.zeros(env.nA)
        for a in range(env.nA):
            for prob, next_state, reward, done in env.P[state][a]:
                A[a] += prob * (reward + discount_factor * V[next_state])
        return A

    V = np.zeros(env.nS)
    # Update value function
    while True:
        # stopping condition
        delta = 0
        # Update each state
        for s in range(env.nS):
            # Find the best action by one-step lookahead
            A = one_step_lookahead(s, V)
            best_action_value = np.max(A)
            # Calculate delta across all states seen so far
            delta = max(delta, np.abs(best_action_value - V[s]))
            # Update the value function
            V[s] = best_action_value
        if delta < theta:
            break

    # Create a deterministic policy using the optimal value function
    policy = np.zeros([env.nS, env.nA])
    for s in range(env.nS):
        # Find the best action by one-step lookahead
        A = one_step_lookahead(s, V)
        best_action = np.argmax(A)
        # Always take the best action
        policy[s, best_action] = 1.0

    return policy, V

policy, v = value_iteration(env)
print("Policy Probability Distribution:")
print(policy)
print("")

print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
print(np.reshape(np.argmax(policy, axis=1), env.shape))
print("")

print("Value Function:")
print(v)
print("")

print("Reshaped Grid Value Function:")
print(v.reshape(env.shape))
print("")

# Test the value function
expected_v = np.array([ 0, -1, -2, -3, -1, -2, -3, -2, -2, -3, -2, -1, -3, -2, -1,  0])
np.testing.assert_array_almost_equal(v, expected_v, decimal=2)

import numpy as np
import sys
if "../" not in sys.path:
    sys.path.append("../")
from lib.envs.gridworld import GridworldEnv

env = GridworldEnv()

def state_action_value(s, a, action_prob, discount_factor, v, V):
    for prob, next_state, reward, done in env.P[s][a]:
        # Calculate the expected value
        v += action_prob * prob * (reward + discount_factor * V[next_state])

    return v

# Taken from policy_evaluation.py
def policy_eval(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics

    Args:
        policy: [S, A] shaped matrix representing the policy
        env: OpenAI env. env.P represents the transition probabilities of the environment
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is the number of of states in the environment
            env.nA is the number of of actions in the environment
        theta: Stop evaluation once the value function change is less than theta for all states.
        discount_factor: Gamma discount factor

    Returns:
        Vector of length env.nS representing the value function
    """

    # Start with a random (all 0) value function
    V = np.zeros(env.nS)

    while True:
        old_v = V
        delta = 0
        # For each state, perform a "full backup"
        for s in range(env.nS):
            v = 0
            # Look at the possible next actions
            for a, action_prob in enumerate(policy[s]):
                # For each action, look at the possible next states
                for  prob, next_state, reward, done in env.P[s][a]:
                    # Calculate the expected value
                    v += action_prob * prob * (reward + discount_factor * V[next_state])

            # How much the value function changed
            # Max over all states
            delta = max(delta, np.abs(v - V[s]))
            V[s] = v

        # Stop evaluation once the value function change
        # is less than theta for all states
        if delta < theta:
            break

    return np.array(V)

random_policy = np.ones([env.nS, env.nA]) / env.nA
v = policy_eval(random_policy, env)

# Test: Make sure the evaluated policy is what is expected
expected_v = np.array([0, -14, -20, -22, -14, -18, -20, -20, -20, -20, -18, -14, -22, -20, -14, 0])
np.testing.assert_array_almost_equal(v, expected_v, decimal=2)

print("Value Function:")
print(v)
print("")

print("Reshaped Grid Value Function:")
print(v.reshape(env.shape))
print("")

import gym
import matplotlib
import numpy as np
import sys

from collections import defaultdict
if "../" not in sys.path:
    sys.path.append("../")
from lib.env.blackjack import BlackjackEnv
from lib import plotting

matplotlib.style.use('ggplot')

env = BlackjackEnv()

def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.

    Args:
        Q: A dictionary that maps from state -> action-values
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action. Float between 0 and 1
        nA: Number of actions in the environment

    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA
    """
    def policy_fn(observation):
        pass
        # Implement
    return policy_fn

def mc_control_epsilon_greedy(env, num_episodes, discount_factor=1.0, epsilon=0.1):
    """
    Monte Carlo Control using Epsilon-Greedy policies
    Finds an optimal epsilon-greedy policy

    Args:
        env: OpenAI gym environment
        num_episodes: Number of episodes to sample
        discount_factor: Gamma discount factor
        epsilon: Chance the sample a random action. Float between 0 and 1

    Returns:
        A tuple (Q, policy).
        Q is a dictionary mapping state -> action values
        policy is a function that takes an observation as an argument and returns
        action probabilities
    """

    # Keep track of sum and count of returns for each state
    # to calculate an average. An attay could be used to save all
    # returns (like in the textbook) but that's memory inefficient
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)

    # The final action-value function
    # A nested dictionary that maps state -> (action -> action-value)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # The policy being followed
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    # Implement

    return Q, policy

Q, policy = mc_control_epsilon_greedy(env, num_episodes=500000, epsilon=0.1)

# For plotting: Create value function from action-value function
# by picking the best action at each state
V = defaultdict(float)
for state, actions in Q.items():
    action_value = np.max(actions)
    V[state] = action_value
plotting.plot_value_function(V, title="Optimal Value Function")

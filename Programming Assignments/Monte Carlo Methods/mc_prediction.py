import gym
import matplotlib
import numpy as np
import sys

from collections import defaultdict

if "../" not in sys.path:
    sys.path.append("../")
from lib.envs.blackjack import BlackjackEnv
from lib import plotting

matplotlib.style.use('ggplot')

env = BlackjackEnv()

def mc_prediction(policy, env, num_episodes, discount_factor=1.0):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given policy using sampling

    Args:
        policy: A function that maps an observation to action probabilities
        env: OpenAI gym environment
        num_episodes: NNumber of episodes to sample
        discount_factor: Gamma discount factor

    Returns:
        A dictionary that maps from state -> value
        The state is a tuple and the value is a float
    """

    # Keep track of sum and count of returns for each state
    # to calculate an average. An array could be used to save all
    # returns (like in the textbook) but that's memory inefficient
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)

    # The final value function
    V = defaultdict(float)

    for i_episode in range(1, num_episodes + 1):
        # Print out episode number for debugging
        if i_episode % 1000 == 0:
            print("Episode {}/{}".format(i_episode, num_episodes))
            sys.stdout.flush()

        # Generate an episode
        # An episode is an array of (state, action, reward) tuples
        episode = []
        state = env.reset()
        for t in range(100):
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            episode.append([state, action, reward])
            if done:
                break
            state = next_state

        # Find all states that have been visited in this episode
        # Convert each state to a tuple so that it can be used as a dictionary key
        states_in_episode = set([tuple(x[0]) for x in episode])
        for state in states_in_episode:
            # Find the first occurence of state in the episode
            first_occurence_idx = next(i for i,x in enumerate(episode) if x[0] == state)
            # Sum up all rewards since the first occurence
            G = sum([x[2]*(discount_factor**i) for i,x in enumerate(episode[first_occurence_idx:])])
            # Calculate average return for this state over all sampled episodes
            returns_sum[state] += G
            returns_count[state] += 1
            V[state] = returns_sum[state] / returns_count[state]

    return V

def sample_policy(observation):
    """
    A policy that sticks if the player score is >= 20 and hits otherwise
    """
    score, dealer_score, usable_ace = observation
    return 0 if score >= 20 else 1

V_10k = mc_prediction(sample_policy, env, num_episodes=10000)
plotting.plot_value_function(V_10k, title="10,000 Steps")

V_500k = mc_prediction(sample_policy, env, num_episodes=500000)
plotting.plot_value_function(V_500k, title="500,000 Steps")

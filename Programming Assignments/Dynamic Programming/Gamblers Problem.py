import numpy as np
import matplotlib.pyplot as plt

def value_iteration_for_gamblers(p_h, theta=0.0001, discount_factor=1.0):
    """
    Args:
        p_h: Probability of the coin coming up heads

    Returns:

    """

    rewards = np.zeros(101)
    rewards[100] = 1

    V = np.zeros(101)

    def one_step_lookahead(s, V, rewards):
        """
        Helper function to calculate the value for all actions in a given state

        Args:
            s: The gambler's cash. Integer
            V: The vector that contains values at each state
            rewards: The reward vector

        Returns:
            A vector containing the expected value of each action
            Its length equals the number of actions
        """
        A = np.zeros(101)
         # Minimum bet is one, maximum bet is min(s, 100-s)
        stakes = range(1, min(s, 100 - s) + 1)
        for a in stakes:
            # rewards[s+a], rewards[s-a] are immediate rewards
            # V[s+a], V[s-a] are values of the next state
            # Bellman equation
            A[a] += p_h * (rewards[s+a] + discount_factor * V[s+a]) + \
                    (1 - p_h) * (rewards[s-a] + discount_factor * V[s-a])
        return A

    # update value function
    while True:
        # Stopping condition
        delta = 0
        # Update each state
        for s in range(1,100):
            # Find the best action by one-step lookahead
            A = one_step_lookahead(s, V, rewards)
            best_action_value = np.max(A)
            # Calculate delta across all states seen so fat
            delta = max(delta, np.abs(best_action_value - V[s]))
            # Update the value function
            V[s] = best_action_value
        if delta < theta:
            break

    # Create a deterministic policy using the optimal value function
    policy = np.zeros(100)
    for s in range(100):
        # Find the best action by one-step lookahead
        A = one_step_lookahead(s, V, rewards)
        best_action = np.argmax(A)
        policy[s] = best_action

    return policy, V

policy, v = value_iteration_for_gamblers(0.25)

print("Optimized Policy:")
print(policy)
print("")

print("Optimized Value Function:")
print(v)
print("")

# Plotting Final Policy (action stake) vs State (Cash)

# x axis values
x = range(100)
# corresponding y axis values
y = v[:100]

# plotting the points
plt.plot(x, y)

# naming the axes
plt.xlabel('Cash')
plt.ylabel('Value Estimates')

# Graph title
plt.title('Final Policy (action stake) vs State (Cash)')

# Show plot
plt.show()

# Plotting Cash vs Final Policy

# x axis values
x = range(100)
# corresponding y axis values
y = policy

# plotting the bars
plt.bar(x, y, align='center', alpha=0.5)

# naming the x axis
plt.xlabel('Cash')
# naming the y axis
plt.ylabel('Final policy (stake)')

# giving a title to the graph
plt.title('Cash vs Final Policy')

# function to show the plot
plt.show()

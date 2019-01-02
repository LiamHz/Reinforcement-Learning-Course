import numpy as np
import matplotlib.pyplot as plt

class SlotMachine:
    def __init__(self, k):
        self.k = k
        # Create action values with gaussian distribution with mean 0 and unit variance
        # k action values are create
        self.actions = np.random.normal(size=self.k)

    def take_action(self, k):
        return np.random.normal(loc=self.actions[k])

class Agent:
    def __init__(self, k, e=0.1):
        self.k = k
        self.epsilon = e

        # First value is action value, second value is num_visits
        self.action_values = np.array([[0.0, 0.0]] * k)


    def change_action_value(self, action, reward):
        # Update Q value for action k
        # NewEstimate = OldEstimate + StepSize * [Target - OldEstimatei]
        old_estimate = self.action_values[action, 0]
        previous_action_visits = self.action_values[action, 1]
        step_size = 1 / max(previous_action_visits, 1)  # Don't divide by 0

        self.action_values[action, 0] = old_estimate + step_size * (reward - old_estimate)

        # Increment num_visits for action
        self.action_values[action, 1] += 1


    def take_action(self):
        rand = np.random.random()
        # Exploration action
        if rand < self.epsilon:
            action = np.random.randint(10)
        # Exploitation action
        else:
            # Return index of action (row) with highest action value
            action = np.argmax(self.action_values, axis=0)[0]

        return action


k = 10
iterations = 1000
epochs = 2000

# Create one slot machine and one agent for each epoch
slots = [SlotMachine(k) for i in range(epochs)]
agents = [Agent(k, 0.1) for i in range(epochs)]

rewards = np.zeros((epochs, iterations))
for e in range(epochs):
    print("Epoch #", e, "/", epochs)
    for i in range(iterations):
        agent_action = agents[e].take_action()

        # Get reward from taking e-greedy action
        reward = slots[e].take_action(agent_action)
        # print(reward)
        rewards[e, i] = reward

        # Update agent's Q-value estimate
        agents[e].change_action_value(agent_action, reward)

# Caluculate average reward at iteration i over epochs
# Take mean of column
rewards = np.mean(rewards, axis=0)

# Plot training and validation loss over epochs
plt.plot(rewards, label="Rewards")
# plt.plot(validation_losses, label="Test loss")
plt.legend(frameon=False)

plt.yticks(np.arange(0, 1.51, step=0.5))

# Axis labels
plt.ylabel("Reward")
plt.xlabel("Iteration")
plt.show()

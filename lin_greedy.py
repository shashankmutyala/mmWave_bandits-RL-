import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from mmWave_bandits import mmWaveEnv 
class EpsilonGreedyAgent:
    def __init__(self, environment, initial_epsilon=1.0, decay_rate=0.99, minimum_epsilon=0.1, learning_rate=0.001, max_clip_value=1.0):
        self.environment = environment
        self.epsilon = initial_epsilon
        self.decay_rate = decay_rate
        self.minimum_epsilon = minimum_epsilon
        self.learning_rate = learning_rate
        self.max_clip_value = max_clip_value
        self.action_space_dimensions = (environment.action_space[0].n, environment.action_space[1].n)
        self.feature_count = 3
        self.weight_matrix = np.zeros((self.action_space_dimensions[0], self.action_space_dimensions[1], self.feature_count))
        self.reward_history = []

    def choose_action(self, state):
        """Choosing action using an epsilon-greedy strategy."""
        if np.random.rand() < self.epsilon:
            # Explore: Random action selection
            selected_action = (np.random.randint(self.action_space_dimensions[0]), np.random.randint(self.action_space_dimensions[1]))
        else:
            # Exploit: Select action with the highest predicted reward
            predicted_rewards = self.evaluate(state)
            selected_action = np.unravel_index(np.argmax(predicted_rewards), predicted_rewards.shape)
        return selected_action

    def evaluate(self, state):
        """Estimating rewards for all actions given the current state."""
        total_rewards = np.zeros((self.action_space_dimensions[0], self.action_space_dimensions[1]))
        for car_state in state:
            predicted = np.tensordot(self.weight_matrix, car_state, axes=([2], [0]))
            total_rewards += np.clip(predicted, -1e6, 1e6)
        return total_rewards / len(state)

    def learn(self, state, action, reward):
        """Updating the weights based on the received reward."""
        for car_state in state:
            predicted_reward = np.dot(self.weight_matrix[action[0], action[1]], car_state)
            error = reward - predicted_reward
            clipped_error = np.clip(error, -self.max_clip_value, self.max_clip_value)
            self.weight_matrix[action[0], action[1]] += self.learning_rate * clipped_error * car_state

    def train_agent(self, episodes=1500):
        """Training the agent over a specified number of episodes and visualize the average reward."""
        window_size = 100
        for episode in range(episodes):
            state, _ = self.environment.reset()
            done = False
            total_reward = 0

            while not done:
                action = self.choose_action(state)
                next_state, reward, terminated, truncated, _ = self.environment.step(action)
                done = terminated or truncated
                self.learn(state, action, reward)
                total_reward += reward
                state = next_state
            self.reward_history.append(total_reward)
            # Updating epsilon value
            self.epsilon = max(self.minimum_epsilon, self.epsilon * self.decay_rate)
            if episode % 100 == 0:
                print(f"Episode {episode}/{episodes}, Total Reward: {total_reward:.2f}, Epsilon: {self.epsilon:.4f}")
        self.visualize_average_reward(window_size)

    def visualize_average_reward(self, window_size):
        """Plot the average reward over episodes."""
        average_rewards = np.convolve(self.reward_history, np.ones(window_size) / window_size, mode='valid')
        plt.plot(average_rewards)
        plt.xlabel('Episode')
        plt.ylabel(f'Average Reward (Window Size={window_size})')
        plt.title('Average Reward Over Episodes')
        plt.show()


if __name__ == "__main__":
    environment = mmWaveEnv()
    agent_instance = EpsilonGreedyAgent(environment)
    agent_instance.train_agent(episodes=1500)
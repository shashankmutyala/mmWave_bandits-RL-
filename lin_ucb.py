import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from mmWave_bandits import mmWaveEnv

class LinearUCBPolicyAgent:
    def __init__(self, env, confidence=0.18, learning_rate=0.001):
        """
        Initializing the Linear UCB Agent.
        """
        self.env = env
        self.confidence = confidence  
        self.learning_rate = learning_rate

        # Action space: Number of base stations and beams.
        self.action_dims = (env.action_space[0].n, env.action_space[1].n)

        # Feature dimensions: Number of features used to define the environment's context.
        self.feature_count = 3  
        self.weights = np.zeros((self.action_dims[0], self.action_dims[1], self.feature_count))

        # Covariance matrix and bias vector for each action.
        self.covariance_matrices = np.array([[np.eye(self.feature_count) for _ in range(self.action_dims[1])]
                                             for _ in range(self.action_dims[0])])
        self.bias_vectors = np.zeros((self.action_dims[0], self.action_dims[1], self.feature_count))

        # To track rewards for analysis.
        self.reward_log = []

    def select_best_action(self, context):
        """
        Selecting an action using the Upper Confidence Bound (UCB) strategy.
        Returns:
        - action: the selected action based on UCB values.
        """
        ucb_scores = np.zeros((self.action_dims[0], self.action_dims[1]))

        for car_context in context:
            for station in range(self.action_dims[0]):
                for beam in range(self.action_dims[1]):
                    A_inv = np.linalg.inv(self.covariance_matrices[station, beam])
                    theta_estimate = np.dot(A_inv, self.bias_vectors[station, beam])
                    predicted_reward = np.dot(theta_estimate, car_context)
                    # Upper Confidence Bound calculation
                    confidence_bound = self.confidence * np.sqrt(np.dot(car_context.T, np.dot(A_inv, car_context)))
                    ucb_scores[station, beam] += predicted_reward + confidence_bound

        return np.unravel_index(np.argmax(ucb_scores), ucb_scores.shape)

    def update_model(self, context, action, reward):
        """
        Updating the agent's model based on the observed outcome.
        """
        for car_context in context:
            self.covariance_matrices[action[0], action[1]] += np.outer(car_context, car_context)
            self.bias_vectors[action[0], action[1]] += self.learning_rate * (reward - np.dot(car_context, self.weights[action[0], action[1]])) * car_context

    def train_agent(self, episodes=1500):
        """
        Training the agent by interacting with the environment.
        """
        window = 100
        average_rewards = []
        rolling_avg_reward = 0.0

        for ep in range(episodes):
            context, _ = self.env.reset()
            episode_done = False
            total_reward_in_episode = 0

            while not episode_done:
                chosen_action = self.select_best_action(context)
                new_context, reward, done, truncated, _ = self.env.step(chosen_action)
                episode_done = done or truncated

                # Stabilizing reward if required.
                reward = np.clip(reward, -10, 10)

                self.update_model(context, chosen_action, reward)
                total_reward_in_episode += reward
                context = new_context

            self.reward_log.append(total_reward_in_episode)

            # Calculate rolling average reward.
            if len(self.reward_log) >= window:
                rolling_avg_reward = np.mean(self.reward_log[-window:])
                average_rewards.append(rolling_avg_reward)

            if ep % 100 == 0:
                print(f"Episode {ep}/{episodes}, Reward: {total_reward_in_episode:.2f}, Avg Reward (last {window}): {rolling_avg_reward:.2f}")

        # Plot the rolling average of rewards.
        self.plot_average_rewards(average_rewards)

    def plot_average_rewards(self, avg_rewards):
        """Plot the average reward per episode with a rolling window."""
        plt.figure(figsize=(10, 6))
        plt.plot(avg_rewards)
        plt.xlabel('Episodes (x100)')
        plt.ylabel('Average Reward (last 100 episodes)')
        plt.title('Reward Trend Across Episodes')
        plt.grid()
        plt.show()

if __name__ == "__main__":
    environment = mmWaveEnv()
    agent = LinearUCBPolicyAgent(environment, confidence=0.2, learning_rate=0.001)
    agent.train_agent(episodes=1500)
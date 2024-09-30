import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from mmWave_bandits import mmWaveEnv  

np.random.seed(0)
tf.random.set_seed(0)

class PolicyGradient:
    def __init__(self, state_dim, action_dim, learning_rate=0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model = self.build_model()
        self.optimizer = keras.optimizers.Adam(learning_rate)
        self.states = []
        self.actions = []
        self.rewards = []
#Building the Model
    def build_model(self):
        model = keras.Sequential([
            keras.layers.Input(shape=(self.state_dim,)),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(self.action_dim, activation='softmax')
        ])
        return model

    def choose_action(self, state):
        state = np.reshape(state, [1, -1])
        probabilities = self.model(state, training=False)[0].numpy()
        probabilities = np.maximum(probabilities, 0)

        # Normalizing probabilities to sum to 1
        sum_probabilities = np.sum(probabilities)
        if sum_probabilities > 0:
            probabilities /= sum_probabilities
        else:
            probabilities = np.ones_like(probabilities) / self.action_dim

        action = np.random.choice(self.action_dim, p=probabilities)
        return action

    def store_transition(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def train(self):
        if len(self.states) == 0:
            print("Warning: No transitions to learn from. Skipping training step.")
            return
        states = np.vstack(self.states)
        actions = np.array(self.actions)
        rewards = np.array(self.rewards)
        discounted_rewards = self.discount_rewards(rewards)
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= (np.std(discounted_rewards) + 1e-10)
       
        with tf.GradientTape() as tape:
            logits = self.model(states, training=True)
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=actions)
            loss = tf.reduce_mean(neg_log_prob * discounted_rewards)
        
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        self.states, self.actions, self.rewards = [], [], []

    def discount_rewards(self, rewards, gamma=0.99):
        discounted_r = np.zeros_like(rewards, dtype=np.float32)
        running_add = 0
        for t in reversed(range(len(rewards))):
            running_add = running_add * gamma + rewards[t]
            discounted_r[t] = running_add
        return discounted_r

def preprocess_state(state):
    max_cars = 4
    flat_state = np.zeros(max_cars * 3)
    for i, car in enumerate(state):
        if i < max_cars:
            flat_state[i*3:(i+1)*3] = car
    return flat_state

# Initializing environment
env = mmWaveEnv()
state_dim = 4 * 3  # max_cars * features_per_car
action_dim = env.action_space[0].n * env.action_space[1].n
# Initializing Policy Gradient agent
agent = PolicyGradient(state_dim, action_dim)
# Training loop
n_episodes = 1000
window_size = 100
rewards = []
for episode in range(n_episodes):
    state, _ = env.reset()
    state = preprocess_state(state)
    episode_reward = 0
    
    for t in range(env.Horizon):
        action_index = agent.choose_action(state)
        
        # Converting action index to tuple
        base_station = action_index // env.Nbeams
        beam = action_index % env.Nbeams
        action = (base_station, beam)
        
        next_state, reward, terminated, truncated, _ = env.step(action)
        next_state = preprocess_state(next_state)
        
        agent.store_transition(state, action_index, reward)
        
        episode_reward += reward
        if terminated or truncated:
            break
        
        state = next_state
    
    agent.train()
    rewards.append(episode_reward)
    
    if (episode + 1) % 100 == 0:
        print(f"Episode {episode + 1}/{n_episodes}, Avg Reward: {np.mean(rewards[-100:]):.2f}")

plt.figure(figsize=(10, 5))
plt.plot(np.convolve(rewards, np.ones(window_size)/window_size, mode='valid'))
plt.title("Policy Gradient: Average Reward over Episodes")
plt.xlabel("Episode")
plt.ylabel("Average Reward (Window Size: 100)")
plt.show()

env.close()
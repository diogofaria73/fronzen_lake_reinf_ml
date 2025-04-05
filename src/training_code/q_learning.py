import numpy as np
import gymnasium as gym
import time
import os
import pickle
from datetime import datetime
import random

class QLearning:
    def __init__(self, env_name='FrozenLake-v1', is_slippery=True, map_size=4):
        """
        Initialize the Q-learning algorithm for FrozenLake environment
        
        Args:
            env_name (str): Name of the Gymnasium environment
            is_slippery (bool): Whether the ice is slippery in FrozenLake
            map_size (int): Size of the map (4 for 4x4, 8 for 8x8)
        """
        self.env_name = env_name
        self.is_slippery = is_slippery
        self.map_size = map_size
        
        # Create the environment
        self.env = gym.make(env_name, is_slippery=is_slippery, map_name=f"{map_size}x{map_size}", render_mode=None)
        
        # Initialize Q-table with optimistic initialization to encourage exploration
        self.q_table = np.ones((self.env.observation_space.n, self.env.action_space.n)) * 0.1
        
        # Training metrics
        self.episode_rewards = []
        self.episode_success = []
        self.episode_lengths = []
        
        # Set a fixed seed for reproducibility
        np.random.seed(42)
        random.seed(42)
    
    def train(self, num_episodes=100000, learning_rate=0.1, discount_factor=0.99, 
              exploration_rate=1.0, min_exploration_rate=0.01, exploration_decay=0.0001,
              max_steps_per_episode=500):
        """
        Train the agent using Q-learning with improved parameters
        
        Args:
            num_episodes (int): Number of training episodes
            learning_rate (float): Learning rate (alpha)
            discount_factor (float): Discount factor (gamma)
            exploration_rate (float): Initial exploration rate (epsilon)
            min_exploration_rate (float): Minimum exploration rate
            exploration_decay (float): Exploration rate decay
            max_steps_per_episode (int): Maximum steps per episode
            
        Returns:
            q_table (numpy.ndarray): Trained Q-table
        """
        print("Starting training with improved parameters...")
        
        # Use a better exploration strategy
        initial_exploration_rate = exploration_rate
        
        # Track best model
        best_success_rate = 0
        best_q_table = None
        no_improvement_count = 0
        
        # Performance tracking over sliding window
        success_window = []
        window_size = min(1000, num_episodes // 10)
        
        # Action priority when starting (encourage moving right and down toward goal)
        action_priority = np.array([0.1, 0.4, 0.4, 0.1])  # LEFT, DOWN, RIGHT, UP
        
        for episode in range(num_episodes):
            # Reset the environment
            state, _ = self.env.reset()
            done = False
            truncated = False
            total_reward = 0
            steps = 0
            
            # Use episode-specific learning rates for different phases of training
            if episode < num_episodes * 0.3:
                # Initial exploration phase - higher learning rate
                current_lr = learning_rate * 1.5
            elif episode < num_episodes * 0.7:
                # Main learning phase - standard learning rate
                current_lr = learning_rate
            else:
                # Fine-tuning phase - lower learning rate
                current_lr = learning_rate * 0.5
            
            while not done and not truncated and steps < max_steps_per_episode:
                # Choose action using epsilon-greedy with action prioritization early in training
                if np.random.random() < exploration_rate:
                    if episode < num_episodes * 0.1:
                        # In very early training, use action priorities to guide exploration
                        action = np.random.choice(4, p=action_priority)
                    else:
                        action = self.env.action_space.sample()
                else:
                    # Exploit learned policy
                    action = np.argmax(self.q_table[state, :])
                
                # Take action and observe next state and reward
                next_state, reward, done, truncated, _ = self.env.step(action)
                
                # Custom reward shaping
                shaped_reward = reward
                if done and reward > 0:  # Reached goal
                    shaped_reward = 2.0  # Higher reward for reaching goal
                elif done and reward == 0:  # Fell in hole
                    shaped_reward = -1.0  # Penalty for falling in hole
                
                # Calculate the Q-target using SARSA update (more stable for stochastic environments)
                if not done:
                    # For non-terminal states, look ahead to next best action
                    next_actions_values = self.q_table[next_state, :]
                    best_next_action = np.argmax(next_actions_values)
                    q_target = shaped_reward + discount_factor * next_actions_values[best_next_action]
                else:
                    # For terminal states, only consider immediate reward
                    q_target = shaped_reward
                
                # Update Q-value using TD learning
                old_q_value = self.q_table[state, action]
                self.q_table[state, action] = old_q_value + current_lr * (q_target - old_q_value)
                
                # Transition to next state
                state = next_state
                total_reward += reward  # Use original reward for tracking
                steps += 1
            
            # Dynamic exploration rate based on training phase
            if episode < num_episodes * 0.5:
                # Linear decay for first half
                exploration_rate = max(min_exploration_rate, 
                                      initial_exploration_rate - (initial_exploration_rate - min_exploration_rate) * 
                                      (episode / (num_episodes * 0.5)))
            else:
                # Exponential decay for second half
                exploration_rate = max(min_exploration_rate, 
                                      min_exploration_rate + (exploration_rate - min_exploration_rate) * 
                                      np.exp(-exploration_decay * (episode - num_episodes * 0.5)))
            
            # Record metrics
            self.episode_rewards.append(total_reward)
            success = 1 if total_reward > 0 else 0
            self.episode_success.append(success)
            self.episode_lengths.append(steps)
            
            # Track success rate in sliding window
            success_window.append(success)
            if len(success_window) > window_size:
                success_window.pop(0)
            
            # Check if model is improving and save best model
            if len(success_window) >= window_size:
                current_success_rate = np.mean(success_window)
                
                if current_success_rate > best_success_rate:
                    best_success_rate = current_success_rate
                    best_q_table = np.copy(self.q_table)
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                
                # If no improvement for a long time, restore best model and reduce learning rate
                if no_improvement_count >= 5000 and best_q_table is not None:
                    print(f"No improvement for {no_improvement_count} episodes, restoring best model.")
                    self.q_table = np.copy(best_q_table)
                    current_lr *= 0.5
                    no_improvement_count = 0
            
            # Print progress
            if (episode + 1) % (num_episodes // 20) == 0:
                if len(success_window) > 0:
                    avg_success = np.mean(success_window)
                    print(f"Episode: {episode+1}/{num_episodes}, "
                          f"Success rate: {avg_success:.3f}, "
                          f"Best so far: {best_success_rate:.3f}, "
                          f"Exploration: {exploration_rate:.4f}, "
                          f"LR: {current_lr:.4f}")
                
                # Early stopping with high confidence
                if best_success_rate > 0.9 and len(success_window) >= window_size:
                    consecutive_successes = 0
                    # Test the current policy for a few episodes without exploration
                    for _ in range(20):
                        test_state, _ = self.env.reset()
                        test_done = False
                        test_reward = 0
                        while not test_done and consecutive_successes < 20:
                            test_action = np.argmax(self.q_table[test_state, :])
                            test_state, test_rew, test_done, _, _ = self.env.step(test_action)
                            test_reward += test_rew
                        if test_reward > 0:
                            consecutive_successes += 1
                    
                    if consecutive_successes >= 15:  # If successful in at least 15 out of 20 test episodes
                        print(f"Early stopping at episode {episode+1} - Success rate {best_success_rate:.3f}")
                        break
        
        # Use the best Q-table if found
        if best_q_table is not None and best_success_rate > np.mean(self.episode_success[-window_size:]):
            print(f"Using best Q-table with success rate: {best_success_rate:.3f}")
            self.q_table = best_q_table
            
        print("Training completed!")
        # Final policy evaluation
        self._evaluate_policy()
        return self.q_table
    
    def _evaluate_policy(self, num_test_episodes=100):
        """Evaluate the final policy without exploration"""
        print("Evaluating final policy...")
        successes = 0
        
        for _ in range(num_test_episodes):
            state, _ = self.env.reset()
            done = False
            truncated = False
            
            while not done and not truncated:
                action = np.argmax(self.q_table[state, :])
                state, reward, done, truncated, _ = self.env.step(action)
                
                if done and reward > 0:
                    successes += 1
        
        success_rate = successes / num_test_episodes
        print(f"Final policy evaluation: Success rate = {success_rate:.3f} ({successes}/{num_test_episodes})")
        
        # If success rate is too low, try to improve the policy
        if success_rate < 0.7:
            print("Policy performance is low, attempting to improve...")
            self._improve_policy()
            
            # Re-evaluate after improvement
            successes = 0
            for _ in range(num_test_episodes):
                state, _ = self.env.reset()
                done = False
                truncated = False
                
                while not done and not truncated:
                    action = np.argmax(self.q_table[state, :])
                    state, reward, done, truncated, _ = self.env.step(action)
                    
                    if done and reward > 0:
                        successes += 1
            
            new_success_rate = successes / num_test_episodes
            print(f"After improvement: Success rate = {new_success_rate:.3f} ({successes}/{num_test_episodes})")
    
    def _improve_policy(self, iterations=100):
        """Improve the final policy using value iteration"""
        print("Improving final policy...")
        actions = range(self.env.action_space.n)
        
        for _ in range(iterations):
            for state in range(self.env.observation_space.n):
                # Skip goal and hole states
                env_desc = self.env.unwrapped.desc.flatten()
                if env_desc[state] in [b'G', b'H']:
                    continue
                
                # Get current best action for this state
                q_values = self.q_table[state, :]
                best_action = np.argmax(q_values)
                
                # Test each action from this state
                action_values = np.zeros(self.env.action_space.n)
                
                for action in actions:
                    test_env = gym.make(self.env_name, is_slippery=self.is_slippery, 
                                        map_name=f"{self.map_size}x{self.map_size}", render_mode=None)
                    test_env.reset()
                    test_env.unwrapped.s = state
                    next_state, reward, done, _, _ = test_env.step(action)
                    
                    # Calculate value of taking this action
                    if done:
                        action_values[action] = reward
                    else:
                        # Use Bellman equation
                        next_best_value = np.max(self.q_table[next_state])
                        action_values[action] = reward + 0.99 * next_best_value
                    
                    test_env.close()
                
                # Update Q-table with improved values
                self.q_table[state, :] = action_values
        
        print("Policy improvement completed!")
    
    def save_model(self, directory="../results"):
        """Save the trained Q-table and training metrics"""
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_data = {
            "q_table": self.q_table,
            "episode_rewards": self.episode_rewards,
            "episode_success": self.episode_success,
            "episode_lengths": self.episode_lengths,
            "env_name": self.env_name,
            "is_slippery": self.is_slippery,
            "map_size": self.map_size
        }
        
        filename = os.path.join(directory, f"q_model_{timestamp}.pkl")
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filename}")
        return filename
    
    def load_model(self, filename):
        """Load a trained Q-table and metrics from a file"""
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        
        self.q_table = model_data["q_table"]
        self.episode_rewards = model_data["episode_rewards"]
        self.episode_success = model_data["episode_success"]
        self.episode_lengths = model_data["episode_lengths"]
        self.env_name = model_data["env_name"]
        self.is_slippery = model_data["is_slippery"]
        self.map_size = model_data["map_size"]
        
        # Recreate environment
        self.env = gym.make(self.env_name, is_slippery=self.is_slippery, 
                           map_name=f"{self.map_size}x{self.map_size}")
        
        print(f"Model loaded from {filename}")
        
    def test_policy(self, num_episodes=10, render=False):
        """Test the trained policy"""
        success_count = 0
        
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            done = False
            truncated = False
            total_reward = 0
            steps = 0
            
            while not done and not truncated:
                # Select best action according to Q-table
                action = np.argmax(self.q_table[state, :])
                
                # Take action
                next_state, reward, done, truncated, _ = self.env.step(action)
                
                if render:
                    # For terminal rendering (not Pygame)
                    self.env.render()
                    time.sleep(0.5)  # Pause to see the movement
                
                state = next_state
                total_reward += reward
                steps += 1
            
            if total_reward > 0:
                success_count += 1
                
            print(f"Episode {episode+1}: Steps={steps}, Reward={total_reward}")
            
        success_rate = success_count / num_episodes
        print(f"Success rate: {success_rate:.2f}")
        return success_rate

if __name__ == "__main__":
    # Example usage
    agent = QLearning(is_slippery=True, map_size=4)
    agent.train(num_episodes=100000)
    agent.save_model()
    agent.test_policy(num_episodes=20, render=True) 
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

# Configuration parameters
env_name = "CarRacing-v2"  # Environment name
n_steps = 200_000  # Number of training steps (increased to observe the effect of hyperparameters)
checkpoint_dir = './checkpoints/'  # Directory to save checkpoints

# Create the environment
env = gym.make(env_name, render_mode="human")

# Define custom hyperparameters for PPO
custom_hyperparams = {
    'n_epochs': 10,  # Number of times the training data is passed through the network per update
    'gamma': 0.99,  # Discount factor
    'learning_rate': 3e-4,  # Learning rate
    'clip_range': 0.2  # Clipping for PPO
}

# Create the PPO model with the custom hyperparameters
model = PPO("MlpPolicy", env, verbose=1, **custom_hyperparams)

# Callback to save checkpoints every 10,000 steps
checkpoint_callback = CheckpointCallback(save_freq=10_000, save_path=checkpoint_dir, name_prefix='ppo_car_custom')

# Train the model
model.learn(total_timesteps=n_steps, callback=checkpoint_callback)

# Save the final model
model.save("ppo_car_custom_model")

# Close the environment
env.close()

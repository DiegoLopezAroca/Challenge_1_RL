from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym
import time

# Carga el entorno en modo visual
env = gym.make("CarRacing-v3", continuous=True, render_mode="human")

# Carga tu mejor modelo guardado
model = PPO.load("checkpoints/best_model.zip")

obs, _ = env.reset()
done = False
total_reward = 0

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward
    done = terminated or truncated
    env.render()
    time.sleep(0.02)

print("Recompensa total:", total_reward)
env.close()

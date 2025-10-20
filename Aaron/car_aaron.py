import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

# -----------------------------
# WRAPPERS
# -----------------------------

class SpeedWrapper(gym.ActionWrapper):
    """
    Aumenta ligeramente la aceleración base del coche.
    """
    def __init__(self, env, speed_factor=1.2, min_acc=0.1):
        super().__init__(env)
        self.speed_factor = speed_factor
        self.min_acc = min_acc

    def action(self, action):
        action = np.array(action, dtype=np.float32)
        action[1] = np.clip(max(action[1], self.min_acc) * self.speed_factor, 0.0, 1.0)
        return action


class TerminateOffTrackWrapper(gym.Wrapper):
    """
    Termina el episodio si el coche se sale de la pista.
    """
    def __init__(self, env, min_reward=-0.1):
        super().__init__(env)
        self.min_reward = min_reward

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Si el reward es muy negativo → fuera de pista → terminar episodio
        if reward < self.min_reward:
            terminated = True
            reward -= 5.0  # penalización fuerte
        return obs, reward, terminated, truncated, info


# -----------------------------
# CONFIGURACIÓN DEL ENTORNO
# -----------------------------
env_name = "CarRacing-v3"
n_steps = 200_000          # entrenamiento largo para maximizar reward
checkpoint_dir = './checkpoints/'

env = gym.make(env_name,render_mode = "rgb_array")
env = SpeedWrapper(env, speed_factor=1.3)
env = TerminateOffTrackWrapper(env)

# -----------------------------
# HYPERPARÁMETROS PPO
# -----------------------------
custom_hyperparams = {
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 20,
    "gamma": 0.999,
    "gae_lambda": 0.95,
    "clip_range": 0.1,
    "learning_rate": 2.5e-4,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
}

# -----------------------------
# CREAR MODELO
# -----------------------------
model = PPO("MlpPolicy", env, verbose=1, **custom_hyperparams)

# -----------------------------
# CHECKPOINTS
# -----------------------------
checkpoint_callback = CheckpointCallback(
    save_freq=50_000,
    save_path=checkpoint_dir,
    name_prefix='ppo_car_maxreward'
)

# -----------------------------
# ENTRENAMIENTO
# -----------------------------
model.learn(total_timesteps=n_steps, callback=checkpoint_callback)

# Guardar modelo final
model.save("ppo_car_maxreward_model")
env.close()

import gymnasium as gym
import numpy as np
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

# -----------------------------
# WRAPPERS
# -----------------------------
class SpeedWrapper(gym.ActionWrapper):
    """Aumenta ligeramente la aceleración base del coche."""
    def __init__(self, env, speed_factor=1.3, min_acc=0.1):
        super().__init__(env)
        self.speed_factor = speed_factor
        self.min_acc = min_acc

    def action(self, action):
        action = np.array(action, dtype=np.float32)
        action[1] = np.clip(max(action[1], self.min_acc) * self.speed_factor, 0.0, 1.0)
        return action


class TerminateOffTrackWrapper(gym.Wrapper):
    """Termina el episodio si el coche se sale de la pista."""
    def __init__(self, env, min_reward=-0.1):
        super().__init__(env)
        self.min_reward = min_reward

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if reward < self.min_reward:
            terminated = True
            reward -= 5.0
        return obs, reward, terminated, truncated, info

# -----------------------------
# CONFIGURACIÓN
# -----------------------------
env_name = "CarRacing-v3"
checkpoint_dir = './checkpoints/'
total_training_steps = 200_000  # pasos totales deseados

# -----------------------------
# CREAR ENTORNO
# -----------------------------
env = gym.make(env_name, render_mode="rgb_array")
env = SpeedWrapper(env, speed_factor=1.3)
env = TerminateOffTrackWrapper(env)

# -----------------------------
# CALLBACK DE CHECKPOINT
# -----------------------------
checkpoint_callback = CheckpointCallback(
    save_freq=20_000,
    save_path=checkpoint_dir,
    name_prefix='ppo_car_maxreward'
)

# -----------------------------
# BUSCAR ÚLTIMO CHECKPOINT
# -----------------------------
def get_last_checkpoint(path):
    files = [f for f in os.listdir(path) if f.endswith('.zip')]
    if not files:
        return None
    files.sort(key=lambda x: int(x.split('_')[-2]))  # asume formato: ppo_car_maxreward_50000_steps.zip
    return os.path.join(path, files[-1])

last_checkpoint = get_last_checkpoint(checkpoint_dir)

if last_checkpoint:
    print(f"Cargando checkpoint: {last_checkpoint}")
    model = PPO.load(last_checkpoint, env=env)
    # Calcular pasos restantes para llegar a total_training_steps
    # Extraer pasos del nombre del checkpoint
    steps_done = int(os.path.basename(last_checkpoint).split('_')[-2])
    steps_remaining = max(total_training_steps - steps_done, 0)
else:
    print("No se encontró checkpoint. Entrenamiento desde cero.")
    # Hiperparámetros PPO
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
    model = PPO("MlpPolicy", env, verbose=1, **custom_hyperparams)
    steps_remaining = total_training_steps

# -----------------------------
# ENTRENAMIENTO
# -----------------------------
if steps_remaining > 0:
    print(f"Entrenando por {steps_remaining} pasos restantes...")
    model.learn(total_timesteps=steps_remaining, callback=checkpoint_callback)
    model.save(os.path.join(checkpoint_dir, "ppo_car_maxreward_model"))

env.close()
print("Entrenamiento completado.")

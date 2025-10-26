import gymnasium as gym
import numpy as np
import torch
import os
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback

from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

# -----------------------------
# Wrappers
# -----------------------------
class SpeedWrapper(gym.ActionWrapper):
    """Aumenta la aceleración del coche."""
    def __init__(self, env, speed_factor=1.15):
        super().__init__(env)
        self.speed_factor = speed_factor

    def action(self, action):
        action = np.array(action, dtype=np.float32)
        action[1] = np.clip(action[1] * self.speed_factor, 0.0, 1.0)
        return action

class TerminateOffTrackWrapper(gym.Wrapper):
    """Termina episodio si hay demasiados píxeles verdes (hierba)."""
    def __init__(self, env, max_green_pixels=50):
        super().__init__(env)
        self.max_green_pixels = max_green_pixels
        self.step_count = 0

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        green_pixels = np.sum((obs[:, :, 1] > 200) & (obs[:, :, 0] < 100) & (obs[:, :, 2] < 100))
        self.step_count += 1

        if self.step_count % 1000 == 0:
            print(f"[INFO] Pasos simulados: {self.step_count}")

        if green_pixels > self.max_green_pixels:
            done = True
            reward -= 5.0

        return obs, reward, done, truncated, info

# -----------------------------
# Configuración
# -----------------------------
env_name = "CarRacing-v3"
checkpoint_dir = './checkpoints_sac/'
os.makedirs(checkpoint_dir, exist_ok=True)
total_timesteps = 200_000

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🧠 Usando dispositivo: {device}")

# -----------------------------
# Crear entorno vectorizado
# -----------------------------
def make_env():
    env = gym.make(env_name)
    env = SpeedWrapper(env)
    env = TerminateOffTrackWrapper(env)
    return env

env = DummyVecEnv([make_env])
env = VecMonitor(env)

# -----------------------------
# Callback de checkpoints
# -----------------------------
checkpoint_callback = CheckpointCallback(
    save_freq=2000,
    save_path=os.path.abspath(checkpoint_dir),
    name_prefix='sac_car'
)

# -----------------------------
# Función para buscar último checkpoint
# -----------------------------
def get_last_checkpoint(path):
    files = [f for f in os.listdir(path) if f.endswith('.zip')]
    if not files:
        return None
    files.sort(key=lambda x: int(x.split('_')[-2]))  # Asume formato: sac_car_5000_steps.zip
    return os.path.join(path, files[-1])

last_checkpoint = get_last_checkpoint(checkpoint_dir)

# -----------------------------
# Crear modelo SAC
# -----------------------------
if last_checkpoint:
    print(f"Cargando checkpoint: {last_checkpoint}")
    model = SAC.load(last_checkpoint, env=env, device=device)
    # Calcular pasos restantes
    steps_done = int(os.path.basename(last_checkpoint).split('_')[-2])
    steps_remaining = max(total_timesteps - steps_done, 0)
else:
    print("No se encontró checkpoint. Entrenando desde cero.")
    model = SAC(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        learning_rate=3e-4,
        gamma=0.99,
        buffer_size=10000,
        batch_size=256,
        tau=0.005,
        train_freq=(1, "step"),
        device=device
    )
    steps_remaining = total_timesteps

# -----------------------------
# Entrenamiento
# -----------------------------
if steps_remaining > 0:
    print(f" Entrenando agente SAC por {steps_remaining:,} pasos restantes...")
    model.learn(total_timesteps=steps_remaining, callback=checkpoint_callback)

# -----------------------------
# Guardar modelo final
# -----------------------------
model.save("sac_car_final_model")
print(" Entrenamiento completado y modelo guardado.")

env.close()

import gymnasium as gym
import numpy as np
import torch
import os
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback

# -----------------------------
# Wrappers
# -----------------------------
class SpeedWrapper(gym.ActionWrapper):
    """Aumenta la aceleración base del coche."""
    def __init__(self, env, speed_factor=1.15):
        super().__init__(env)
        self.speed_factor = speed_factor

    def action(self, action):
        action = np.array(action, dtype=np.float32)
        action[1] = np.clip(action[1] * self.speed_factor, 0.0, 1.0)
        return action


class TerminateOffTrackWrapper(gym.Wrapper):
    """Termina el episodio si el coche se sale de la pista (verde)."""
    def __init__(self, env, max_green_pixels=50):
        super().__init__(env)
        self.max_green_pixels = max_green_pixels
        self.step_count = 0

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        green_pixels = np.sum((obs[:, :, 1] > 200) & (obs[:, :, 0] < 100) & (obs[:, :, 2] < 100))
        self.step_count += 1

        # Mostrar progreso cada 1000 pasos
        if self.step_count % 1000 == 0:
            print(f"[INFO] Pasos simulados: {self.step_count}")

        # Si detecta mucha hierba → termina el episodio
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

# Si tienes GPU, úsala
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🧠 Usando dispositivo: {device}")

# -----------------------------
# Crear entorno (sin renderizado para más velocidad)
# -----------------------------
env = gym.make(env_name)  # ⚠️ Sin render_mode="rgb_array" → más rápido
env = SpeedWrapper(env, speed_factor=1.15)
env = TerminateOffTrackWrapper(env)

# -----------------------------
# Callback de checkpoints
# -----------------------------
checkpoint_callback = CheckpointCallback(
    save_freq=5_000,  # guarda cada 5k pasos
    save_path=checkpoint_dir,
    name_prefix='sac_car_fast'
)

# -----------------------------
# Crear modelo SAC optimizado
# -----------------------------
model = SAC(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    learning_rate=3e-4,
    gamma=0.99,
    buffer_size=10_000,   # ⚡ mucho más rápido para arrancar
    batch_size=256,
    tau=0.005,
    train_freq=(1, "step"),
    device=device
)

# -----------------------------
# Entrenamiento
# -----------------------------
print(f"🚗 Entrenando agente SAC por {total_timesteps:,} pasos...")
model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)

# -----------------------------
# Guardar modelo final
# -----------------------------
model.save("sac_car_fast_model")
print("✅ Entrenamiento completado y modelo guardado.")

env.close()

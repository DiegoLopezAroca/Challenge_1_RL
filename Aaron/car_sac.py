import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback

# -----------------------------
# Wrappers
# -----------------------------
class SpeedWrapper(gym.ActionWrapper):
    """Aumenta la aceleración por defecto."""
    def __init__(self, env, speed_factor=1.15):
        super().__init__(env)
        self.speed_factor = speed_factor

    def action(self, action):
        action = np.array(action, dtype=np.float32)
        # action[1] = acelerador
        action[1] = np.clip(action[1] * self.speed_factor, 0.0, 1.0)
        return action

class TerminateOffTrackWrapper(gym.Wrapper):
    """Termina episodio si toca hierba (verde)."""
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        green_pixels = np.sum((obs[:,:,1] > 200) & (obs[:,:,0] < 100) & (obs[:,:,2] < 100))
        if green_pixels > 50:
            done = True
            reward -= 5.0  # penalización por salirse
        return obs, reward, done, truncated, info

# -----------------------------
# Config
# -----------------------------
env_name = "CarRacing-v3"
checkpoint_dir = './checkpoints_sac/'
total_timesteps = 200_000

# -----------------------------
# Crear entorno con wrappers
# -----------------------------
env = gym.make(env_name, render_mode="rgb_array")
env = SpeedWrapper(env, speed_factor=1.15)
env = TerminateOffTrackWrapper(env)

# -----------------------------
# Callback para checkpoints
# -----------------------------
checkpoint_callback = CheckpointCallback(
    save_freq=10_000,
    save_path=checkpoint_dir,
    name_prefix='sac_car_maxreward'
)

# -----------------------------
# Crear modelo SAC
# -----------------------------
model = SAC(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    learning_rate=3e-4,
    gamma=0.99,
    buffer_size=100_000,
    batch_size=256,
    tau=0.005,
    train_freq=(1, "step")
)

# -----------------------------
# Entrenar
# -----------------------------
model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)

# Guardar modelo final
model.save("sac_car_maxreward_model")

# Cerrar entorno
env.close()

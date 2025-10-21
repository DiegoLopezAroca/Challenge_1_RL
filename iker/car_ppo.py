import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
import os

# Configuración
env_name = "CarRacing-v3"
n_steps = 2_000_000
checkpoint_dir = './checkpoints/'

# Crear entorno de evaluación
eval_env = make_vec_env(
    lambda: gym.make(env_name, continuous=True, render_mode='rgb_array', lap_complete_percent=0.95, domain_randomize=False),
    n_envs=1
)

# Crear carpeta de checkpoints
os.makedirs(checkpoint_dir, exist_ok=True)

# Entorno vectorizado para entrenamiento
env = make_vec_env(
    lambda: gym.make(env_name, continuous=True, render_mode=None, lap_complete_percent=0.95, domain_randomize=False),
    n_envs=4
)

# Callback para guardar checkpoints
checkpoint_callback = CheckpointCallback(
    save_freq=50_000,
    save_path=checkpoint_dir,
    name_prefix='ppo_car_custom'
)

# Callback para evaluación periódica
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=checkpoint_dir,
    log_path=checkpoint_dir,
    eval_freq=100_000,
    n_eval_episodes=5,
    render=False,
)

# Cargar modelo anterior
model = PPO.load("checkpoints/best_model.zip", env=env)

# Continuar entrenamiento
model.learn(total_timesteps=n_steps, callback=[checkpoint_callback, eval_callback])

# Guardar modelo final
model.save("ppo_car_custom_model_continued")

# Cerrar entornos
env.close()
eval_env.close()

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
import os

# Configuración
env_name = "CarRacing-v3"
n_steps = 200_000
checkpoint_dir = './checkpoints/'

# Crear entorno de evaluación
eval_env = make_vec_env(
    lambda: gym.make(env_name, continuous=True, render_mode='rgb_array', lap_complete_percent=0.95, domain_randomize=False),
    n_envs=1
)

# Crear carpeta de checkpoints 
os.makedirs(checkpoint_dir, exist_ok=True)

# Entorno vectorizado para entrenamiento
# No renderizar para que sea mas rapido y vectorizar con 4 entornos para recolectar mas experiencia en menos tiempo
env = make_vec_env(
    lambda: gym.make(env_name, continuous=True, render_mode=None, lap_complete_percent=0.95, domain_randomize=False),
    n_envs=4
)

# Callback para guardar checkpoints cada 10,000 pasos
checkpoint_callback = CheckpointCallback(
    save_freq=10_000,
    save_path=checkpoint_dir,
    name_prefix='ppo_car_custom'
)

# Callback para evaluación periódica cada 20,000 pasos
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=checkpoint_dir,
    log_path=checkpoint_dir,
    eval_freq=20_000,
    n_eval_episodes=5,
    render=False,   
)

# Crear modelo PPO
model = PPO(
    "CnnPolicy",            # Pollitica Cnn
    env,
    verbose=1,
    learning_rate=2.5e-4,
    n_steps=4096,
    batch_size=64,
    n_epochs=10,
    gamma=0.98,
    clip_range=0.2,
)

# Entrenar el modelo 
model.learn(total_timesteps=n_steps, callback=[checkpoint_callback, eval_callback])

# Guardar el modelo final
model.save("ppo_car_custom_model")

# Cerrar entornos
env.close()
eval_env.close()

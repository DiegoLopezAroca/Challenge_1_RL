import numpy as np
import pygame
from gymnasium.envs.box2d.car_racing import CarRacing
from PIL import Image
import os

# ----------------------------
# CONFIGURACIÓN
# ----------------------------
num_episodios = 1    # Número de episodios a recolectar
fps = 50               # Frames por segundo
save_npy = "expert_data.npy"
save_frames = True     # True si quieres guardar frames como PNG
frames_dir = "frames"  # Carpeta raíz para guardar frames y txt por episodio

# ----------------------------
# INICIALIZACIÓN
# ----------------------------
env = CarRacing(render_mode="human")
clock = pygame.time.Clock()
expert_data = []  # Lista de episodios, cada episodio es lista de (obs, action)

# Crear carpeta raíz de frames
if save_frames:
    os.makedirs(frames_dir, exist_ok=True)

print("🚗 Usa flechas para conducir. Q o ESC para salir.")

# ----------------------------
# FUNCIONES DE TECLADO A ACCIÓN
# ----------------------------
def get_keyboard_action():
    keys = pygame.key.get_pressed()
    steering = 0.0
    if keys[pygame.K_LEFT]:
        steering = -1.0
    elif keys[pygame.K_RIGHT]:
        steering = 1.0
    gas = 1.0 if keys[pygame.K_UP] else 0.0
    brake = 1.0 if keys[pygame.K_DOWN] else 0.0
    return np.array([steering, gas, brake], dtype=np.float32)

# ----------------------------
# RECOLECCIÓN DE DATOS
# ----------------------------
for ep in range(num_episodios):
    obs, _ = env.reset()
    done = False
    step = 0
    frames_episode = []  # Lista temporal para almacenar frames y acciones de este episodio

    # Crear subcarpeta para el episodio
    ep_dir = os.path.join(frames_dir, f"ep{ep}")
    os.makedirs(ep_dir, exist_ok=True)
    
    # Archivo txt para este episodio
    txt_path = os.path.join(ep_dir, f"acciones_ep{ep}.txt")
    with open(txt_path, "w") as f:
        f.write("step steering gas brake\n")

    print(f"🚗 Episodio {ep+1}/{num_episodios}")

    while not done:
        # Capturar eventos de pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_q, pygame.K_ESCAPE]:
                    done = True

        # Construir acción a partir del teclado
        action = get_keyboard_action()

        # Ejecutar paso en el entorno
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Guardar frame y acción en memoria
        frames_episode.append((obs.copy(), action.copy()))

        # Guardar acción en txt del episodio
        with open(txt_path, "a") as f:
            f.write(f"{step} {action[0]:.2f} {action[1]:.2f} {action[2]:.2f}\n")

        obs = next_obs
        step += 1
        clock.tick(fps)

    # Al final del episodio, añadir episodio completo a dataset global
    expert_data.append(frames_episode)

    # Guardar frames como PNG si se desea
    if save_frames:
        for s, (frame_obs, _) in enumerate(frames_episode):
            Image.fromarray(frame_obs).save(f"{ep_dir}/step{s:05d}.png")

    # Guardar dataset parcial en .npy con dtype=object y allow_pickle
    np.save(save_npy, np.array(expert_data, dtype=object), allow_pickle=True)
    print(f"Episodio {ep+1} guardado. Total episodios: {len(expert_data)}")

# ----------------------------
# CIERRE
# ----------------------------
env.close()
print(f"✅ Guardado {len(expert_data)} episodios en {save_npy}")
if save_frames:
    print(f"✅ Frames y txt por episodio guardados en '{frames_dir}'")

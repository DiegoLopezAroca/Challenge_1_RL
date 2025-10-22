import numpy as np
import pygame
from gymnasium.envs.box2d.car_racing import CarRacing
from PIL import Image
import os

# ----------------------------
# CONFIGURACIÓN
# ----------------------------
num_episodios = 1      # Número de episodios nuevos a grabar
fps = 50               
save_npy = "expert_data.npy"
save_frames = True     
frames_dir = "frames"

# ----------------------------
# INICIALIZACIÓN
# ----------------------------
env = CarRacing(render_mode="human")
clock = pygame.time.Clock()

#cargar episodios previos si existen
if os.path.exists(save_npy):
    expert_data = list(np.load(save_npy, allow_pickle=True))
    start_ep = len(expert_data)
    print(f"Cargados {start_ep} episodios previos. Continuando desde ep{start_ep}.")
else:
    expert_data = []
    start_ep = 0
    print("No existe dataset previo. Comenzando desde cero.")

# Crear carpeta raíz de frames si no existe
if save_frames:
    os.makedirs(frames_dir, exist_ok=True)

print("Usa flechas para conducir. Q o ESC para salir.")

# ----------------------------
# FUNCIONES DE TECLADO → ACCIÓN
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
for ep in range(start_ep, start_ep + num_episodios):
    obs, _ = env.reset()
    done = False
    step = 0
    frames_episode = []

    # Crear carpeta para este nuevo episodio
    ep_dir = os.path.join(frames_dir, f"ep{ep}")
    os.makedirs(ep_dir, exist_ok=True)

    txt_path = os.path.join(ep_dir, f"acciones_ep{ep}.txt")
    with open(txt_path, "w") as f:
        f.write("step steering gas brake\n")

    print(f"Jugando episodio {ep+1}...")

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_q, pygame.K_ESCAPE]:
                    done = True

        action = get_keyboard_action()
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        frames_episode.append((obs.copy(), action.copy()))

        with open(txt_path, "a") as f:
            f.write(f"{step} {action[0]:.2f} {action[1]:.2f} {action[2]:.2f}\n")

        obs = next_obs
        step += 1
        clock.tick(fps)

    expert_data.append(frames_episode)

    # Guardar frames como PNG
    if save_frames:
        for s, (frame_obs, _) in enumerate(frames_episode):
            Image.fromarray(frame_obs).save(f"{ep_dir}/step{s:05d}.png")

    # Guardar dataset acumulado
    np.save(save_npy, np.array(expert_data, dtype=object), allow_pickle=True)
    print(f"Episodio {ep} guardado (total: {len(expert_data)} episodios)")

# ----------------------------
# CIERRE
# ----------------------------
env.close()
print(f"Guardados {len(expert_data)} episodios en {save_npy}")
if save_frames:
    print(f"Frames y txt guardados en '{frames_dir}'")

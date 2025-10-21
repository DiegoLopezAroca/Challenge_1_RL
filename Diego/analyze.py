import numpy as np
import os

# Ruta al dataset y a los frames
save_npy = "/home/diego/RL/Challenge_1_RL/Diego/expert_data.npy"
frames_dir = "/home/diego/RL/Challenge_1_RL/frames/"

data = np.load(save_npy, allow_pickle=True)

def episode_stats(ep_idx, ep_data):
    actions = np.array([a for (_, a) in ep_data])
    steps = len(actions)
    mean_action = actions.mean(axis=0)
    std_action = actions.std(axis=0)
    gas_ratio = (actions[:,1] > 0.1).mean()
    brake_ratio = (actions[:,2] > 0.1).mean()
    return {
        "ep": ep_idx,
        "steps": steps,
        "steer_std": std_action[0],
        "gas_std": std_action[1],
        "brake_std": std_action[2],
        "gas_ratio": gas_ratio,
        "brake_ratio": brake_ratio,
        "mean_actions": mean_action
    }

stats = [episode_stats(i, ep) for i, ep in enumerate(data)]
stats = sorted(stats, key=lambda x: x["steps"], reverse=True)

print(f"{'Ep':<4} {'Steps':<6} {'SteerStd':<9} {'GasStd':<8} {'Gas%':<6} {'Brake%':<7}")
for s in stats:
    print(f"{s['ep']:<4} {s['steps']:<6} {s['steer_std']:<9.3f} {s['gas_std']:<8.3f} {s['gas_ratio']*100:<6.1f} {s['brake_ratio']*100:<7.1f}")

# Opcional: guardar a CSV para ver en Excel
import csv
with open("episode_stats.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(stats[0].keys()))
    writer.writeheader()
    writer.writerows(stats)

print("\nResultados guardados en episode_stats.csv")

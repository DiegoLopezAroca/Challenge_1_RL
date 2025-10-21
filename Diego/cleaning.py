import numpy as np

data = np.load("/home/diego/RL/Challenge_1_RL/Diego/expert_data.npy", allow_pickle=True)

# los mejores episodios
good_eps = [1]

filtered = [data[i] for i in good_eps]

np.save("/home/diego/RL/Challenge_1_RL/Diego/expert_data_filtered.npy", 
        np.array(filtered, dtype=object), allow_pickle=True)

print(f"Guardado expert_data_filtered.npy con {len(filtered)} episodios limpios.")

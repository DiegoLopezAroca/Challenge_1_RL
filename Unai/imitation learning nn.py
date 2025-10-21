import os
import numpy as np
from PIL import Image
import torch
from torch import nn    
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T 

frames_dir = "./UNAI/frames"
expert_data_file = "./UNAIexpert_data.npy"
batch_size = 64
num_epochs = 10
lr = 1e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

'''
Ahora tenemos que poner los datos recolectados en un formato (dataset) correcto
Al ejecutar el anterior script, conseguimos un frame (screenshot) de la pantalla (por cada segundo)
y las acciones tomadas (en un .txt)
El formato que necesiamos será: (frame, action)
action = [steering, gas, brake]
steering: [-1,0, 1.0], -1 izquierda, 0 recto, 1 derecha
gas: [0, 1.0], aceleración o gas
brake: [0, 1.0], aún así parece ser que suele ser un valor constante 0.8
'''

class CustomDataset(Dataset):
    
    def __init__(self, frames_dir, transform=None):
        self.trials = []
        self.transform = transform
        for ep in sorted(os.listdir(frames_dir)):
            ep_dir = os.path.join(frames_dir, ep) # así recorreremos cada subcarpeta
            txt_path = os.path.join(ep_dir, f"acciones_{ep}.txt")
            with open(txt_path, "r") as f:
                next(f) # para saltarnos el header
                lines = f.readlines()
            for l in lines:
                l = l.strip().split()
                if len(l) != 4:
                    continue # No están todos los datos
                step, steer, gas, brake = map(float, l)
                img_path = os.path.join(ep_dir, f"step{int(step):05d}.png")
                if os.path.exists(img_path):
                    self.trials.append(
                    (img_path, np.array([steer, gas, brake], dtype=np.float32))
                )

    def __len__(self):
        return len(self.trials)
    
    def __getitem__(self, idx):
        img_path, action = self.trials[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(action)
    
# Para evitarnos problemas de dimensionalidades grandes, reduciremos las dimensiones de los frames a una medida arbitraria (puede que lo cambiemos más adelante)
transform = T.Compose([
    T.Resize((96, 96)),
    T.ToTensor(),
])

dataset = CustomDataset(frames_dir, transform)
dataloader = DataLoader(dataset, batch_size, shuffle=True)

# Ahora definiremos el NN, en nuestro caso una CNN (va a ser sencilla)
class ImitationCNN(nn.Module):
    def __init__(self):
        super(ImitationCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2), 
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 5, stride=2),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self.fc = None  

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        if self.fc is None:
            self.fc = nn.Sequential(
                nn.Linear(x.shape[1], 256),
                nn.ReLU(),
                nn.Linear(256, 3),  # salida = [steer, gas, brake]
                nn.Tanh(),
            ).to(x.device)
        return self.fc(x)

    
model = ImitationCNN().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr)

# Entrenamiento ahora
for epoch in range(num_epochs):
    total_loss = 0.0
    for imgs, actions in dataloader:
        imgs, actions = imgs.to(device), actions.to(device)
        preds = model(imgs)
        loss = criterion(preds, actions)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss/len(dataloader):.4f}")
    
torch.save(model.state_dict(), "imitation_model.pth")
print(f"Modelo guardado")
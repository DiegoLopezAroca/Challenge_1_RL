import os
import numpy as np
import random
from typing import List, Tuple, Optional, Deque
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---------- Config ----------
NPY_PATH = "C:\\Users\\diego\\Carrera\\4º\\RL\\Challenge_1_RL\\UNAI\\expert_data.npy"
   # tu archivo
STACK = 4                      # nº de frames apilados (historia). Pon 1 si no quieres apilar
BATCH_SIZE = 128
LR = 1e-4
EPOCHS = 20
VAL_SPLIT = 0.1
RANDOM_SEED = 42
IMG_SIZE = 84                  # reescalar a 84x84 (rápido y suficiente)
GRAYSCALE = True               # usar escala de grises (reduce params)
NORM_MEAN = 0.5
NORM_STD  = 0.5                # normaliza a [-1,1] aprox

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# ---------- Utils ----------
def to_grayscale(img: np.ndarray) -> np.ndarray:
    # img: HxWx3 uint8 -> HxWx1 float32 [0,1]
    img_f = img.astype(np.float32) / 255.0
    gray = 0.299*img_f[:,:,0] + 0.587*img_f[:,:,1] + 0.114*img_f[:,:,2]
    return gray[..., None]  # HxWx1

def resize_nn(img: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    # bilinear usando torch (rápido y correcto)
    t = torch.from_numpy(img.transpose(2,0,1)).unsqueeze(0)  # 1xCxHxW
    t = F.interpolate(t, size=(new_h, new_w), mode="bilinear", align_corners=False)
    out = t.squeeze(0).numpy().transpose(1,2,0)
    return out

def normalize(img: np.ndarray) -> np.ndarray:
    # (x - mean)/std
    return (img - NORM_MEAN) / NORM_STD

# ---------- Dataset ----------
class BCDataset(Dataset):
    """
    A partir del .npy (episodios con (obs, action)):
    - opcionalmente apila últimos K frames
    - devuelve tensor (C,H,W) y acción (3,)
    """
    def __init__(self, npy_path: str, stack: int = 4, grayscale: bool = True,
                 img_size: int = 84, val: bool = False, val_ratio: float = 0.1):
        data = np.load(npy_path, allow_pickle=True)  # list of episodes
        all_pairs = []  # list of (obs, action)
        for ep in data:
            for (obs, act) in ep:
                all_pairs.append((obs, act))

        # Split train/val
        idxs = list(range(len(all_pairs)))
        random.shuffle(idxs)
        val_count = int(len(idxs) * val_ratio)
        if val:
            idxs = idxs[:val_count]
        else:
            idxs = idxs[val_count:]

        # Preprocesamos y además preconstruimos una deques por episodio para stacking.
        # Como perdimos la noción de episodios arriba, implementamos un stack "local":
        # usamos un buffer FIFO que se resetea cuando detectemos un corte (no detectable aquí).
        # Alternativa robusta: construir ejemplos con stack dentro del bucle de episodios ANTES
        # de mezclar. Para simplicidad, haremos prox: para cada índice, repetimos el frame actual
        # K veces si no hay historia previa suficiente.
        self.frames: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.stack = stack
        self.grayscale = grayscale
        self.img_size = img_size

        for i in idxs:
            obs, act = all_pairs[i]
            # Preproc única del frame
            if grayscale:
                img = to_grayscale(obs)  # HxWx1 in [0,1]
            else:
                img = obs.astype(np.float32)/255.0  # HxWx3

            img = resize_nn(img, img_size, img_size)  # HxWxC
            img = normalize(img)  # (x-mean)/std

            self.frames.append(img.astype(np.float32))
            self.actions.append(act.astype(np.float32))

        # Para stacking “sintético”, duplicamos el frame actual K veces (baseline sólida)
        # Si quieres un stack real secuencial, construye el dataset episodio por episodio.
        # (Te dejo abajo una versión alternativa por episodio)
    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        img = self.frames[idx]            # HxWxC
        act = self.actions[idx]           # (3,)
        # Build stack
        if self.stack <= 1:
            x = img.transpose(2,0,1)      # CxHxW
        else:
            # Repite el mismo frame K veces como baseline
            x = np.repeat(img, self.stack, axis=2).transpose(2,0,1)  # (C*K)xHxW

        x = torch.from_numpy(x).float()
        y = torch.from_numpy(act).float()
        return x, y

# ---------- Modelo CNN sencillo (tipo Nature, continuo) ----------
class ActorCNN(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(inplace=True),
        )
        # calcular tamaño conv out
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, IMG_SIZE, IMG_SIZE)
            n_flat = self.conv(dummy).view(1, -1).shape[1]
        self.fc = nn.Sequential(
            nn.Linear(n_flat, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 3)  # steer, gas, brake
        )

    def forward(self, x):
        z = self.conv(x)
        z = z.view(z.size(0), -1)
        out = self.fc(z)
        # steer en [-1,1], gas/brake en [0,1]
        steer = torch.tanh(out[:, 0:1])
        gas   = torch.sigmoid(out[:, 1:2])
        brake = torch.sigmoid(out[:, 2:3])
        return torch.cat([steer, gas, brake], dim=1)

def make_loaders():
    in_channels = (1 if GRAYSCALE else 3) * STACK
    train_ds = BCDataset(NPY_PATH, stack=STACK, grayscale=GRAYSCALE, img_size=IMG_SIZE,
                         val=False, val_ratio=VAL_SPLIT)
    val_ds   = BCDataset(NPY_PATH, stack=STACK, grayscale=GRAYSCALE, img_size=IMG_SIZE,
                         val=True,  val_ratio=VAL_SPLIT)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader, in_channels

def train():
    train_loader, val_loader, in_channels = make_loaders()
    model = ActorCNN(in_channels).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-6)
    # MSE está bien para BC continuo
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=="cuda"))

    best_val = float("inf")
    for epoch in range(1, EPOCHS+1):
        model.train()
        tr_loss = 0.0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type=="cuda")):
                yhat = model(x)
                loss = F.mse_loss(yhat, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tr_loss += loss.item() * x.size(0)
        tr_loss /= len(train_loader.dataset)

        # Validación
        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                yhat = model(x)
                loss = F.mse_loss(yhat, y)
                va_loss += loss.item() * x.size(0)
        va_loss /= len(val_loader.dataset)

        print(f"[{epoch}/{EPOCHS}] train_mse={tr_loss:.4f}  val_mse={va_loss:.4f}")

        if va_loss < best_val:
            best_val = va_loss
            torch.save({"model": model.state_dict(),
                        "in_channels": in_channels,
                        "img_size": IMG_SIZE,
                        "grayscale": GRAYSCALE,
                        "stack": STACK}, "bc_carracing.pt")
            print("  ✓ Guardado: bc_carracing.pt  (mejor val)")

if __name__ == "__main__":
    train()

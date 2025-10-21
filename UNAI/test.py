import torch, numpy as np
import torch.nn.functional as F
import torch.nn as nn
import gymnasium as gym

# Debe coincidir con lo usado en entrenamiento
GRAYSCALE=True; IMG_SIZE=84; STACK=1; NORM_MEAN=0.5; NORM_STD=0.5

class ActorCNN(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, 4), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU(inplace=True),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, IMG_SIZE, IMG_SIZE)
            n_flat = self.conv(dummy).view(1, -1).shape[1]
        self.fc = nn.Sequential(
            nn.Linear(n_flat, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 3)
        )
    def forward(self, x):
        z = self.conv(x).view(x.size(0), -1)
        out = self.fc(z)
        steer = torch.tanh(out[:,0:1])
        gas   = torch.sigmoid(out[:,1:2])
        brake = torch.sigmoid(out[:,2:3])
        return torch.cat([steer, gas, brake], dim=1)

def to_gray(img):
    img_f = img.astype(np.float32)/255.0
    g = 0.299*img_f[:,:,0] + 0.587*img_f[:,:,1] + 0.114*img_f[:,:,2]
    return g[...,None]

def resize_nn(img, new_h, new_w):
    t = torch.from_numpy(img.transpose(2,0,1)).unsqueeze(0)
    t = F.interpolate(t, size=(new_h, new_w), mode="bilinear", align_corners=False)
    return t.squeeze(0).numpy().transpose(1,2,0)

def normalize(img):
    return (img - NORM_MEAN)/NORM_STD

def preprocess(obs):
    img = to_gray(obs) if GRAYSCALE else obs.astype(np.float32)/255.0
    img = resize_nn(img, IMG_SIZE, IMG_SIZE)
    img = normalize(img).astype(np.float32)
    x = img.transpose(2,0,1)  # CxHxW
    return x

def main():
    ckpt = torch.load("bc_carracing.pt", map_location="cpu")
    in_channels = ckpt.get("in_channels", (1 if GRAYSCALE else 3)*STACK)
    model = ActorCNN(in_channels)
    model.load_state_dict(ckpt["model"])
    model.eval()

    env_name = "CarRacing-v3"
    env = gym.make(env_name, render_mode="human")
    for ep in range(3):
        obs, _ = env.reset()
        done = False
        # buffer simple para stack
        frames = []
        while not done:
            x = preprocess(obs)
            frames.append(x)
            while len(frames) < STACK:
                frames.insert(0, x)
            frames = frames[-STACK:]
            x_stacked = np.concatenate(frames, axis=0)  # (C*STACK,H,W)
            with torch.no_grad():
                a = model(torch.from_numpy(x_stacked).unsqueeze(0).float()).squeeze(0).numpy()
            obs, r, term, trunc, _ = env.step(a)
            done = term or trunc
    env.close()

if __name__ == "__main__":
    main()

import torch
import torchvision.transforms as T
from PIL import Image
from gymnasium.envs.box2d.car_racing import CarRacing
from imitationLearningCNN import ImitationCNN # importamos la CNN que hemos hecho antes

device = "cuda" if torch.cuda.is_available() else "cpu" # creo que Diego es el único con Cuda
transform = T.Compose([
    T.Resize((96, 96)),
    T.ToTensor()
])

model = ImitationCNN().to(device)
model.load_state_dict(torch.load("Unai/imitation_model.pth", map_location=device))
model.eval()

env = CarRacing(render_mode="human")
obs, _ = env.reset()
done = False

while not done:
    img = Image.fromarray(obs)
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        action = model(img).cpu().numpy()[0]
    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

env.close()
import matplotlib.pyplot as plt
import numpy as np
import torch

from model import Generator

data = torch.load("clarifai_100_128.pth")
model = Generator(128)
model.eval()
model.load_state_dict(data["model_state_dict"])

arr = torch.Tensor(np.load("test/128/elon_musk.npy").reshape(-1, 128))
output = model(arr).permute(0, 2, 3, 1).detach().numpy()
plt.imshow(output[0])
plt.show()

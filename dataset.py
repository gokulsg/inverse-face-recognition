import glob

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms

class EmbedImagePairs(Dataset):
    def __init__(self, data_dir, train=True, size=64, n_hidden=128):
        super().__init__()

        self.embeds = sorted(glob.glob(f'{data_dir}/**.npy'))
        
        self.images = [direc.replace(".npy", ".jpg") for direc in self.embeds]

        self.train = train
        self.n_hidden = n_hidden
        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor()
        ])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        embedding = np.load(self.embeds[idx])

        image = self.transform(image)

        if self.train: embedding = embedding + np.random.randn(self.n_hidden) * 0.0001
        embedding = embedding.astype(np.float32)

        return embedding, image

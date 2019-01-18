import torch
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader

from dataset import EmbedImagePairs
from model import Generator

# face_detector = dlib.get_frontal_face_detector()
# landmark_detector = dlib.shape_predictor('models/dlib/shape_predictor_68_face_landmarks.dat')
# face_embedder = dlib.face_recognition_model_v1('models/dlib/dlib_face_recognition_resnet_model_v1.dat')

def train_epoch(model, trn_dataloader, criterion, optimizer, device):
    model.train()

    for x, y in trn_dataloader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        outputs = model(x)

        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
    
    print(loss.item())

def val_epoch(model, val_dataloader, criterion, device):
    pass 

def mean_average_distance():
    pass 

def save_model(epoch, model, optimizer, fname):
    print('saving')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        }, f'fname_{epoch}.pth')

def train(model, dataloaders, criterion, optimizer, device, out_name, num_epochs=100):
    for epoch in tqdm(range(num_epochs)):
        train_epoch(model, dataloaders['train'], criterion, optimizer, device)
        print(epoch)
    
    save_model(epoch, model, optimizer, out_name)

def distance_metric(pred_photo, input, dlib_models):
    pass
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import EmbedImagePairs
from model import Generator

# face_detector = dlib.get_frontal_face_detector()
# landmark_detector = dlib.shape_predictor('models/dlib/shape_predictor_68_face_landmarks.dat')
# face_embedder = dlib.face_recognition_model_v1('models/dlib/dlib_face_recognition_resnet_model_v1.dat')

def train_epoch(model, trn_dataloader, criterion, optimizer, device):
    model.train()

    running_loss = 0

    for x, y in trn_dataloader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        outputs = model(x)

        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
    
    trn_loss = running_loss/len(trn_dataloader.dataset)
    
    return trn_loss

def distance_metric(pred_photo, input_emb, dlib_models, ret_emb=False):
    face_detector, landmark_detector, face_embedder = dlib_models
    
    dist = None 
    embedding = None
    
    face_boxes = face_detector(pred_photo, 1)

    if len(face_boxes) == 1:
        landmarks = landmark_detector(pred_photo, face_boxes[0])
        embedding = np.array(face_embedder.compute_face_descriptor(pred_photo, landmarks))
        
        assert embedding.shape == input_emb.shape; "shape of true embedding and predicted embedding not same"
        
        dist = np.linalg.norm(np.array(embedding) - np.array(input_emb))
    
    if ret_emb:
        return dist, embedding
    else:
        return dist

def val_epoch(model, val_dataloader, criterion, dlib_models, device):
    model.eval() 
    dists = []

    running_loss = 0

    for x, y in val_dataloader:
        x = x.to(device)
        y = y.to(device)

        outputs = model(x)

        loss = criterion(outputs, y)
        running_loss += loss.item() * x.size(0)
        
        x = x.cpu().numpy()
        photos = outputs.detach().permute(0, 2, 3, 1).cpu().numpy()
        photos = (photos*255).astype('uint8')
            
        for input_emb, photo in zip(x, photos):
            dist = distance_metric(photo, input_emb, dlib_models)
            if dist is not None: dists += [dist]
        
    val_loss = running_loss/len(val_dataloader.dataset)

    return val_loss, dists

def save_model(epoch, model, optimizer, fname):
    print(f'saving at epoch {epoch}')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        }, f'{fname}_{epoch}.pth')


def train(model, dataloaders, criterion, optimizer, device, out_name, dlib_models=None, validate=False, validate_every=10, num_epochs=100):
    trn_losses = []
    
    if validate:
        avg_dists = []
        val_losses = []
        
        assert len(dataloaders) == 2
        assert dlib_models is not None
    
    # start at zero, end at num_epochs (including)
    for epoch in tqdm(range(1, num_epochs+1)):
        # Training phase
        trn_loss = train_epoch(model, dataloaders['train'], criterion, optimizer, device)
        trn_losses += [trn_loss]

        # Validation Phase
        if epoch % validate_every == 0:
            save_model(epoch, model, optimizer, out_name)

            if validate:
                val_loss, dists = val_epoch(model, dataloaders['val'], criterion, dlib_models, device)
                val_losses += [val_loss]

                avg_dist = np.mean(dists)
                avg_dists += [avg_dist]

                np.save(out_name + str(epoch) + "_dists.npy", dists)

                print("Epoch: ", epoch, "Train Loss:", trn_loss, "Val Loss:", val_loss, "Average Distance:", avg_dist)
    
    save_model(epoch, model, optimizer, out_name + "Final")
    
    np.save(out_name + "trn_losses.npy", trn_losses)
    
    if validate: 
        np.save(out_name + "val_losses.npy", val_losses)
        np.save(out_name + "avg_dists.npy", avg_dists)

def test(model, test_dataloader, dlib_models, device, raw_dists=False):
    model.eval() 
    dists = []
    
    if raw_dists:
        true_vecs = []
        synth_vecs = []
        
    for x, y in test_dataloader:
        x = x.to(device)
        y = y.to(device)

        outputs = model(x)
        
        x = x.cpu().numpy()
        photos = outputs.detach().permute(0, 2, 3, 1).cpu().numpy()
        photos = (photos*255).astype('uint8')
            
        for input_emb, photo in zip(x, photos):
            if not raw_dists:
                dist = distance_metric(photo, input_emb, dlib_models, raw_dists)
                if dist is not None: dists += [dist]
            else:
                dist, embedding = distance_metric(photo, input_emb, dlib_models, raw_dists)
                true_vecs += [input_emb]
                synth_vecs += [embedding]
    
    if raw_dists:
        return true_vecs, synth_vecs
    else:
        return dists

def synthesize(model, vis_dataloader, device):
    model.eval()
    
    all_photos = []
    
    for x, y in vis_dataloader:
        x = x.to(device)
        y = y.to(device)

        outputs = model(x)
        
        x = x.cpu().numpy()
        photos = outputs.detach().permute(0, 2, 3, 1).cpu().numpy()
        photos = (photos*255).astype('uint8')
        
        all_photos += [photos]
    
    all_photos = np.concatenate(all_photos, axis=0)
    names = vis_dataloader.dataset.embeds
    
    return all_photos, names
    
        
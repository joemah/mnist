from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from torch.utils.data.dataset import Subset
from torchvision.transforms import Compose, ToTensor
from torchvision.datasets import MNIST
from copy  import deepcopy
from tqdm import tqdm 
from auto import AutoEncoder
import matplotlib.pyplot as plt
from torch import nn
import numpy as np 
import random 
import torch
import os 


def trainfunc(net, dataloader,loss_fn,optimizer,device):
    net.train()
    train_loss = []
    for batch in dataloader:
        imgbatch = batch[0].to(device) # extract data and move tensors to the device
        output = net(imgbatch) # forward pass
        loss = loss_fn(output,imgbatch)
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.data.item())
    
    return np.mean(train_loss)

def testfunc(net, dataloader, loss_fn,optimizer,device):
    net.eval() # evaluation mode
    with torch.no_grad(): # no  need to track gradients
        out = torch.Tensor().float()
        lab = torch.Tensor().float()
        
        for batch in  dataloader:
            imagbatch = batch[0].to(device) # move extacted data to  device
            output = net(imagbatch) # forward pass
            # concatenate with previous outputs
            conc_out = torch.cat([out, output.cpu()])
            conc_lab = torch.cat([lab, imagbatch.cpu()])
            del imagbatch
        # evaluate loss
        val_loss = loss_fn(conc_out, conc_lab)
    
    return val_loss.data
        

def train_cross_val(idxs, device, train_data, encoded_dim=8, lr=1e-3,wd=0,epochs=20):
    kfold = KFold(n_splits=3, random_state=42,shuffle=True)
    train_loss_log = []
    val_loss_log = []
    for fold, (tranidx,validx) in enumerate(kfold.split(idxs)):
        print(f'Fold: {fold}')
        train_loss_log_fold = []
        val_loss_log_fold = []
        # initilize the net 
        cvnet = AutoEncoder(encoded_space_dim=encoded_dim)
        cvnet.to(device)
        loss_fn = torch.nn.MSELoss()
        optim = torch.optim.Adam(cvnet.parameters(),lr=lr,weight_decay=wd)
        train_dl_fold = DataLoader(Subset(train_data, tranidx),batch_size=500,shuffle=False)
        val_dl_fold = DataLoader(Subset(train_data, validx),batch_size=500,shuffle=False)

        for epoch in range(epochs):
            print('|Epoch> %d/%d'%(epoch+1, epochs))
            avg_tranloss = trainfunc(cvnet,train_dl_fold,loss_fn,optim,device)
            avg_valloss = trainfunc(cvnet,val_dl_fold,loss_fn,optim,device)
            print('\tTraining loss: %f' % (avg_tranloss))
            print('\tValidation loss: %f' % (avg_valloss))

            train_loss_log_fold.append(avg_tranloss)
            val_loss_log_fold.append(avg_valloss)
        
        train_loss_log.append(train_loss_log_fold)
        val_loss_log.append(val_loss_log_fold)
    
    return {"train_loss": np.mean(train_loss_log, axis=0),
            "val_loss": np.mean(val_loss_log, axis=0)}



def test_rand_noise(net, dataloader, loss_fn, device, noise, sigma = 1, plot = True):
    """
    Test the trained autoencoder on randomly corrupted images. 
    The random  noise can be generated using the 'gaussian', 'uniform', or 'occlusion'
    """
    np.random.seed(42)
    net.eval() # evaluation mode (e.g. disable dropout)
    with torch.no_grad(): # No need to track the gradients
        conc_out = torch.Tensor().float()
        conc_label = torch.Tensor().float()
        for batch in dataloader:
            imgbatch = batch[0].to(device) # Extract data and move tensors to the selected device
            # Add noise
            gaussian = ''
            if noise == 'Gaussian':
                distortion = torch.Tensor(np.random.normal(0,sigma,batch[0].shape))
                distorted_image = (batch[0] + distortion).to(device)
                gaussian = f' with N(0, {sigma})'
            if noise == 'Uniform':
                distortion = torch.Tensor(np.random.rand(*batch[0].shape))
                distorted_image = (batch[0] + distortion).to(device)
            if noise == 'Occlusion':
                idx = np.random.choice((0,1), 2)
                distorted_image = deepcopy(imgbatch)
                distorted_image[:, :, idx[0]*14:(idx[0]+1)*14, idx[1]*14:(idx[1]+1)*14] = 0

            # Forward pass
            out = net(distorted_image)

            # Concatenate with previous outputs
            conc_out = torch.cat([conc_out, out.cpu()])
            conc_label = torch.cat([conc_label, imgbatch.cpu()])

        # plot images
        if plot:
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.title("Original", fontsize=18)
            plt.imshow(imgbatch[6].squeeze().cpu(), cmap='gist_gray')
            plt.subplot(1, 3, 2)
            plt.title(f"{noise} noise"+gaussian, fontsize=18)
            plt.imshow(distorted_image[6].squeeze().cpu(), cmap='gist_gray')
            plt.subplot(1, 3, 3)
            plt.title("Reconstructed", fontsize=18)
            plt.imshow(out[6].squeeze().cpu(), cmap='gist_gray')
            plt.savefig(f"./images/{noise}" + gaussian + ".png")
            plt.show()


        # evaluate global loss
        val_loss = loss_fn(conc_out, conc_label)

    return val_loss.data
import os
import torch
import time
import numpy as np
from torch import nn, optim
import matplotlib.pyplot as plt
from datetime import datetime

from torchvision import transforms

from torch.utils.data import DataLoader
from fastai.vision.learner import create_body
from torchvision.models.resnet import resnet18
from torchvision.io import read_image, ImageReadMode
from fastai.vision.models.unet import DynamicUnet
from tqdm.notebook import tqdm

global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Datasets
class Places365():
    def __init__(self, dir_path: str, batch_size: int = 64, split: str = None, size_transform=None):
        if split is None:
          self.dir_path = dir_path
        else:
          self.dir_path = os.path.join(dir_path, split)

        self.img_names = os.listdir(self.dir_path)
        self.batch_size = batch_size
        self.size_transform = size_transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.dir_path, self.img_names[idx])

        bw_img = read_image(img_path, mode=ImageReadMode.GRAY)
        color_img = read_image(img_path, mode=ImageReadMode.RGB)

        if self.size_transform:
            bw_img = self.size_transform(bw_img)
            color_img = self.size_transform(color_img)


        bw_img = bw_img / 255
        color_img = color_img / 255

        return {'bw': bw_img, 'color': color_img}


def make_dataloader(batch_size=32, n_workers=2, pin_memory=True, **kwargs):
    dataset = Places365(**kwargs)
    dataloader = DataLoader(dataset,
                            shuffle=False,
                            batch_size=batch_size,
                            num_workers=n_workers,
                            pin_memory=pin_memory)
    return dataloader


# Generator
def build_res_unet(n_input=1, n_output=3, size=256):
    body = create_body(resnet18(), pretrained=True, n_in=n_input, cut=-2)
    net_G = DynamicUnet(body, n_output, (size, size)).to(device)
    return net_G

def pretrain_generator(net_G, train_loader, opt, criterion, epochs, checkpoints_dir='', start_epoch=0):
    if start_epoch > 0:
      resume_epoch = start_epoch - 1
      resume(net_G, os.path.join(checkpoints_dir, f"epoch-{resume_epoch}.pth"))


    for e in range(start_epoch, epochs):
        loss_meter = AverageMeter()
        for data in tqdm(train_loader):
            bw, color = data['bw'].to(device), data['color'].to(device)
            preds = net_G(bw)
            loss = criterion(preds, color)
            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_meter.update(loss.item(), bw.size(0))

        print(f"Epoch {e + 1}/{epochs}")
        print(f"L1 Loss: {loss_meter.avg:.5f}")
        checkpoint(net_G, os.path.join(checkpoints_dir, f"epoch-{e}.pth"))


# Discriminator
class PatchDiscriminator(nn.Module):
    def __init__(self, input_channels=3, num_filters=64, n_down=3):
        super().__init__()
        model = [self.get_layers(input_channels, num_filters, norm=False)]

        # making downsampling blocks
        # stride 2 except for last block
        model += [self.get_layers(num_filters * 2 ** i, num_filters * 2 ** (i + 1), s=1 if i == (n_down-1) else 2)
                          for i in range(n_down)]

        # last layer doesn't get normalization nor activation
        model += [self.get_layers(num_filters * 2 ** n_down, 1, s=1, norm=False, act=False)]

        self.model = nn.Sequential(*model)

    # function to make block of layers
    def get_layers(self, ni, nf, k=4, s=2, p=1, norm=True, act=True):
        layers = [nn.Conv2d(ni, nf, k, s, p, bias=not norm)]

        if norm:
          layers += [nn.BatchNorm2d(nf)]

        if act:
          # 0.2 slope as sugested from Radford A., Metz L., and Chintala S. (2016)
          layers += [nn.LeakyReLU(0.2, True)]

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Loss
class GANLoss(nn.Module):
    def __init__(self, real_label=1.0, fake_label=0.0):
        super().__init__()
        self.register_buffer('real_label', torch.tensor(real_label))
        self.register_buffer('fake_label', torch.tensor(fake_label))
        self.loss = nn.BCEWithLogitsLoss()

    def get_labels(self, preds, target_is_real):
        if target_is_real:
            labels = self.real_label
        else:
            labels = self.fake_label
        return labels.expand_as(preds)

    def __call__(self, preds, target_is_real):
        labels = self.get_labels(preds, target_is_real)
        loss = self.loss(preds, labels)
        return loss

# Main model
class MainModel(nn.Module):
    def __init__(self, net_G=None, lr_G=2e-4, lr_D=2e-4,
                 beta1=0.5, beta2=0.999, lambda_L1=100.):
        super().__init__()

        self.device = device
        self.lambda_L1 = lambda_L1

        # If the generator is not provided, a default untrained one is used
        if net_G is None:
            net_G = build_res_unet(n_input=1, n_output=3, size=256)

        # Initialize generator and discriminator
        self.net_G = net_G.to(device)
        self.net_D = init_model(PatchDiscriminator(input_channels=3, n_down=3, num_filters=64), self.device)

        # Initialize losses
        self.GANcriterion = GANLoss().to(self.device)
        self.L1criterion = nn.L1Loss()

        # Initialize optimizers
        self.opt_G = optim.Adam(self.net_G.parameters(), lr=lr_G, betas=(beta1, beta2))
        self.opt_D = optim.Adam(self.net_D.parameters(), lr=lr_D, betas=(beta1, beta2))

    def set_requires_grad(self, model, requires_grad=True):
        """"
        Sets requires_grad attribute for all parameters in model
        """
        for p in model.parameters():
            p.requires_grad = requires_grad

    def setup_input(self, data):
        self.gray = data['bw'].to(self.device)
        self.color = data['color'].to(self.device)

    # Generator forward
    def forward(self):
        self.fake_color = self.net_G(self.gray)

    def predict(self, data):
        with torch.no_grad():
            self.setup_input(data)
            self.forward()
        return self.fake_color

    def test_predict(self, data):
        with torch.no_grad():
            data = data.to(self.device)
            pred = self.net_G(data)
        return pred
        
    def backward_D(self):
        # Generate fake images and predictions
        fake_image = self.fake_color
        fake_preds = self.net_D(fake_image.detach()) # detach to prevent gradients from flowing back

        # Compute discriminator loss (its ability to guess correctly)
        self.loss_D_fake = self.GANcriterion(fake_preds, False)

        # Get real images and their discriminator predictions
        real_image = self.color
        real_preds = self.net_D(real_image)

        # Compute discriminator loss
        self.loss_D_real = self.GANcriterion(real_preds, True)

        # Combine the losses
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        # Backward pass
        self.loss_D.backward()

    # Generator backward
    def backward_G(self):
        # Generate fake images and predictions
        fake_image = self.fake_color
        fake_preds = self.net_D(fake_image)

        # Compute generator loss (its ability to trick the discriminator)
        self.loss_G_GAN = self.GANcriterion(fake_preds, True)

        # Compute L1 loss (difference between generated and real image)
        self.loss_G_L1 = self.L1criterion(self.fake_color, self.color) * self.lambda_L1

        # Combine the losses
        self.loss_G = self.loss_G_GAN + self.loss_G_L1

        # Backward pass
        self.loss_G.backward()

    def optimize(self):
        # Generator forward pass
        self.forward()

        # Set discriminator to train mode and enable gradient computation
        self.net_D.train()
        self.set_requires_grad(self.net_D, True)

        # Reset discriminator gradient
        self.opt_D.zero_grad()

        # Discriminator backward pass
        self.backward_D()

        # Update discriminator parameters
        self.opt_D.step()

        # Set generator to train mode and disable gradient computation for discriminator
        self.net_G.train()
        self.set_requires_grad(self.net_D, False)

        # Reset generator gradient
        self.opt_G.zero_grad()

        # Generator backward pass
        self.backward_G()

        # Update generator parameters
        self.opt_G.step()

    def save_weights(self, filepath):
        torch.save({
            'net_G_state_dict': self.net_G.state_dict(),
            'net_D_state_dict': self.net_D.state_dict(),
            'opt_G_state_dict': self.opt_G.state_dict(),
            'opt_D_state_dict': self.opt_D.state_dict()
        }, filepath)

    def load_weights(self, filepath):
        checkpoint = torch.load(filepath, map_location=torch.device(self.device))
        self.net_G.load_state_dict(checkpoint['net_G_state_dict'])
        self.net_D.load_state_dict(checkpoint['net_D_state_dict'])
        self.opt_G.load_state_dict(checkpoint['opt_G_state_dict'])
        self.opt_D.load_state_dict(checkpoint['opt_D_state_dict'])

def train_model(model, train_dl, epochs, display_every=300, checkpoints_dir='', start_epoch=0):
    if start_epoch > 0:
        resume_epoch = start_epoch - 1
        checkpoint_file = list(filter(lambda x: x.startswith(f'epoch-{resume_epoch}'), os.listdir(checkpoints_dir)))[0]
        checkpoint_file = os.path.join(checkpoints_dir, checkpoint_file)
        model.load_weights(checkpoint_file)

    for e in range(start_epoch, epochs):
        loss_meter_dict = create_loss_meters() # function returing a dictionary of objects to
        i = 0                                  # log the losses of the complete network
        for data in tqdm(train_dl):
            model.setup_input(data) # input batch data
            model.optimize() # train batch
            update_losses(model, loss_meter_dict, count=data['bw'].size(0)) # function updating the log objects
            i += 1

            if i % display_every == 0:
                print(f"\nEpoch {e+1}/{epochs}")
                print(f"Iteration {i}/{len(train_dl)}")
                log_results(loss_meter_dict) # function to print out the losses

                data_cpu = {key: value.cpu() for key, value in data.items()}

                visualize(model, data_cpu, save=False) # function displaying the model's outputs

    timestamp = datetime.now().strftime("%d%H%M")
    model.save_weights(os.path.join(checkpoints_dir, f'epoch-{e}_{timestamp}.pth'))



class AverageMeter():
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def create_loss_meters():
    loss_D_fake = AverageMeter()
    loss_D_real = AverageMeter()
    loss_D = AverageMeter()
    loss_G_GAN = AverageMeter()
    loss_G_L1 = AverageMeter()
    loss_G = AverageMeter()

    return {'loss_D_fake': loss_D_fake,
            'loss_D_real': loss_D_real,
            'loss_D': loss_D,
            'loss_G_GAN': loss_G_GAN,
            'loss_G_L1': loss_G_L1,
            'loss_G': loss_G}

def update_losses(model, loss_meter_dict, count):
    for loss_name, loss_meter in loss_meter_dict.items():
        loss = getattr(model, loss_name)
        loss_meter.update(loss.item())

# initialize weights
# in this example using zero-centered Normal dist with std 0.02

def init_weights(net, gain=0.02):
    def init_func(model):
        classname = model.__class__.__name__
        if hasattr(model, 'weight') and 'Conv' in classname:
            nn.init.normal_(model.weight.data, mean=0.0, std=gain)

            if hasattr(model, 'bias') and model.bias is not None:
                nn.init.constant_(model.bias.data, 0.0)

        elif 'BatchNorm2d' in classname:
            nn.init.normal_(model.weight.data, 1., gain)
            nn.init.constant_(model.bias.data, 0.)

    net.apply(init_func)
    print(f"model initialized with norm initialization")
    return net


def init_model(model, device):
    model = model.to(device)
    model = init_weights(model)
    return model

def visualize(model, data, save=True):
    model.net_G.eval()
    with torch.no_grad():
        model.setup_input(data)
        model.forward()
    model.net_G.train()
    fake_color = model.fake_color.detach()
    real_color = model.color
    gray = model.gray

    fig = plt.figure(figsize=(15, 8))
    for i in range(5):
        ax = plt.subplot(3, 5, i + 1)
        ax.imshow(gray[i][0].cpu(), cmap='gray')
        ax.axis("off")
        ax = plt.subplot(3, 5, i + 1 + 5)
        fake_i = transforms.ToPILImage()(fake_color[i].cpu())
        fake_i = np.array(fake_i) / 255
        ax.imshow(fake_i)
        ax.axis("off")
        ax = plt.subplot(3, 5, i + 1 + 10)
        real_i = transforms.ToPILImage()(real_color[i].cpu())
        real_i = np.array(real_i)
        ax.imshow(real_i)
        ax.axis("off")
    plt.show()
    if save:
        fig.savefig(f"colorization_{time.time()}.png")

def log_results(loss_meter_dict):
    for loss_name, loss_meter in loss_meter_dict.items():
        print(f"{loss_name}: {loss_meter.avg:.5f}")

def checkpoint(model, filename):
    torch.save(model.state_dict(), filename)

def resume(model, filename):
    model.load_state_dict(torch.load(filename))

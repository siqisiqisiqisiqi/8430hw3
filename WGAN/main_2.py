from __future__ import print_function
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
from torch import autograd
from torch import optim
import pandas as pd

cudnn.benchmark = True

#set manual seed to a constant get a consistent output
manualSeed = random.randint(1, 10000)
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

image_size = 64

batch_size = 128

ngpu = 0

DIM = 64

lr = 2e-4

LAMBDA = 10

number_epochs = 25

nc = 3

nz = 128

CRITIC_ITERS = 5

ngf = 64

dataset = dset.CIFAR10(root="./data", download=True,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))


dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True)

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
            return output


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        main = nn.Sequential(
            nn.Conv2d(3, DIM, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(DIM, 2 * DIM, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(2 * DIM, 4 * DIM, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(4 * DIM, 8 * DIM, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(8 * DIM, 1, 4, 1, 0, bias=False)
        )

        self.main = main

    def forward(self, input):
        output = self.main(input)
        return output

netG = Generator(0)
netD = Discriminator()

netG.apply(weights_init)
netD.apply(weights_init)
print(netG)
print(netD)

optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.99))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.99))

use_cuda = torch.cuda.is_available()
if use_cuda:
    gpu = 0
if use_cuda:
    netD = netD.cuda(gpu)
    netG = netG.cuda(gpu)

def calc_gradient_penalty(netD, real_data, fake_data):
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size, int(real_data.nelement()/batch_size)).contiguous().view(batch_size, 3, 64, 64)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(gpu) if use_cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


one = torch.tensor(1, dtype=torch.float)
mone = one * -1

# input = torch.FloatTensor(batch_size, nc, image_size, image_size)
# fixed_noise = torch.FloatTensor(batch_size, nz, 1, 1).normal_(0, 1)

G_losses = []
D_losses = []

for epoch in range(number_epochs):
    for i, data in enumerate(dataloader, 0):
        for p in netD.parameters():
            p.requires_grad = True
        for j in range(CRITIC_ITERS):
            netD.zero_grad()

            real_cpu = data[0].to(device)
            input = real_cpu
            inputv = autograd.Variable(input)
            D_real = netD(inputv)
            D_real = D_real.mean()
            D_real.backward(mone)

            # train with fake
            noise = torch.randn(batch_size, nz,1,1)
            with torch.no_grad():
                noisev = autograd.Variable(noise)
            fake = autograd.Variable(netG(noisev).data)
            inputv = fake
            D_fake = netD(inputv)
            D_fake = D_fake.mean()
            D_fake.backward(one)

            gradient_penalty = calc_gradient_penalty(netD, inputv.data,fake.data)
            gradient_penalty.backward()

            D_cost = D_fake-D_real+gradient_penalty
            Wasserstein_D = D_real - D_fake
            optimizerD.step()

        for p in netD.parameters():
            p.requires_grad = False  # to avoid computation
        netG.zero_grad()

        noise = torch.randn(batch_size, nz,1,1)
        noisev = autograd.Variable(noise)
        fake = netG(noisev)
        G = netD(fake)
        G = G.mean()
        G.backward(mone)
        G_cost = -G
        optimizerG.step()

        G_losses.append(D_cost.item())
        D_losses.append(G_cost.item())

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f' % (
        epoch, number_epochs, i, len(dataloader), D_cost.item(), G_cost.item(), D_real, D_fake, G))

        if i % 100 == 0:
            print('saving the output')
            vutils.save_image(real_cpu, 'output2/real_samples.png', normalize=True)
            fake_img = netG(noisev)
            vutils.save_image(fake_img.detach(), 'output2/fake_samples_epoch_%03d.png' % (epoch), normalize=True)


    torch.save(netG.state_dict(), 'weights2/netG_epoch_%d.pth' % (epoch))
    torch.save(netD.state_dict(), 'weights2/netD_epoch_%d.pth' % (epoch))

dict = {'G_losses': G_losses, "D_losses": D_losses}
df = pd.DataFrame(dict)
df.to_csv('data2.csv')



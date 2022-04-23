import torch
import torch.nn as nn
from torchvision.utils import make_grid
import torch.optim as optim
import numpy as np
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.utils as vutils
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

tf = transforms.Compose([transforms.Resize(64),
                         transforms.ToTensor(),
                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                         ])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                        transform=tf)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                       transform=tf)

dataset = torch.utils.data.ConcatDataset([trainset, testset])

trainloader = torch.utils.data.DataLoader(dataset, batch_size=100,
                                           shuffle=True)

print(len(dataset))
print(dataset[0][0].size())
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck', 'fake')


def showImage(images, epoch=-99, idx=-99):
    images = images.cpu().numpy()
    images = images / 2 + 0.5
    save_images = torch.tensor(images)
    plt.imshow(np.transpose(images, axes=(1, 2, 0)))
    plt.axis('off')
    if epoch != -99:
        plt.savefig("output2/"+"e" + str(epoch) + "i" + str(idx) + ".png")
        vutils.save_image(save_images, 'output/fake_samples_epoch_%03d.png' % (epoch), normalize=True)


dataiter = iter(trainloader)
images, labels = dataiter.next()
print(images.size())
showImage(make_grid(images[0:64]))


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        # input 100*1*1
        self.layer1 = nn.Sequential(nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
                                    nn.ReLU(True))

        # input 512*4*4
        self.layer2 = nn.Sequential(nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(True))
        # input 256*8*8
        self.layer3 = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(True))
        # input 128*16*16
        self.layer4 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(True))
        # input 64*32*32
        self.layer5 = nn.Sequential(nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
                                    nn.Tanh())
        # output 3*64*64

        self.embedding = nn.Embedding(10, 100)

    def forward(self, noise, label):
        label_embedding = self.embedding(label)
        x = torch.mul(noise, label_embedding)
        x = x.view(-1, 100, 1, 1)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        # input 3*64*64
        self.layer1 = nn.Sequential(nn.Conv2d(3, 64, 4, 2, 1, bias=False),
                                    nn.BatchNorm2d(64),
                                    nn.LeakyReLU(0.2, True),
                                    nn.Dropout2d(0.5))

        # input 64*32*32
        self.layer2 = nn.Sequential(nn.Conv2d(64, 128, 4, 2, 1, bias=False),
                                    nn.BatchNorm2d(128),
                                    nn.LeakyReLU(0.2, True),
                                    nn.Dropout2d(0.5))
        # input 128*16*16
        self.layer3 = nn.Sequential(nn.Conv2d(128, 256, 4, 2, 1, bias=False),
                                    nn.BatchNorm2d(256),
                                    nn.LeakyReLU(0.2, True),
                                    nn.Dropout2d(0.5))
        # input 256*8*8
        self.layer4 = nn.Sequential(nn.Conv2d(256, 512, 4, 2, 1, bias=False),
                                    nn.BatchNorm2d(512),
                                    nn.LeakyReLU(0.2, True))
        # input 512*4*4
        self.validity_layer = nn.Sequential(nn.Conv2d(512, 1, 4, 1, 0, bias=False),
                                            nn.Sigmoid())

        self.label_layer = nn.Sequential(nn.Conv2d(512, 11, 4, 1, 0, bias=False),
                                         nn.LogSoftmax(dim=1))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        validity = self.validity_layer(x)
        plabel = self.label_layer(x)

        validity = validity.view(-1)
        plabel = plabel.view(-1, 11)

        return validity, plabel


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


gen = Generator().to(device)
gen.apply(weights_init)
batch_size = 100

gen.load_state_dict(torch.load('weights/netG_epoch_25.pth'))
noise = torch.randn(batch_size, 100, device=device)
sample_labels = torch.randint(0, 10, (batch_size,), device=device, dtype=torch.long)

fakes = gen(noise, sample_labels)
fake_image = fakes.detach()
for i in range(100):
    fake_image_1 = fake_image[i]
    vutils.save_image(fake_image_1, 'fake_image/image_%03d.jpg' % (i), normalize=True)
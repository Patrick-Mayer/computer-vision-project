import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# Basic 3 pooling U-net model
class UNet(nn.Module):
    def __init__(self, n_class):
        super(UNet, self).__init__()
        # U-net is a full CNN, so it's similar
        # Encoder

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # Maybe update to use F?

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # Maybe update to use F?

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # Maybe update to use F?

        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        # Decoder

        self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.deconv1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.deconv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.deconv3 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.deconv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.deconv5 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.deconv6 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Output layer
        self.outconv = nn.Conv2d(64, n_class, kernel_size=1)

    def forward(self, x):
        # Encoder

        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        xp1 = self.pool1(x2)

        x3 = F.relu(self.conv3(xp1))
        x4 = F.relu(self.conv4(x3))
        xp2 = self.pool2(x4)

        x5 = F.relu(self.conv5(xp2))
        x6 = F.relu(self.conv6(x5))
        xp3 = self.pool3(x6)

        x7 = F.relu(self.conv7(xp3))
        x8 = F.relu(self.conv8(x7))

        # Decoder

        xu1 = self.upconv1(x8)
        xu2 = torch.cat([xu1, x6], dim=1)
        xd1 = F.relu(self.deconv1(xu2))
        xd2 = F.relu(self.deconv2(xd1))

        xu3 = self.upconv2(xd2)
        xu4 = torch.cat([xu3, x4], dim=1)
        xd3 = F.relu(self.deconv3(xu4))
        xd4 = F.relu(self.deconv4(xd3))

        xu5 = self.upconv3(xd4)
        xu6 = torch.cat([xu5, x2], dim=1)
        xd5 = F.relu(self.deconv5(xu6))
        xd6 = F.relu(self.deconv6(xd5))

        # Output layer
        x = self.outconv(xd6)

        return x
    
model = UNet(n_class=2)
x = torch.randn(1, 3, 256, 256)  # batch of 1 RGB image
y = model(x)
print(y.shape)
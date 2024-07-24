import torch.nn as nn
import torch.nn.functional as F
import torch


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.enc1 = self.contracting_block(3, 64)
        self.enc2 = self.contracting_block(64, 128)
        self.enc3 = self.contracting_block(128, 256)
        self.enc4 = self.contracting_block(256, 512)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.upconv4 = self.expansive_block(512, 256)
        self.upconv3 = self.expansive_block(256, 128)
        self.upconv2 = self.expansive_block(128, 64)
        self.upconv1 = self.expansive_block(64, 64)
        
        self.final_layer = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        dec4 = self.center_crop(self.upconv4(self.pool(enc4)), enc4)
        dec3 = self.center_crop(self.upconv3(torch.cat((dec4, enc4), 1)), enc3)
        dec2 = self.center_crop(self.upconv2(torch.cat((dec3, enc3), 1)), enc2)
        dec1 = self.center_crop(self.upconv1(torch.cat((dec2, enc2), 1)), enc1)
        
        return torch.sigmoid(self.final_layer(torch.cat((dec1, enc1), 1)))

    def contracting_block(self, in_channels, out_channels, kernel_size=3):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=1),
            nn.ReLU(),
        )
        return block

    def expansive_block(self, in_channels, out_channels, kernel_size=3):
        block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=1),
            nn.ReLU(),
        )
        return block

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size.size(2)) // 2
        diff_x = (layer_width - target_size.size(3)) // 2
        return layer[:, :, diff_y:diff_y + target_size.size(2), diff_x:diff_x + target_size.size(3)]

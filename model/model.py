import torch
import torch.nn as nn


class ColorizationModel(nn.Module):

    def __init__(self, learning_rate = 1e-5):

        # Hyperparameters
        self.learning_rate = learning_rate

        # Trainable layers

        # Global Hints Network

        # input shape: (-1, 316, 1, 1)
        global_conv = nn.Sequential([
            nn.Conv2d(in_channels=316, out_channels=512, kernel_size=1, stride=1, padding=0, dilation=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0, dilation=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0, dilation=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0, dilation=1),
            nn.ReLU()])
        # output shape: (-1, 512, 1, 1)


        # Main network

        # conv1

        # input shape: (-1, 4, H, W)
        conv1 = nn.Sequential([
            nn.Conv2d(in_channels=4, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(),
            nn.BatchNorm2d(64)])
        # output shape: (-1, 64, H, W)

        # conv2

        # input shape: (-1, 64, H, W)
        conv2 = nn.Sequential([
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(),
            nn.BatchNorm2d(128)])
        # output shape: (-1, 128, H/2, W/2)

        # conv3

        # input shape: (-1, 128, H/2, W/2)
        conv3 = nn.Sequential([
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(),
            nn.BatchNorm2d(256)])
        # output shape: (-1, 256, H/4, W/4)

        # conv4

        # input shape: (-1, 256, H/4, W/4)
        conv4 = nn.Sequential([
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(),
            nn.BatchNorm2d(512)])
        # output shape: (-1, 512, H/8, W/8)

        # conv5

        # input shape: (-1, 512, H/8, W/8)
        conv5 = nn.Sequential([
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(),
            nn.BatchNorm2d(512)])
        # output shape: (-1, 512, H/8, W/8)

        # conv6

        # input shape: (-1, 512, H/8, W/8)
        conv6 = nn.Sequential([
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(),
            nn.BatchNorm2d(512)])
        # output shape: (-1, 512, H/8, W/8)

        # conv7

        # input shape: (-1, 512, H/8, W/8)
        conv7 = nn.Sequential([
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(),
            nn.BatchNorm2d(512)])
        # output shape: (-1, 512, H/8, W/8)

        # conv8

        # input shape: (-1, 512, H/8, W/8)
        conv8_up = \
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1, dilation=1)
        conv8_shortcut3 = 
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1)

        # input shape: (-1, 256, H/4, W/4)
        conv8 = nn.Sequential([
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(),
            nn.BatchNorm2d(256)])
        # output shape: (-1, 256, H/4, W/4)

        # conv9

        # input shape: (-1, 256, H/4, W/4)
        conv9_up = \
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, dilation=1)
        conv9_shortcut2 = 
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1)

        # input shape: (-1, 128, H/2, W/2)
        conv9 = nn.Sequential([
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(),
            nn.BatchNorm2d(128)])
        # output shape: (-1, 128, H/2, W/2)

        # conv10

        # input shape: (-1, 128, H/2, W/2)
        conv10_up = \
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, dilation=1)
        conv10_shortcut2 = 
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1)

        # input shape: (-1, 64, H, W)
        conv10 = nn.Sequential([
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1),
            # nn.ReLU(),
            nn.BatchNorm2d(64)])
        # output shape: (-1, 64, H, W)

        # main_out
        main_out = nn.Sequential([
            nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1, stride=1, padding=0, dilation=1),
            nn.TanH()])


        # Local Hints Network

        # hint generators
        hint3 = \
            nn.Conv2D(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1, dilation=1)
        hint4 = \
            nn.ConvTranspose2D(in_channels=512, out_channels=384, kernel_size=4, stride=2, padding=1, dilation=1)
        hint5 = \
            nn.ConvTranspose2D(in_channels=512, out_channels=384, kernel_size=4, stride=2, padding=1, dilation=1)
        hint6 = \
            nn.ConvTranspose2D(in_channels=512, out_channels=384, kernel_size=4, stride=2, padding=1, dilation=1)
        hint7 = \
            nn.ConvTranspose2D(in_channels=512, out_channels=384, kernel_size=4, stride=2, padding=1, dilation=1)
        hint8 = \
            nn.Conv2D(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1, dilation=1)
        
        # hint cummulator
        # sum the above results

        # input size: (-1, 384, H/4, W/4)
        hint_network = nn.Sequential([
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=313, kernel_size=1, stride=1, padding=0, dilation=1),
            nn.ConvTranspose2d(in_channels=313, out_channels=313, kernel_size=4, stride=2, padding=1, group=313, bias=False),
            nn.ConvTranspose2d(in_channels=313, out_channels=313, kernel_size=4, stride=2, padding=1, group=313, bias=False),
            #nn.Mul(),
            nn.Softmax()])



    def forward(self, grayscale_image, global_hints, local_hints, local_hints_mask):
        pass
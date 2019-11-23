import torch
import torch.nn as nn


class ColorizationModel(nn.Module):

    def __init__(self):
        super(ColorizationModel, self).__init__()

        # Trainable layers

        # Global Hints Network

        # input shape: (-1, 316, 1, 1)
        self.global_conv = nn.Sequential(
            nn.Conv2d(in_channels=316, out_channels=512, kernel_size=1, stride=1, padding=0, dilation=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0, dilation=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0, dilation=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0, dilation=1),
            nn.ReLU())
        # output shape: (-1, 512, 1, 1)


        # Main network

        # conv1

        # input shape: (-1, 4, H, W)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(),
            nn.BatchNorm2d(64))
        # output shape: (-1, 64, H, W)

        # conv2

        # input shape: (-1, 64, H, W)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(),
            nn.BatchNorm2d(128))
        # output shape: (-1, 128, H/2, W/2)

        # conv3

        # input shape: (-1, 128, H/2, W/2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(),
            nn.BatchNorm2d(256))
        # output shape: (-1, 256, H/4, W/4)

        # conv4

        # input shape: (-1, 256, H/4, W/4)
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(),
            nn.BatchNorm2d(512))
        # output shape: (-1, 512, H/8, W/8)

        # conv5

        # input shape: (-1, 512, H/8, W/8)
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(),
            nn.BatchNorm2d(512))
        # output shape: (-1, 512, H/8, W/8)

        # conv6

        # input shape: (-1, 512, H/8, W/8)
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(),
            nn.BatchNorm2d(512))
        # output shape: (-1, 512, H/8, W/8)

        # conv7

        # input shape: (-1, 512, H/8, W/8)
        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(),
            nn.BatchNorm2d(512))
        # output shape: (-1, 512, H/8, W/8)

        # conv8

        # input shape: (-1, 512, H/8, W/8)
        self.conv8_up = \
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1, dilation=1)
        self.conv8_shortcut3 = \
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1)

        # input shape: (-1, 256, H/4, W/4)
        self.conv8 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(),
            nn.BatchNorm2d(256))
        # output shape: (-1, 256, H/4, W/4)

        # conv9

        # input shape: (-1, 256, H/4, W/4)
        self.conv9_up = \
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, dilation=1)
        self.conv9_shortcut2 = \
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1)

        # input shape: (-1, 128, H/2, W/2)
        self.conv9 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(),
            nn.BatchNorm2d(128))
        # output shape: (-1, 128, H/2, W/2)

        # conv10

        # input shape: (-1, 128, H/2, W/2)
        self.conv10_up = \
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, dilation=1)
        self.conv10_shortcut1 = \
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1)

        # input shape: (-1, 64, H, W)
        self.conv10 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(64))
        # output shape: (-1, 64, H, W)

        # main_out
        self.main_out = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1, stride=1, padding=0, dilation=1),
            nn.Tanh())


        # Local Hints Network

        # hint generators
        self.hint3 = \
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1, dilation=1)
        self.hint4 = \
            nn.ConvTranspose2d(in_channels=512, out_channels=384, kernel_size=4, stride=2, padding=1, dilation=1)
        self.hint5 = \
            nn.ConvTranspose2d(in_channels=512, out_channels=384, kernel_size=4, stride=2, padding=1, dilation=1)
        self.hint6 = \
            nn.ConvTranspose2d(in_channels=512, out_channels=384, kernel_size=4, stride=2, padding=1, dilation=1)
        self.hint7 = \
            nn.ConvTranspose2d(in_channels=512, out_channels=384, kernel_size=4, stride=2, padding=1, dilation=1)
        self.hint8 = \
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1, dilation=1)
        
        # hint cummulator
        # sum the above results

        # input shape: (-1, 384, H/4, W/4)
        self.hint_network = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=313, kernel_size=1, stride=1, padding=0, dilation=1),
            nn.ConvTranspose2d(in_channels=313, out_channels=313, kernel_size=4, stride=2, padding=1, groups=313, bias=False),
            nn.ConvTranspose2d(in_channels=313, out_channels=313, kernel_size=4, stride=2, padding=1, groups=313, bias=False),
            #nn.Mul(),
            nn.Softmax())
        # output shape: (-1, 313, H, W)


    def forward(self, grayscale_image, global_hints, local_hints, local_hints_mask):
        """
        Parameters:
        grayscale_image :: (-1, 1, H, W)
        global_hints :: (-1, 316, 1, 1)
        local_hints :: (-1, 2, H, W)
        local_hints_mask :: (-1, 1, H, W)

        Returns:
        main_output :: (-1, 2, H, W)
        hint_output :: (-1, 313, H, W)
        """

        # Add dimension for batch if single image given
        for inp in (grayscale_image, global_hints, local_hints, local_hints_mask):
            if len(inp.size()) == 3:
                inp.unsqueeze_(0)

        # Check input dims are correct
        batch_size, _, height, width = grayscale_image.size()[2:]
        assert grayscale_image.size() == (batch_size, 1, height, width)
        assert global_hints.size() == (batch_size, 316, 1, 1)
        assert local_hints.size() == (batch_size, 2, height, width)
        assert local_hints_mask.size() == (batch_size, 1, height, width)

        # Global pass
        global_output = self.global_conv(global_hints)

        # Main pass
        image_plus_hints = torch.cat((grayscale_image, local_hints, local_hints_mask), 1)

        conv1 = self.conv1(image_plus_hints)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv4_plus_global = torch.add(conv4, global_output)
        conv5 = self.conv5(conv4_plus_global)
        conv6 = self.conv6(conv5)
        conv7 = self.conv7(conv6)

        conv8_up = self.conv8_up(conv7)
        conv8_short3 = self.conv8_shortcut3(conv3)
        conv8_in = torch.add(conv8_up, conv8_short3, alpha=0.5)
        conv8 = self.conv8(conv8_in)

        conv9_up = self.conv9_up(conv8)
        conv9_short2 = self.conv9_shortcut2(conv2)
        conv9_in = torch.add(conv9_up, conv9_short2, alpha=0.5)
        conv9 = self.conv9(conv9_in)

        conv10_up = self.conv10_up(conv9)
        conv10_short1 = self.conv10_shortcut1(conv1)
        conv10_in = torch.add(conv10_up, conv10_short1, alpha=0.5)
        conv10 = self.conv10(conv10_in)

        main_output = self.main_out(conv10)

        # Local pass
        hint3 = self.hint3(conv3)
        hint4 = self.hint3(conv4)
        hint5 = self.hint3(conv5)
        hint6 = self.hint3(conv6)
        hint7 = self.hint3(conv7)
        hint8 = self.hint3(conv8)
        hint_total = torch.sum(torch.stack([
            hint3, hint4, hint5, hint6, hint7, hint8]), dim=0)

        hint_output = self.hint_network(hint_total)

        assert main_output.size() == (batch_size, 2, height, width)
        assert hint_output.size() == (batch_size, 313, height, width)

        return main_output, hint_output


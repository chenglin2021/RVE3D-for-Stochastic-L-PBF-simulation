from torch.utils.data import Dataset, ConcatDataset, DataLoader
import torch
import os
import torch.nn as nn
import torch.nn.functional as F


class WeightedMSELoss(nn.Module):
    def __init__(self, weights):
        super(WeightedMSELoss, self).__init__()
        self.weights = weights

    def forward(self, inputs, targets):
        return ((F.mse_loss(inputs, targets, reduction='none') * self.weights.view(1, -1, 1, 1, 1)).mean())


class ScalingLayer(nn.Module):
    def __init__(self, scaling_factors):
        super().__init__()
        self.scaling_factors = nn.Parameter(scaling_factors.clone().detach(), requires_grad=False)

    def forward(self, x):
        return x * self.scaling_factors.view(1, -1, 1, 1, 1)


class SdataScalingLayer(nn.Module):
    def __init__(self, scaling_factors):
        super().__init__()
        self.scaling_factors = nn.Parameter(scaling_factors.clone().detach(), requires_grad=False)

    def forward(self, x):
        # print(x.size())
        # print(self.scaling_factors.size())
        return (x * self.scaling_factors).view(1, -1)


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True)
    )


class ResidualConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()
        '''
        standard ResNet style convolutional block
        '''
        self.same_channels = in_channels==out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )


# This embedding is for the in-situ information from the cameras
class EmbedConv2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EmbedConv2D, self).__init__()
        '''
        process and upscale the image feature maps
        '''
        layers = [
            nn.Conv2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)
        return x


class EmbedConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EmbedConv3D, self).__init__()
        '''
        process and upscale the image feature maps
        '''
        layers = [
            double_conv(in_channels, out_channels),
            nn.MaxPool3d(2),
            double_conv(out_channels, out_channels),
            # nn.MaxPool3d(2),
            # double_conv(out_channels, out_channels),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)
        return x


# This embedding is for the laser power from the photodiode, like laser power and location
class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        '''
        generic one layer FC NN for embedding things  
        '''
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)


class MyDataset(Dataset):
    def __init__(self, dir_path):
        self.dir_path = dir_path
        self.num_examples = len([name for name in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, name))])

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        RVE_start, RVE_energy, RVE_end = torch.load(f'{self.dir_path}/example_{idx}.pt')
        input = torch.cat((RVE_start, RVE_energy.unsqueeze(0)), dim=0)
        output = RVE_end

        desired_shape = (148, 28, 6)  # Desired depth, height, and width

        # # Pad input
        # for idx, dim in enumerate(desired_shape):
        #     if input.shape[idx+1] < dim:  # Add 1 to idx to skip the channel dimension
        #         pad = [0, 0]*len(input.shape)  # Initialize the padding
        #         pad[2*(idx+1)] = dim - input.shape[idx+1]  # Add padding to 'before' part of dimension
        #         input = F.pad(input, pad=pad)

        # # Pad output
        # for idx, dim in enumerate(desired_shape):
        #     if output.shape[idx+1] < dim:  # Add 1 to idx to skip the channel dimension
        #         pad = [0, 0]*len(output.shape)  # Initialize the padding
        #         pad[2*(idx+1)] = dim - output.shape[idx+1]  # Add padding to 'before' part of dimension
        #         output = F.pad(output, pad=pad)

        return input.float(), output.float()


class MyDatasetRAM(Dataset):
    def __init__(self, dir_path):
        self.dir_path = dir_path
        file_names = [name for name in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, name))]
        self.data = []
        self.desired_shape = (148, 28, 6)  # Desired depth, height, and width

        for name in file_names:
            RVE_start, RVE_Q, RVE_phi, RVE_end, RVE_surf_end, RVE_sensordata = torch.load(os.path.join(dir_path, name))
            input = torch.cat((RVE_start, RVE_Q.unsqueeze(0)), dim=0)
            input = torch.cat((input, RVE_phi.unsqueeze(0)), dim=0)
            camera_embed = RVE_surf_end
            sensor_embed = RVE_sensordata
            output = RVE_end

            self.data.append((input.float(), camera_embed.float(), sensor_embed.float(), output.float()))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet3D, self).__init__()

        def block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 3, padding=0),  # Remove padding
                nn.ReLU(),
                nn.Conv3d(out_channels, out_channels, 3, padding=0),  # Remove padding
                nn.ReLU()
            )

        self.enc1 = block(in_channels, 16)
        self.enc2 = block(16, 32)
        self.enc3 = block(32, 64)

        self.pool = nn.MaxPool3d(2)

        self.dec2 = block(64 + 32, 32)
        self.dec1 = block(32 + 16, 16)

        self.out = nn.Conv3d(16, out_channels, 1)

    def forward(self, x):
        def pad_to_even(tensor):
            # If shape is odd, add padding to make it even
            shape = list(tensor.shape[2:])
            padding = [0, 0, 0]
            for i, dim in enumerate(shape):
                if dim % 2 != 0:
                    padding[i] = 1
            return F.pad(tensor, (padding[0], padding[0], padding[1], padding[1], padding[2], padding[2]))

        x = pad_to_even(x)

        enc1 = self.enc1(x)
        enc1_p = self.pool(pad_to_even(enc1))

        enc2 = self.enc2(enc1_p)
        enc2_p = self.pool(pad_to_even(enc2))

        enc3 = self.enc3(enc2_p)

        up3 = F.interpolate(enc3, size=enc2.size()[2:], mode='trilinear', align_corners=True)
        dec2 = self.dec2(pad_to_even(torch.cat([up3, enc2], dim=1)))

        up2 = F.interpolate(dec2, size=enc1.size()[2:], mode='trilinear', align_corners=True)
        dec1 = self.dec1(pad_to_even(torch.cat([up2, enc1], dim=1)))

        return self.out(dec1)


class RVE3D(nn.Module):
    def __init__(self, in_channels, out_channels, input_scaling_factors, output_scaling_factors):
        super().__init__()

        self.input_scaling = ScalingLayer(input_scaling_factors)

        self.dconv_down1 = double_conv(in_channels, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)

        self.maxpool = nn.MaxPool3d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(64 + 128, 64)

        self.conv_last = nn.Conv3d(64, out_channels, 1)

        self.output_scaling = ScalingLayer(output_scaling_factors)

    def forward(self, x):
        x = self.input_scaling(x)

        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        x = self.dconv_down3(x)

        x = self.upsample(x)
        x = F.interpolate(x, size=conv2.size()[2:], mode='trilinear', align_corners=True)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = F.interpolate(x, size=conv1.size()[2:], mode='trilinear', align_corners=True)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        out = self.conv_last(x)
        # Apply sigmoid function to first two channels
        # out[:, 0:2, :, :, :] = torch.sigmoid(out[:, 0:2, :, :, :])
        # Apply hard sigmoid function to first two channels
        # out[:, 0:2, :, :, :] = torch.clamp(out[:, 0:2, :, :, :] + 1, 0, 2) / 2

        return self.output_scaling(out)


class ConRVE3D(nn.Module):
    def __init__(self, in_channels, out_channels, n_sensors, n_feats, emb_dim, input_scaling_factors,
                 output_scaling_factors, sdata_scaling_factors, cdata_scaling_factors):
        super().__init__()

        self.input_scaling = ScalingLayer(input_scaling_factors)
        self.n_sensors = n_sensors
        self.n_feats = n_feats
        self.emb_dim = emb_dim

        self.dconv_down1 = double_conv(in_channels, emb_dim)
        self.dconv_down2 = double_conv(emb_dim, emb_dim*2)
        self.dconv_down3 = double_conv(emb_dim*2, emb_dim*4)

        self.maxpool = nn.MaxPool3d(2)

        self.sdata_emb_scaling = SdataScalingLayer(sdata_scaling_factors)
        self.cdata_emb_scaling = ScalingLayer(cdata_scaling_factors)

        self.imageembed1 = EmbedConv3D(2, emb_dim*2+emb_dim*4)
        # self.imageembed2 = EmbedConv3D(2, 192)
        self.sensorembed1 = EmbedFC(n_sensors, emb_dim*2+emb_dim*4)
        # self.sensorembed2 = EmbedFC(n_sensors, 192)
        # self.sensorembed1 = nn.Embedding(n_points, self.emb_dim)
        # self.sensorembed2 = nn.Embedding(n_points, self.emb_dim*2)

        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        self.dconv_up2 = double_conv(emb_dim*2+emb_dim*4, emb_dim*2)
        self.dconv_up1 = double_conv(emb_dim+emb_dim*2, emb_dim)

        self.conv_last = nn.Conv3d(emb_dim, out_channels, 1)

        self.output_scaling = ScalingLayer(output_scaling_factors)

    def forward(self, x, cdata, sdata):
        # x is the process parameter
        # cdata is the surface information from IR camera, video camera, etc.
        # sdata is the sensor data from photodiodes, thermal couple, etc.

        x = self.input_scaling(x)

        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        x = self.dconv_down3(x)

        x = self.upsample(x)
        x = F.interpolate(x, size=conv2.size()[2:], mode='trilinear', align_corners=True)
        x = torch.cat([x, conv2], dim=1)

        # embed sensor data to the latent space
        sdata = self.sdata_emb_scaling(sdata)
        sensorembed1 = self.sensorembed1(sdata).view(-1, self.emb_dim*2+self.emb_dim*4, 1, 1, 1)
        # sensorembed2 = self.sensorembed2(sdata).view(-1, 192, 1, 1, 1)

        # # embed camera data to the latent space
        cdata = self.cdata_emb_scaling(cdata)
        imageembed1 = self.imageembed1(cdata)
        # imageembed2 = self.imageembed2(cdata)

        x = self.dconv_up2(x*sensorembed1 + imageembed1)
        # x = self.dconv_up2(x * sensorembed1)
        x = self.upsample(x)
        x = F.interpolate(x, size=conv1.size()[2:], mode='trilinear', align_corners=True)
        x = torch.cat([x, conv1], dim=1)

        # x = self.dconv_up1(x*sensorembed2 + imageembed2)
        x = self.dconv_up1(x)
        # x = self.dconv_up1(x * sensorembed2)

        out = self.conv_last(x)
        # Apply sigmoid function to first two channels
        # out[:, 0:2, :, :, :] = torch.sigmoid(out[:, 0:2, :, :, :])
        # Apply hard sigmoid function to first two channels
        # out[:, 0:2, :, :, :] = torch.clamp(out[:, 0:2, :, :, :] + 1, 0, 2) / 2

        return self.output_scaling(out)
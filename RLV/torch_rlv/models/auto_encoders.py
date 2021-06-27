import torch
import torch.nn as nn
from torch.nn.functional import softmax


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(5, 5), stride=(3, 3)),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=(5, 5), stride=(3, 3)),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=(5, 5), stride=(3, 3)),
            nn.ReLU(True))

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=(5, 5), stride=(3, 3), padding='same'),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=(5, 5), stride=(2, 2), padding='same'),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, kernel_size=(5, 5), stride=(1, 1), padding='same'))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class CoordinateUtils(object):
    @staticmethod
    def get_image_coordinates(h, w, normalise):
        x_range = torch.arange(w, dtype=torch.float32)
        y_range = torch.arange(h, dtype=torch.float32)
        if normalise:
            x_range = (x_range / (w - 1)) * 2 - 1
            y_range = (y_range / (h - 1)) * 2 - 1
        image_x = x_range.unsqueeze(0).repeat_interleave(h, 0)
        image_y = y_range.unsqueeze(0).repeat_interleave(w, 0).t()
        return image_x, image_y


class SpatialSoftArgmax(nn.Module):
    def __init__(self, temperature=None, normalise=False):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1)) if temperature is None else torch.tensor([temperature])
        self.normalise = normalise

    def forward(self, x):
        n, c, h, w = x.size()
        spatial_softmax_per_map = softmax(x.view(n * c, h * w) / self.temperature, dim=1)
        spatial_softmax = spatial_softmax_per_map.view(n, c, h, w)

        image_x, image_y = CoordinateUtils.get_image_coordinates(h, w, normalise=self.normalise)
        # size (H, W, 2)
        image_coordinates = torch.cat((image_x.unsqueeze(-1), image_y.unsqueeze(-1)), dim=-1)
        image_coordinates = image_coordinates.to(device=x.device)
        expanded_spatial_softmax = spatial_softmax.unsqueeze(-1)
        image_coordinates = image_coordinates.unsqueeze(0)
        out = torch.sum(expanded_spatial_softmax * image_coordinates, dim=[2, 3])
        # (N, C, 2)
        return out


class DSEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, temperature=None, normalise=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels[0], kernel_size=(7, 7), stride=(2, 2))
        self.batch_norm1 = nn.BatchNorm2d(out_channels[0])
        self.conv2 = nn.Conv2d(in_channels=out_channels[0], out_channels=out_channels[1], kernel_size=(5, 5))
        self.batch_norm2 = nn.BatchNorm2d(out_channels[1])
        self.conv3 = nn.Conv2d(in_channels=out_channels[1], out_channels=out_channels[2], kernel_size=(5, 5))
        self.batch_norm3 = nn.BatchNorm2d(out_channels[2])
        self.activ = nn.ReLU()
        self.spatial_soft_argmax = SpatialSoftArgmax(temperature=temperature, normalise=normalise)

    def forward(self, x):
        out_conv1 = self.activ(self.batch_norm1(self.conv1(x)))
        out_conv2 = self.activ(self.batch_norm2(self.conv2(out_conv1)))
        out_conv3 = self.activ(self.batch_norm3(self.conv3(out_conv2)))
        out = self.spatial_soft_argmax(out_conv3)
        return out


class DSDecoder(nn.Module):
    def __init__(self, image_output_size, latent_dimension, normalise=True):
        super().__init__()
        self.height, self.width = image_output_size
        self.latent_dimension = latent_dimension
        self.decoder = nn.Linear(in_features=latent_dimension, out_features=self.height * self.width)
        self.activ = nn.Tanh() if normalise else nn.Sigmoid()

    def forward(self, x):
        out = self.activ(self.decoder(x))
        out = out.view(-1, 1, self.height, self.width)
        return out


class DeepSpatialAutoEncoder(nn.Module):
    def __init__(self, image_output_size=(60, 60), in_channels=3, out_channels=(64, 32, 16), latent_dimension=32,
                 temperature=None, normalise=False):
        super().__init__()
        if out_channels[-1] * 2 != latent_dimension:
            raise ValueError("Spatial SoftArgmax produces a location (x,y) per feature map!")
        self.encoder = DSEncoder(in_channels=in_channels, out_channels=out_channels, temperature=temperature,
                                 normalise=normalise)
        self.decoder = DSDecoder(image_output_size=image_output_size, latent_dimension=latent_dimension)

    def forward(self, x):
        # (N, C, 2)
        spatial_features = self.encoder(x)
        n, c, _2 = spatial_features.size()
        # (N, C * 2 = latent dimension)
        return self.decoder(spatial_features.view(n, c * 2))


class DSAELoss(object):
    def __init__(self, add_g_slow=True):
        self.add_g_slow = add_g_slow
        self.mse_loss = nn.MSELoss(reduction="sum")

    def __call__(self, reconstructed, target, ft_minus1=None, ft=None, ft_plus1=None):
        loss = self.mse_loss(reconstructed, target)
        g_slow_contrib = torch.zeros(1, device=loss.device)
        if self.add_g_slow:
            g_slow_contrib = self.mse_loss(ft_plus1 - ft, ft - ft_minus1)
        return loss, g_slow_contrib

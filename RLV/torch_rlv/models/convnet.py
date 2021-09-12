import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T
import numpy as np
from RLV.torch_rlv.data.visual_pusher_data.adapter_visual_pusher import AdapterVisualPusher


class ConvNet(nn.Module):
    """
    Image classifier using convolutional layers with max pooling.
    """
    def __init__(self, output_dims=20):
        """
        Model Constructor, Initialize all the layers to be used
        """
        super(ConvNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, 5)
        self.conv2 = nn.Conv2d(16, 16, 5)
        self.conv3 = nn.Conv2d(16, 32, 5)
        self.max_pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 11 * 11, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dims)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        """
        This function defines the forward pass of this net model.
        Once this function is defined, the gradient back-propagation can be
        automatically computed by PyTorch.

        :param x: input data of this model
        :return: output data of this model
        """
        x = F.relu(self.conv1(x))
        x = self.max_pool(x)

        x = F.relu(self.conv2(x))
        x = self.max_pool(x)

        x = F.relu(self.conv3(x))
        x = self.max_pool(x)
       
        x = T.flatten(x, 1)
        
        x = self.fc1(x)
        x = F.relu(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        
        x = self.fc3(x)
        return x

if __name__ == '__main__':
    conv_network = ConvNet(output_dims=20)
    a = AdapterVisualPusher()

    input = T.from_numpy(a.observation_img,).float()
    output = conv_network.forward(input)

    print(a.observation.shape)
    print(input.shape)
    print(output.shape)

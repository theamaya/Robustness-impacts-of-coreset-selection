import torch.nn as nn
import torch.nn.functional as F
from torch import set_grad_enabled
from .nets_utils import EmbeddingRecorder

# Acknowledgement to
# https://github.com/kuangliu/pytorch-cifar,
# https://github.com/BIGBALLON/CIFAR-ZOO,


''' MLP '''


class CNN(nn.Module):
    def __init__(self, channel, num_classes, im_size, record_embedding: bool = False, no_grad: bool = False,
                 pretrained: bool = False, no_dropout: bool= True):
        if pretrained:
            raise NotImplementedError("torchvison pretrained models not available.")
        super(MLP, self).__init__()
        self.conv1 = nn.Conv2d(channel, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

        self.no_dropout= no_dropout

        self.embedding_recorder = EmbeddingRecorder(record_embedding)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))

        if self.no_dropout:
            x = F.relu(F.max_pool2d(self.conv2(x), 2))
        else:
            x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))

        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))

        if not self.no_dropout:
            x = F.dropout(x, training=self.training)

        x = self.embedding_recorder(x)
        x = self.fc2(x)
        return x #F.log_softmax(x, dim=1)

    def get_last_layer(self):
        return self.fc2



import torch
import torch.nn as nn
import torch.nn.functional as F

#4a
dropout_1 = 0.9
dropout_2 = 0.9
class ResBlock(nn.Module):

    def __init__(self, in_size:int, out_size:int, kernel_size:int, groups=1):

        super().__init__()

        # convolutional layers

        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size, padding="same", groups=groups)

        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size, padding="same", groups=groups)

        self.skip_connection = nn.Identity() if in_size==out_size else nn.Conv2d(in_size, out_size, kernel_size=1, padding="same")

        # normalization layers

        self.norm1 = nn.BatchNorm2d(in_size)

        self.norm2 = nn.BatchNorm2d(out_size)

        # activation functions

        self.act1 = nn.ReLU()

        self.act2 = nn.ReLU()

        


    def forward(self, x: torch.Tensor)->torch.Tensor:

        h = self.conv1(self.act1(self.norm1(x)))

        h = self.conv2(self.act2(self.norm2(h)))

        x = self.skip_connection(x)

        return x + h



class Embed(nn.Module):

    def __init__(self, embedding_dim:int, dropout1:float=dropout_1, dropout2:float=dropout_2):

        super(Embed, self).__init__()

        self.conv = nn.Conv2d(1, 32, kernel_size=5, padding="same")

        # res blocks

        self.block1 = ResBlock(32, 32, 3, groups=2)

        self.block2 = ResBlock(32, 32, 3, groups=2)

        self.block3 = ResBlock(32, 64, 3, groups=4)

        self.block4 = ResBlock(64, 64, 3, groups=4)

        # pooling layer

        self.pool = nn.MaxPool2d(2)

        # normalization layers

        self.norm1 = nn.BatchNorm2d(32)

        self.norm2 = nn.BatchNorm2d(32)

        self.norm3 = nn.BatchNorm1d(64)

        # linear layers

        self.fc1   = nn.Linear(64, 128)

        self.fc2   = nn.Linear(128, embedding_dim)

        # dropout

        self.dropout1 = nn.Dropout(dropout1)

        self.dropout2 = nn.Dropout(dropout2)



    def forward(self, x:torch.Tensor)->torch.Tensor:

        out = F.relu(self.norm1(self.conv(x)))   # convolution to prepare for res blocks

        # out.shape = ? (tensor shape 1)

        out = self.block2(self.block1(out))  # first set of res blocks

        # out.shape = ? (tensor shape 2)

        out = self.norm2(self.pool(out))  # pooling

        # out.shape = ? (tensor shape 3)

        out = self.block4(self.block3(out))  # second set of res blocks

        # out.shape = ? (tensor shape 4)

        out = torch.mean(out, dim=(-1, -2))  # average over spatial dimensions

        out = self.norm3(out)  # batch norm before fully connected part

        # out.shape = ? (tensor shape 5)

        

        # fully connected part:

        out = self.dropout1(out)

        out = self.fc1(out)

        out = F.relu(out)

        out = self.dropout2(out)

        out = self.fc2(out)

        

        return out
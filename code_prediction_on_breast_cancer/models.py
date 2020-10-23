import torch
from torch import nn
import torch.nn.functional as F


class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self, in_dim):
        super(Self_Attn, self).__init__()
        self.query_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.ones(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( 1 X C X N)
            returns :
                out : self attention value + input feature
                attention:  1 X N X N  (N is number of patches)
        """
        proj_query = self.query_conv(x).permute(0, 2, 1)  # 1 X N X C
        proj_key = self.key_conv(x)  # 1 X C x N
        energy = torch.bmm(proj_query, proj_key)  # 1 X N X N
        attention = self.softmax(energy)  # 1 X N X N

        out = torch.bmm(x, attention.permute(0, 2, 1))
        out = self.gamma * out + x

        return out, attention


class AttnClassifier(nn.Module):
    def __init__(self, in_features, num_classes):
        super(AttnClassifier, self).__init__()

        self.fc1 = nn.Linear(in_features, 512)
        self.fc2 = nn.Linear(512, 128)
        self.attn = Self_Attn(128)
        self.fc3 = nn.Linear(128, num_classes)

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x, attn = self.attn(x.permute(1, 2, 0))
        x2 = x.mean(dim=-1)
        x = self.fc3(x2)

        return x, attn

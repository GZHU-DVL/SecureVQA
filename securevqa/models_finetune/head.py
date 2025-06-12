import torch.nn as nn
import torch


class Bottleneck(nn.Module):
    """Bottleneck of the Fusion model.
    Args:
        in_channels: input channels for MLP
        hidden_channels: hidden channels for MLP
    """
    def __init__(self, inp, oup, expand_ratio):
        super(Bottleneck, self).__init__()
        hidden_dim = int(inp * expand_ratio)
        self.bottleneckBlock = nn.Sequential(
            # pw
            nn.Conv3d(inp, hidden_dim, 1, bias=False),
            nn.ReLU6(inplace=True),
            nn.Conv3d(hidden_dim, hidden_dim, 1, groups=hidden_dim, bias=False),
            nn.ReLU6(inplace=True),
            nn.Conv3d(hidden_dim, oup, 1, bias=False),
        )
    def forward(self, x):
        return self.bottleneckBlock(x)

class Embeddings_fusion(nn.Module):
    def __init__(self):
        super(Embeddings_fusion, self).__init__()
        self.Inter = Bottleneck(inp=768, oup=256, expand_ratio=2)
        self.Intra = Bottleneck(inp=256, oup=768, expand_ratio=2)
        self.shffleconv = nn.Conv3d(768 + 256, 768 + 256, kernel_size=1,
                                    stride=1, padding=0, bias=True)

    def separateFeature(self, x):
        z1, z2 = x[:, :768], x[:, 768:]
        return z1, z2

    def forward(self, z1, z2):
        z1, z2 = self.separateFeature(
            self.shffleconv(torch.cat((z1, z2), dim=1)))   # E^{intra}, E^{inter}
        z3 = z1 + self.Intra(z2)       #  E^{intra}_{fuse}
        z4 = z2 + self.Inter(z1)       #  E^{inter}_{fuse}
        return z3, z4



class VQAHead(nn.Module):
    """MLP Regression Head for VQA.
    Args:
        in_channels: input channels for MLP
        hidden_channels: hidden channels for MLP
        dropout_ratio: the dropout ratio for features before the MLP (default 0.5)
    """

    def __init__(
            self, in_channels=768 + 256, hidden_channels=64, dropout_ratio=0.5, **kwargs
    ):
        super().__init__()
        self.dropout_ratio = dropout_ratio
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.fc_hid = nn.Conv3d(768 + 256, self.hidden_channels,
                                (1, 1, 1))  # The dimension of the intra branch i2 768, while the inter is 256
        self.fc_last = nn.Conv3d(self.hidden_channels, 1, (1, 1, 1))
        self.gelu = nn.GELU()

        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.Embeddings_fusion = Embeddings_fusion()

    def forward(self, Intra, Inter, rois=None):
        Inter = Inter.permute(0, 3, 2, 1).unsqueeze(4).to('cuda')

        Intra = Intra.mean((-2, -1), keepdims=True)
        Intra, Inter_frame = self.Embeddings_fusion(Intra, Inter)  # Fusing the embeddings of the two branches
        Intra = torch.cat((Intra, Inter_frame), dim=1)
        Intra = self.dropout(Intra)

        qlt_score = self.fc_last(self.dropout(self.gelu(self.fc_hid(Intra))))
        return qlt_score




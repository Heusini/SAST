import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from typing import Dict, Optional, Tuple


class SimpleRGBEncoder(nn.Module):
    def __init__(self, out_channels=256):
        super().__init__()
        self.rgb_encoder = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=4, stride=4, padding=0),  # [B, 64, H/2, W/2]
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
            # nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # [B, 128, H/4, W/4]
            # nn.BatchNorm2d(128),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(128, out_channels, kernel_size=3, stride=2, padding=1),  # [B, 256, H/8, W/8]
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True),
            nn.Linear(3*90*160, 96*160),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.rgb_encoder(x)  # [B, 256, H/8, W/8]

class Reduce(nn.Module):
    def __init__(self, in_channel, out_channel, patch_size):
        super().__init__()
        # self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=patch_size, stride=patch_size)
        groups = 32
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, patch_size, patch_size, bias=False),
            nn.GroupNorm(groups, out_channel),     # groups should divide embed_dim
            nn.GELU(),
        )

    def forward(self, input):
        return self.proj(input)

class FPN(nn.Module):
    def __init__(self, in_channel, out_channels = (64, 128, 256)):
        super().__init__()

        self.in_channel = in_channel
        self.out_channels = out_channels

        self.conv1 = nn.Sequential(nn.Conv2d(in_channel, out_channels[0], kernel_size=3, stride=2, padding=1),
                                   nn.ReLU(),
                                   nn.Conv2d(out_channels[0], out_channels[0], kernel_size=3, stride=2, padding=1),
                                   nn.ReLU(),
                                   )
        self.conv2 = nn.Conv2d(out_channels[0], out_channels[1], kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Sequential(nn.Conv2d(out_channels[1], out_channels[2], kernel_size=3, stride=2, padding=1),
                                   nn.ReLU())

    def forward(self, input):
        p1 = self.conv1(input)
        p2 = self.conv2(p1)
        p3 = self.conv3(p2)

        return [p1, p2, p3]



class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x



class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads)


    def forward(self, q_feat, k_feat, v_feat):
        B, C, H, W = q_feat.shape

        q = q_feat.view(B, C, -1).permute(2, 0, 1)
        print(f"{q.shape=}")
        kv = k_feat.view(B, C, -1).permute(2, 0, 1)
        print(f"{kv.shape=}")
        attn_out, _ = self.attn(q, kv, kv)
        return attn_out.permute(1, 2, 0).view(B, C, H, W)

    

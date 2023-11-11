from __future__ import division, absolute_import
import warnings
import torch
from torch import nn
from torch.nn import functional as F
from torchreid.models.osnet import OSNet, ConvLayer, Conv1x1, OSBlock, init_pretrained_weights


class TransformerBlock(nn.Module):
    def __init__(self, feature_dim, num_heads, dim_feedforward=2048, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(feature_dim, num_heads, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(feature_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, feature_dim)

        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        # Self-attention
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # Feedforward
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src



class BagMultiheadAttention(nn.Module):
    def __init__(self, feature_dim, num_heads):
        super(BagMultiheadAttention, self).__init__()
        self.transformer_block = TransformerBlock(feature_dim, num_heads)

    def forward(self, bag_features):
        # bag_features shape: [B, I, K]
        transformed = self.transformer_block(bag_features)
        # Mean pooling across instances
        bag_representations = transformed.mean(dim=1)  # Shape: [B, K]
        return bag_representations


class OSNetMil(OSNet):
    """Omni-Scale Network.
    
    Reference:
        - Zhou et al. Omni-Scale Feature Learning for Person Re-Identification. ICCV, 2019.
        - Zhou et al. Learning Generalisable Omni-Scale Representations
          for Person Re-Identification. TPAMI, 2021.
    """

    def __init__(
        self,
        num_classes,
        blocks,
        layers,
        channels,
        num_heads=4,
        batch_size=1,
        bag_size=1,
        **kwargs
    ):
        super(OSNetMil, self).__init__(num_classes, blocks, layers, channels, **kwargs)
        self.single_batch_attn_size = self.feature_dim
        self.attention = BagMultiheadAttention(self.single_batch_attn_size, num_heads=num_heads)
        self.bag_classifier = nn.Linear(self.single_batch_attn_size, num_classes)

    def bag_attention(self, bags_of_instances):
        bag_features = self.attention(bags_of_instances)
        bag_predictions = self.bag_classifier(bag_features)
        return bag_predictions, bag_features


def osnet_x1_0_mil(num_classes=1000, pretrained=True, loss='softmax', **kwargs):
    # standard size (width x1.0)
    model = OSNetMil(
        num_classes,
        blocks=[OSBlock, OSBlock, OSBlock],
        layers=[2, 2, 2],
        channels=[64, 256, 384, 512],
        loss=loss,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, key='osnet_x1_0')
    return model

from __future__ import division, absolute_import
import warnings
import torch
from torch import nn
from torch.nn import functional as F
from torchreid.models.osnet import OSNet, ConvLayer, Conv1x1, OSBlock, init_pretrained_weights
import math

# class TransformerBlock(nn.Module):
#     def __init__(self, feature_dim, num_heads, dim_feedforward=2048, dropout=0.1):
#         super(TransformerBlock, self).__init__()
#         self.self_attn = nn.MultiheadAttention(feature_dim, num_heads, dropout=dropout, batch_first=True)
#         # Implementation of Feedforward model
#         self.linear1 = nn.Linear(feature_dim, dim_feedforward)
#         self.dropout = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(dim_feedforward, feature_dim)
#
#         self.norm1 = nn.LayerNorm(feature_dim)
#         self.norm2 = nn.LayerNorm(feature_dim)
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)
#
#         # Initialize a learnable <cls> token embedding
#         self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim))
#
#     def forward(self, src):
#         # Prepend the <cls> token to each sequence in the batch
#         batch_size = src.size(0)
#         cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # replicate <cls> token for each item in the batch
#         src = torch.cat((cls_tokens, src), dim=1)
#         # Self-attention
#         src2 = self.self_attn(src, src, src)[0]
#         src = src + self.dropout1(src2)
#         src = self.norm1(src)
#         # Feedforward
#         src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
#         src = src + self.dropout2(src2)
#         src = self.norm2(src)
#         return src


class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O

class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)

class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)


class DeepSet(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output, dim_hidden=128):
        super(DeepSet, self).__init__()
        self.num_outputs = num_outputs
        self.dim_output = dim_output
        self.enc = nn.Sequential(
                nn.Linear(dim_input, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden))
        self.dec = nn.Sequential(
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, num_outputs*dim_output))

    def forward(self, X):
        X = self.enc(X).mean(-2)
        X = self.dec(X).reshape(-1, self.num_outputs, self.dim_output)
        return X

class SetTransformer(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output,
            num_inds=32, dim_hidden=128, num_heads=4, ln=False):
        super(SetTransformer, self).__init__()
        self.enc = nn.Sequential(
                ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
                ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln))
        self.dec = nn.Sequential(
                PMA(dim_hidden, num_heads, num_outputs, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                nn.Linear(dim_hidden, dim_output))

    def forward(self, X):
        return self.dec(self.enc(X))


class BagMultiheadAttention(nn.Module):
    def __init__(self, feature_dim, num_heads):
        super(BagMultiheadAttention, self).__init__()
        self.transformer_block = SetTransformer(dim_input=feature_dim, num_outputs=1, dim_output=feature_dim,
                                                num_heads=num_heads)

    def forward(self, bag_features):
        # bag_features shape: [B, I, K]
        transformed = self.transformer_block(bag_features)
        # Todo: Do not use mean pooling - have a <cls> token or something and use it.
        # bag_representations = transformed.mean(dim=1)  # Shape: [B, K]
        # Extract <cls> token representation (assuming it's the first token)
        bag_representations = transformed[:, 0, :]  # Shape: [B, K]
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
        num_heads=8,
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
        # and also just get the average bag representation from the bags_of_instances
        # bag_features = bags_of_instances.mean(dim=1)
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

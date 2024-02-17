from __future__ import division, absolute_import
import torch
import torch.nn as nn


class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    
    Reference:
        Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    
    Imported from `<https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py>`_.
    
    Args:
        margin (float, optional): margin for triplet. Default is 0.3.
    """

    def __init__(self, margin=0.3, reduction='mean'):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin, reduction=reduction)

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape (num_classes).
        """
        n = inputs.size(0)

        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)

        dist = dist + dist.t()
        dist.addmm_(inputs, inputs.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt() # for numerical stability

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            try:
                dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
            except RuntimeError:
                # if there is no negative for this anchor, just add one set to the margin
                dist_an.append(torch.zeros_like(dist[i][mask[i]].max().unsqueeze(0)) + self.margin)
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)

        rl = self.ranking_loss(dist_an, dist_ap, y)

        return rl


class NoReductionTripletLoss(nn.Module):
    """Triplet loss with no reduction. All possible triplets are formed and returned.

    Args:
        margin (float, optional): margin for triplet. Default is 0.3.
    """

    def __init__(self, margin=0.3):
        super(NoReductionTripletLoss, self).__init__()
        self.margin = margin

    def forward(self, inputs, targets):
        n = inputs.size(0)

        # Compute pairwise distance
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(inputs, inputs.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()  # Numerical stability

        losses = []
        anchor_labels = []
        # this is gonna be slow af
        for anchor in range(n):
            for positive in range(n):
                if targets[anchor] != targets[positive]:
                    continue  # Skip if not the same class
                for negative in range(n):
                    if targets[anchor] == targets[negative]:
                        continue  # Skip if the same class
                    loss = torch.clamp(dist[anchor, positive] - dist[anchor, negative] + self.margin, min=1e-12, max=None)
                    losses.append(loss)
                    anchor_labels.append(targets[anchor])

        # print some statistics about the number of losses for each label. Print, min, max, avg, median, mode
        # num_losses_per_anchor = {}
        # for l, al in zip(losses, anchor_labels):
        #     al = al.item()
        #     if al not in num_losses_per_anchor:
        #         num_losses_per_anchor[al] = 0
        #     num_losses_per_anchor[al] += 1

        if losses:
            losses = torch.stack(losses)
            anchor_labels = torch.stack(anchor_labels)
        else:
            # If no valid triplet was found, return a zero loss and an empty anchor_labels tensor
            losses = torch.zeros(1, device=inputs.device)
            anchor_labels = torch.zeros(0, device=inputs.device, dtype=torch.long)

        return losses

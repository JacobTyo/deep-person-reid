import torch
import torch.nn as nn


class ConvexAccumulator(nn.Module):

    def __init__(self, batch_size, num_instances=4, normalization='softmax', update_frequency=1, batch_reduction='mean'):
        # num_instances is the number of instances per identity
        super(ConvexAccumulator, self).__init__()
        # for now, force the num_instances to be balanced in the batch:
        assert batch_size % num_instances == 0, 'batch size must be divisible by num_instances'
        self.num_triplets = int(num_instances * num_instances * (batch_size - num_instances)) // num_instances
        print(f'convex accumulator layer input size: {self.num_triplets}')
        self.batch_size = batch_size
        self.layer = nn.Linear(self.num_triplets, 1)
        self.normalization = normalization
        self.counter = 0
        self.update_freq = update_frequency
        # make convex combination
        self.normalize_weights()
        self.num_instances = num_instances
        self.batch_reduction = batch_reduction
        if self.batch_reduction == 'learned':
            print('using convex accumualtor for batch reduction as well')
            self.batch_reduction_layer = nn.Linear(batch_size, 1)


    def normalize_weights(self, force=False):
        # ensure this operation equates to a convex combination
        if self.normalization == 'softmax':
            return
        if ((self.counter % self.update_freq) == 0) or force:
            with torch.no_grad():
                if self.normalization == 'threshold':
                    tmp = self.layer.weight.detach()
                    tmp = tmp - torch.min(tmp)
                    tmp = tmp / tmp.sum()
                    self.layer.weight.data = tmp
                elif self.normalization == 'relu':
                    tmp = self.layer.weight.detach()
                    tmp = nn.functional.relu(tmp)
                    tmp = tmp / tmp.sum()
                    self.layer.weight.data = tmp
                else:
                    raise ValueError('the supported normalization methods are "reweight" and "softmax"')

    def get_normalized_weights(self):
        self.normalize_weights(True)
        weights = self.layer.weight.detach().cpu()
        if self.normalization == 'softmax':
            weights = nn.functional.softmax(weights, dim=1)
        return weights

    def forward(self, x, y):
        if self.training:
            self.normalize_weights()
            self.counter += 1

        # x is a loss value, y is the label for that loss
        # first, sort all x's by the y's
        xy = torch.vstack((x, y))
        xy = xy.sort(dim=1, descending=True).values
        # now we want all x's for a single y to be in a single row
        # first, drop the y's, then we can do this by reshaping the tensor
        x = xy[0, :]
        x = x.view(self.batch_size, -1)
        # now we want each row to be sorted
        x = x.sort(dim=1, descending=True).values.transpose(0, 1)
        # now we can apply the accumulator
        if self.normalization == 'softmax':
            x = nn.functional.softmax(self.layer.weight, dim=1) @ x
        else:
            x = self.layer(x)

        if self.batch_reduction == 'mean':
            x = x.mean()
        elif self.batch_reduction == 'learned':
            x = nn.functional.softmax(self.batch_reduction_layer.weight, dim=1) @ x.transpose(0, 1)

        return x

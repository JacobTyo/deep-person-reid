from __future__ import division, print_function, absolute_import

import torch
from torchreid import metrics
from torchreid.losses import TripletLoss, CrossEntropyLoss, BagInstanceDifferenceLoss

from ..engine import Engine


class ImageMilTripletEngine(Engine):
    r"""Triplet-loss engine for image-reid.

    Args:
        datamanager (DataManager): an instance of ``torchreid.data.ImageDataManager``
            or ``torchreid.data.VideoDataManager``.
        model (nn.Module): model instance.
        optimizer (Optimizer): an Optimizer.
        margin (float, optional): margin for triplet loss. Default is 0.3.
        weight_t (float, optional): weight for triplet loss. Default is 1.
        weight_x (float, optional): weight for softmax loss. Default is 1.
        scheduler (LRScheduler, optional): if None, no learning rate decay will be performed.
        use_gpu (bool, optional): use gpu. Default is True.
        label_smooth (bool, optional): use label smoothing regularizer. Default is True.

    Examples::

        import torchreid
        datamanager = torchreid.data.ImageDataManager(
            root='path/to/reid-data',
            sources='market1501',
            height=256,
            width=128,
            combineall=False,
            batch_size=32,
            num_instances=4,
            train_sampler='RandomIdentitySampler' # this is important
        )
        model = torchreid.models.build_model(
            name='resnet50',
            num_classes=datamanager.num_train_pids,
            loss='triplet'
        )
        model = model.cuda()
        optimizer = torchreid.optim.build_optimizer(
            model, optim='adam', lr=0.0003
        )
        scheduler = torchreid.optim.build_lr_scheduler(
            optimizer,
            lr_scheduler='single_step',
            stepsize=20
        )
        engine = torchreid.engine.ImageTripletEngine(
            datamanager, model, optimizer, margin=0.3,
            weight_t=0.7, weight_x=1, scheduler=scheduler
        )
        engine.run(
            max_epoch=60,
            save_dir='log/resnet50-triplet-market1501',
            print_freq=10
        )
    """

    def __init__(
            self,
            datamanager,
            model,
            optimizer,
            margin=0.3,
            weight_t=1,
            weight_x=1,
            weight_d=1,
            scheduler=None,
            use_gpu=True,
            label_smooth=True
    ):
        super(ImageMilTripletEngine, self).__init__(datamanager, use_gpu)

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.register_model('model', model, optimizer, scheduler)

        assert weight_t >= 0 and weight_x >= 0 and weight_d >= 0
        assert weight_t + weight_x + weight_d > 0
        self.weight_t = weight_t
        self.weight_x = weight_x
        self.weight_d = weight_d

        self.criterion_t = TripletLoss(margin=margin)
        self.criterion_x = CrossEntropyLoss(
            num_classes=self.datamanager.num_train_pids,
            use_gpu=self.use_gpu,
            label_smooth=label_smooth
        )
        self.criterion_d = BagInstanceDifferenceLoss(use_gpu=self.use_gpu)

    def forward_backward(self, data):
        imgs, pids = self.parse_data_for_train(data)

        if self.use_gpu:
            imgs = imgs.cuda()
            pids = pids.cuda()

        # for every instance, get the representation
        outputs, features = self.model(imgs)

        # Group features by pids (bag IDs)
        bag_features_dict = {}
        bag_pids = []
        for feature, pid in zip(features, pids):
            pid_key = pid.item()
            if pid_key not in bag_features_dict:
                bag_features_dict[pid_key] = []
                bag_pids.append(pid_key)
            bag_features_dict[pid_key].append(feature)

        # Convert bag_pids to a PyTorch tensor
        bag_pids = torch.tensor(bag_pids, dtype=torch.long, device=imgs.device)

        # Group the instances together by bag (i.e. pid), transform to tensor
        grouped_bags = torch.stack([torch.stack(bag_features_dict[pid], dim=0) for pid in bag_features_dict])

        # now, we need to pass each bag through the attention network to get the bag representation
        bag_predictions, bag_features = self.model.module.bag_attention(grouped_bags)

        # now, we need to compute the loss. We can give the pid of each bag and the bag features to the loss function
        loss = 0
        loss_summary = {}

        if self.weight_t > 0:
            loss_t = self.compute_loss(self.criterion_t, bag_features, bag_pids)
            loss += self.weight_t * loss_t
            loss_summary['loss_t'] = loss_t.item()

        if self.weight_x > 0:
            loss_x = self.compute_loss(self.criterion_x, bag_predictions, bag_pids)
            loss += self.weight_x * loss_x
            loss_summary['loss_x'] = loss_x.item()
            loss_summary['acc'] = metrics.accuracy(outputs, pids)[0].item()

        if self.weight_d > 0:
            loss_d = self.criterion_d(bag_features, grouped_bags)
            loss += self.weight_d * loss_d
            loss_summary['loss_d'] = loss_d.item()

        assert loss_summary

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss_summary

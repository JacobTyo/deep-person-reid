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
            label_smooth=True,
            bag_size=None
    ):
        super(ImageMilTripletEngine, self).__init__(datamanager, use_gpu)

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.bag_size = bag_size
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
            # so this is a list of tensors, each of shape [feature_dim]
            bag_features_dict[pid_key].append(feature)

        # now make sure each bag has the same number of instances = self.bag_size
        if self.bag_size is not None:
            for pid in bag_features_dict:
                # cast the list of tensors to a list of tensors each of shape [number_of_instances, feature_dim]
                # so shape is [num_bags_this_pid, number_of_instances, feature_dim]
                bag_features_dict[pid] = [torch.vstack(bag_features_dict[pid][x:x+self.bag_size])
                                          for x in range(0, len(bag_features_dict[pid]), self.bag_size)]

        # now get the bag_pid list and bag_features lists for input to attention
        grouped_bag_features = []
        grouped_bag_labels = []
        for pid, feature_list in bag_features_dict.items():
            for feat in feature_list:
                grouped_bag_features.append(feat)
                grouped_bag_labels.append(pid)

        # and cast to tensor
        grouped_bag_features = torch.stack(grouped_bag_features, dim=0)
        grouped_bag_labels = torch.tensor(grouped_bag_labels)

        # now, we need to pass each bag through the attention network to get the bag representation
        # the input should be of shape [number_of_bags, number_of_instances, feature_dim]
        bag_predictions, bag_features = self.model.module.bag_attention(grouped_bag_features)

        # now, we need to compute the loss. We can give the pid of each bag and the bag features to the loss function
        loss = 0
        loss_summary = {}

        # if self.weight_t > 0:
        loss_t = self.compute_loss(self.criterion_t, bag_features, grouped_bag_labels)
        loss += self.weight_t * loss_t
        loss_summary['loss_t'] = loss_t.item()

        # if self.weight_x > 0:
        loss_x = self.compute_loss(self.criterion_x, bag_predictions, grouped_bag_labels)
        loss += self.weight_x * loss_x
        loss_summary['loss_x'] = loss_x.item()
        loss_summary['acc'] = metrics.accuracy(outputs, pids)[0].item()

        # if self.weight_d > 0:
        loss_d = self.criterion_d(bag_features, grouped_bag_features)
        loss += self.weight_d * loss_d
        loss_summary['loss_d'] = loss_d.item()

        assert loss_summary

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss_summary

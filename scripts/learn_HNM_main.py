import sys
import time
import os.path as osp
import argparse
import torch
import torch.nn as nn



import torchreid
from torchreid.losses import TripletLoss, NoReductionTripletLoss
from torchreid.models import ConvexAccumulator, ConvexAccumulator_TrueMining
from torchreid.utils import (
    Logger, check_isfile, set_random_seed, collect_env_info,
    resume_from_checkpoint, load_pretrained_weights, compute_model_complexity
)

from default_config import (
    imagedata_kwargs, optimizer_kwargs, videodata_kwargs, engine_run_kwargs,
    get_default_config, lr_scheduler_kwargs
)


def build_datamanager(cfg):
    if cfg.data.type == 'image':
        return torchreid.data.ImageDataManager(**imagedata_kwargs(cfg))
    else:
        return torchreid.data.VideoDataManager(**videodata_kwargs(cfg))


def build_engine(cfg, datamanager, model, optimizer, scheduler):
    if cfg.data.type == 'image':
        if cfg.loss.name == 'softmax':
            engine = torchreid.engine.ImageSoftmaxEngine(
                datamanager,
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                use_gpu=cfg.use_gpu,
                label_smooth=cfg.loss.softmax.label_smooth
            )

        else:
            if 'mil' in cfg.data.sources[0] or 'sysu30k' in cfg.data.sources or 'mil' in cfg.model.name:
                print('-------------------------Using MIL Engine-------------------------')
                engine = torchreid.engine.ImageMilTripletEngine(
                    datamanager,
                    model,
                    optimizer=optimizer,
                    margin=cfg.loss.triplet.margin,
                    weight_t=cfg.loss.triplet.weight_t,
                    weight_x=cfg.loss.triplet.weight_x,
                    weight_d=cfg.loss.triplet.weight_d,
                    scheduler=scheduler,
                    use_gpu=cfg.use_gpu,
                    label_smooth=cfg.loss.softmax.label_smooth,
                    bag_size=cfg.train.bag_size,
                )
            elif cfg.model.learn_mining_fn:
                print('-------------------------Using Learned Mining Engine-------------------------')
                assert cfg.sampler.train_sampler == 'RandomIdentitySampler', f'The learned mining engine only supports the RandomIdentitySampler, got {cfg.sampler.train_sampler} instead'
                # accumulator_fn = ConvexAccumulator(batch_size=cfg.train.batch_size,
                #                                    num_instances=cfg.sampler.num_instances,
                #                                    batch_reduction=cfg.model.learn_mining_batch_reduction)
                accumulator_fn = ConvexAccumulator_TrueMining(batch_size=cfg.train.batch_size,
                                                              num_instances=cfg.sampler.num_instances,
                                                              batch_reduction=cfg.model.learn_mining_batch_reduction)
                accumulator_fn.to(torch.device('cuda'))
                accumulator_optimizer = torch.optim.Adam(accumulator_fn.parameters(), lr=cfg.train.acc_lr)
                inner_steps = cfg.model.learn_mining_inner_steps
                first_order_approx = cfg.model.learn_mining_first_order_approx
                inner_learning_rate = cfg.model.learn_mining_inner_learning_rate

                # this triple loss does not do "no reduction" the way I want it to.
                per_sample_loss_fn = NoReductionTripletLoss()
                # per_sample_loss_fn = TripletLoss(reduction='none')
                # just use the mean for val risk function for now
                val_risk_function = TripletLoss(reduction='mean')

                engine = torchreid.engine.ImageTripletEngineLearnedMining(
                    datamanager,
                    model,
                    optimizer=optimizer,
                    margin=cfg.loss.triplet.margin,
                    weight_t=cfg.loss.triplet.weight_t,
                    weight_x=cfg.loss.triplet.weight_x,
                    scheduler=scheduler,
                    use_gpu=cfg.use_gpu,
                    label_smooth=cfg.loss.softmax.label_smooth,
                    accumulator_fn=accumulator_fn,
                    accumulator_optimizer=accumulator_optimizer,
                    accumulator_lr=cfg.train.acc_lr,
                    inner_steps=inner_steps,
                    first_order_approx=first_order_approx,
                    inner_learning_rate=inner_learning_rate,
                    per_sample_loss_fn=per_sample_loss_fn,
                    val_risk_function=val_risk_function
                )
            else:
                engine = torchreid.engine.ImageTripletEngine(
                    datamanager,
                    model,
                    optimizer=optimizer,
                    margin=cfg.loss.triplet.margin,
                    weight_t=cfg.loss.triplet.weight_t,
                    weight_x=cfg.loss.triplet.weight_x,
                    scheduler=scheduler,
                    use_gpu=cfg.use_gpu,
                    label_smooth=cfg.loss.softmax.label_smooth
                )

    else:
        if cfg.loss.name == 'softmax':
            engine = torchreid.engine.VideoSoftmaxEngine(
                datamanager,
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                use_gpu=cfg.use_gpu,
                label_smooth=cfg.loss.softmax.label_smooth,
                pooling_method=cfg.video.pooling_method
            )

        else:
            engine = torchreid.engine.VideoTripletEngine(
                datamanager,
                model,
                optimizer=optimizer,
                margin=cfg.loss.triplet.margin,
                weight_t=cfg.loss.triplet.weight_t,
                weight_x=cfg.loss.triplet.weight_x,
                scheduler=scheduler,
                use_gpu=cfg.use_gpu,
                label_smooth=cfg.loss.softmax.label_smooth
            )

    return engine


def reset_config(cfg, args):
    # kinda a bummer but need manual mapping
    if args.root:
        cfg.data.root = args.root
    if args.sources:
        cfg.data.sources = args.sources
    if args.targets:
        cfg.data.targets = args.targets
    if args.transforms:
        cfg.data.transforms = args.transforms
    if args.model_pretrained:
        cfg.model.pretrained = args.model_pretrained
    if args.lr:
        cfg.train.lr = args.lr
    if args.fixbase_epoch:
        cfg.train.fixbase_epoch = args.fixbase_epoch
    if args.dist_metric:
        cfg.test.dist_metric = args.dist_metric
    if args.normalize_feature:
        cfg.test.normalize_feature = args.normalize_feature
    if args.weight_t:
        cfg.loss.triplet.weight_t = args.weight_t
    if args.weight_x:
        cfg.loss.triplet.weight_x = args.weight_x
    if args.weight_d:
        cfg.loss.triplet.weight_d = args.weight_d
    if args.margin:
        cfg.loss.triplet.margin = args.margin
    if args.bag_size:
        cfg.train.bag_size = args.bag_size
    if args.batch_size:
        cfg.train.batch_size = args.batch_size


def check_cfg(cfg):
    if cfg.loss.name == 'triplet' and cfg.loss.triplet.weight_x == 0:
        assert cfg.train.fixbase_epoch == 0, \
            'The output of classifier is not included in the computational graph'


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--config-file', type=str, default=None, help='path to config file'
    )
    parser.add_argument(
        '-s',
        '--sources',
        type=str,
        nargs='+',
        default=None,
        help='source datasets (delimited by space)'
    )
    parser.add_argument(
        '-t',
        '--targets',
        type=str,
        default=None,
        nargs='+',
        help='target datasets (delimited by space)'
    )
    parser.add_argument(
        '--transforms', type=str, nargs='+', default=None, help='data augmentation'
    )
    parser.add_argument(
        '--root', type=str, default=None, help='path to data root'
    )
    parser.add_argument(
        '--model_pretrained', type=bool, default=None, help='use pretrained model'
    )
    parser.add_argument(
        '--lr', type=float, default=None, help='learning rate'
    )
    parser.add_argument(
        '--fixbase_epoch', type=int, default=None, help='number of epochs to fix base layers'
    )
    parser.add_argument(
        '--dist_metric', type=str, default=None, help='distance metric'
    )
    parser.add_argument(
        '--normalize_feature', type=bool, default=None, help='normalize feature vectors before computing distance'
    )
    parser.add_argument(
        '--weight_t', type=float, default=None, help='weight to balance hard triplet loss'
    )
    parser.add_argument(
        '--weight_x', type=float, default=None, help='weight to balance cross entropy loss'
    )
    parser.add_argument(
        '--weight_d', type=float, default=None, help='weight to balance bag and instance difference'
    )
    parser.add_argument(
        '--margin', type=float, default=None, help='distance margin'
    )
    parser.add_argument(
        '--bag_size', type=int, default=None, help='number of images per bag'
    )
    parser.add_argument(
        '--batch_size', type=int, default=None, help='batch size - the number of bags'
    )
    parser.add_argument(
        'opts',
        default=None,
        nargs=argparse.REMAINDER,
        help='Modify config options using the command-line'
    )
    args = parser.parse_args()

    cfg = get_default_config()
    cfg.use_gpu = torch.cuda.is_available()
    assert cfg.use_gpu, 'CUDA is not available'
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    reset_config(cfg, args)
    cfg.merge_from_list(args.opts)
    set_random_seed(cfg.train.seed)
    check_cfg(cfg)

    log_name = 'test.log' if cfg.test.evaluate else 'train.log'
    log_name += time.strftime('-%Y-%m-%d-%H-%M-%S')
    sys.stdout = Logger(osp.join(cfg.data.save_dir, log_name))

    print('Show configuration\n{}\n'.format(cfg))
    print('Collecting env info ...')
    print('** System info **\n{}\n'.format(collect_env_info()))

    if cfg.use_gpu:
        torch.backends.cudnn.benchmark = True

    datamanager = build_datamanager(cfg)

    print('Building model: {}'.format(cfg.model.name))
    model = torchreid.models.build_model(
        name=cfg.model.name,
        num_classes=datamanager.num_train_pids,
        loss=cfg.loss.name,
        pretrained=cfg.model.pretrained,
        use_gpu=cfg.use_gpu,
        batch_size=cfg.train.batch_size,
        bag_size=cfg.train.bag_size,
        acc_fn=cfg.model.acc_fn
    )

    # print(model)

    num_params, flops = compute_model_complexity(
        model, (1, 3, cfg.data.height, cfg.data.width)
    )
    print('Model complexity: params={:,} flops={:,}'.format(num_params, flops))

    if cfg.model.load_weights and check_isfile(cfg.model.load_weights):
        load_pretrained_weights(model, cfg.model.load_weights)

    if cfg.use_gpu:
        model = nn.DataParallel(model).cuda()

    optimizer = torchreid.optim.build_optimizer(model, **optimizer_kwargs(cfg))
    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer, steps_per_epoch=len(datamanager.train_loader), **lr_scheduler_kwargs(cfg)
    )
    print(f'Steps Per Epoch: {len(datamanager.train_loader)}')

    if cfg.model.resume and check_isfile(cfg.model.resume):
        cfg.train.start_epoch = resume_from_checkpoint(
            cfg.model.resume, model, optimizer=optimizer, scheduler=scheduler
        )

    print(
        'Building {}-engine for {}-reid'.format(cfg.loss.name, cfg.data.type)
    )
    engine = build_engine(cfg, datamanager, model, optimizer, scheduler)
    print(cfg)
    engine.run(**engine_run_kwargs(cfg), wandb_config=cfg)


if __name__ == '__main__':
    main()

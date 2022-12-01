import os
import math
import tempfile
import argparse
import wandb
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import transforms

from model import create_model
from utils.distributed_utils import init_distributed_mode, dist, cleanup
from utils.train_eval_utils import train_one_epoch, evaluate
import torchvision.datasets as datasets
from torch.utils.data import DataLoader


def main(args):
    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training.")

    # 初始化各进程环境
    init_distributed_mode(args=args)

    rank = args.rank
    device = torch.device(args.device)
    batch_size = args.batch_size
    weights_path = args.weights
    args.lr *= args.world_size  # 学习率要根据并行GPU的数量进行倍增
    checkpoint_path = ""
    use_wandb = args.use_wandb

    if rank == 0:  # 在第一个进程中打印信息，并实例化tensorboard
        print(args)
        if os.path.exists("./weights") is False:
            os.makedirs("./weights")

    train_dir = os.path.join(args.data_path, 'train')
    val_dir = os.path.join(args.data_path, 'val')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        train_dir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    val_dataset = datasets.ImageFolder(
        val_dir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    # 给每个rank对应的进程分配训练的样本索引
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

    # 将样本索引每batch_size个元素组成一个list
    train_batch_sampler = torch.utils.data.BatchSampler(
        train_sampler, batch_size, drop_last=True)

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    if rank == 0:
        print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_sampler=train_batch_sampler,
                                               pin_memory=True,
                                               num_workers=nw)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             sampler=val_sampler,
                                             pin_memory=True,
                                             num_workers=nw)
    # 实例化模型
    model = create_model(num_class=args.num_class).to(device)

    if rank == 0:
        if use_wandb:
            wandb.init(project="vit-multi-gpu")
            wandb.watch(model, log="all")

    # 如果存在预训练权重则载入
    if os.path.exists(weights_path):
        weights_dict = torch.load(weights_path, map_location=device)
        load_weights_dict = {k: v for k, v in weights_dict.items()
                             if model.state_dict()[k].numel() == v.numel()}
        model.load_state_dict(load_weights_dict, strict=False)
    else:
        checkpoint_path = os.path.join(tempfile.gettempdir(), "initial_weights.pt")
        # 如果不存在预训练权重，需要将第一个进程中的权重保存，然后其他进程载入，保持初始化权重一致
        if rank == 0:
            torch.save(model.state_dict(), checkpoint_path)

        dist.barrier()
        # 这里注意，一定要指定map_location参数，否则会导致第一块GPU占用更多资源
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # 是否冻结权重
    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除最后的全连接层外，其他权重全部冻结
            if "fc" not in name:
                para.requires_grad_(False)
    else:
        # 只有训练带有BN结构的网络时使用SyncBatchNorm采用意义
        if args.syncBN:
            # 使用SyncBatchNorm后训练会更耗时
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)

    # 转为DDP模型
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    # optimizer
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-5)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    torch.autograd.set_detect_anomaly(True)

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)

        mean_loss = train_one_epoch(model=model,
                                    optimizer=optimizer,
                                    data_loader=train_loader,
                                    device=device,
                                    epoch=epoch,
                                    use_wandb=use_wandb)

        scheduler.step()

        sum_num = evaluate(model=model,
                           data_loader=val_loader,
                           device=device)
        acc = sum_num / val_sampler.total_size

        if rank == 0:
            print("[epoch {}] accuracy: {}".format(epoch, round(acc, 3)))
            if use_wandb:
                wandb.log({'epoch': epoch, 'loss': mean_loss, 'accuracy': acc,
                           'learning_rate': optimizer.param_groups[0]["lr"]})

            torch.save(model.module.state_dict(), "./weights/model-{}.pth".format(epoch))

    # 删除临时缓存文件
    if rank == 0:
        if os.path.exists(checkpoint_path) is True:
            os.remove(checkpoint_path)

    cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default="/data/cjj/imagenet")
    parser.add_argument('--num_class', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=120)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)
    # 是否启用SyncBatchNorm
    parser.add_argument('--syncBN', type=bool, default=True)

    parser.add_argument('--weights', type=str, default='./weights/checkpoint.pth',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    # 不要改该参数，系统会自动分配
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    # 开启的进程数(注意不是线程),不用设置该参数，会根据nproc_per_node自动设置
    parser.add_argument('--world-size', default=4, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--use-wandb', type=bool, default=True)
    opt = parser.parse_args()

    main(opt)

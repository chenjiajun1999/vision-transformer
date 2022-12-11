import sys
import wandb
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchtoolbox.tools import mixup_data, mixup_criterion
from utils.distributed_utils import reduce_value, is_main_process


def self_cross_entropy(input, target, ignore_index=None):
    '''自己用pytorch实现cross_entropy，
       有时候会因为各种原因，如：样本问题等，出现个别样本的loss为nan，影响模型的训练，
       不适用于所有样本loss都为nan的情况
       input:n*categ
       target:n
    '''
    input = input.contiguous().view(-1, input.shape[-1])
    log_prb = F.log_softmax(input, dim=1)

    one_hot = torch.zeros_like(input).scatter(1, target.view(-1, 1), 1)  # 将target转换成one-hot编码
    loss = -(one_hot * log_prb).sum(dim=1)  # n,得到每个样本的loss

    if ignore_index:  # 忽略[PAD]的label
        non_pad_mask = target.ne(0)
        loss = loss.masked_select(non_pad_mask)

    not_nan_mask = ~torch.isnan(loss)  # 找到loss为非nan的样本
    loss = loss.masked_select(not_nan_mask).mean()
    return loss


def train_one_epoch(model, optimizer, data_loader, device, epoch, use_wandb, use_mixup):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    mean_loss = torch.zeros(1).to(device)
    optimizer.zero_grad()

    # 在进程0中打印训练进度
    if is_main_process():
        data_loader = tqdm(data_loader, file=sys.stdout)

    for step, (images, labels) in enumerate(data_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Scheduler https://arxiv.org/pdf/2205.01580.pdf
        if use_mixup is True:
            images, labels_a, labels_b, lam = mixup_data(images, labels, 0.2)
            pred = model(images)
            loss = mixup_criterion(loss_function, pred, labels_a, labels_b, lam)
        else:
            pred = model(images)
            loss = loss_function(pred, labels)

        loss.backward()
        loss = reduce_value(loss, average=True)
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)  # update mean losses
        # 在进程0中打印平均loss
        if is_main_process():
            data_loader.desc = "[epoch {}] loss: {:.3f}".format(epoch, mean_loss.item())
            if use_wandb:
                wandb.log({'loss (each step)': mean_loss.item()})

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    return mean_loss.item()


@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()

    # 用于存储预测正确的样本个数
    sum_num = torch.zeros(1).to(device)

    # 在进程0中打印验证进度
    if is_main_process():
        data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        images, labels = data
        pred = model(images.to(device))
        pred = torch.max(pred, dim=1)[1]
        sum_num += torch.eq(pred, labels.to(device)).sum()

    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    sum_num = reduce_value(sum_num, average=False)

    return sum_num.item()

import torch
import torch.backends.cudnn as cudnn
import os
import argparse
import numpy as np

import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
import time
from model import create_model

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', default='/data/cjj/imagenet', type=str, help='trainset directory')
parser.add_argument('--dataset', default='ImageNet', type=str, help='Dataset name')
parser.add_argument('--arch', default='resnet50', type=str, help='network architecture')
parser.add_argument('--batch-size', type=int, default=256, help='batch size')
parser.add_argument('--num-workers', type=int, default=8, help='number workers')
parser.add_argument('--pretrained', default='./weights/checkpoint.pth', type=str, help='pretrained weights')
parser.add_argument('--gpu', type=str, default='1')
parser.add_argument('--manual_seed', type=int, default=0)

# global hyperparameter set
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device('cuda')

np.random.seed(args.manual_seed)
torch.manual_seed(args.manual_seed)
torch.cuda.manual_seed_all(args.manual_seed)

num_classes = 1000

test_set = datasets.ImageFolder(
    os.path.join(args.data, 'val'),
    transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ]))

testloader = torch.utils.data.DataLoader(
    test_set, batch_size=args.batch_size, shuffle=False,
    num_workers=args.num_workers, pin_memory=True)
# --------------------------------------------------------------------------------------------

# Model
print('==> Building model..')

net = create_model(num_class=1000).to(device)
if args.pretrained != "":
    assert os.path.exists(args.pretrained), "weights file: '{}' not exist.".format(args.weights)
    weights_dict = torch.load(args.pretrained, map_location=device)
    print(net.load_state_dict(weights_dict, strict=False))
cudnn.benchmark = True


def correct_num(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    correct = pred.eq(target.view(-1, 1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:, :k].float().sum()
        res.append(correct_k.item())
    return res


net.eval()
correct1 = 0
correct5 = 0
total = 0
sum_time = time.time()
with torch.no_grad():
    batch_start_time = time.time()
    for batch_idx, (inputs, target) in enumerate(testloader):
        inputs, target = inputs.cuda(), target.cuda()
        logits = net(inputs)
        print('batch_idx:{}/{}, Duration:{:.2f}'.format(batch_idx, len(testloader), time.time() - batch_start_time))
        batch_start_time = time.time()

        prec1, prec5 = correct_num(logits, target, topk=(1, 5))
        correct1 += prec1
        correct5 += prec5
        total += target.size(0)

    acc1 = round(correct1 / total, 4)
    acc5 = round(correct5 / total, 4)

    print('Test accuracy_1:{:.4f}\n'
          'Test accuracy_5:{:.4f}\n'
          .format(acc1, acc5))

print('avg times:', (time.time() - sum_time) / 50000)

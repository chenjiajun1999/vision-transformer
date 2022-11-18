import os
import json
import argparse

import torch
from PIL import Image
from torchvision import transforms
from model import ViT
from pathlib import Path

IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'


def run(weight='', source='', dict='', device=''):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    # load image
    assert os.path.exists(source), "file: '{}' dose not exist.".format(source)
    imgs_path = []
    if Path(source).suffix[1:] in IMG_FORMATS:
        imgs_path.append(source)
    else:
        assert os.path.isdir(source), "file: '{}' is not dir.".format(source)
        for file in os.listdir(source):
            if Path(file).suffix[1:] in IMG_FORMATS:
                imgs_path.append(os.path.join(source, file))
    imgs = []
    for path in imgs_path:
        img = Image.open(path)
        # [N, C, H, W]
        img = data_transform(img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)
        imgs.append(img)

    assert len(imgs) != 0, "file: '{}' dose not have images.".format(source)

    # read class_indict
    assert os.path.exists(dict), "file: '{}' dose not exist.".format(dict)
    with open(dict, "r") as f:
        class_indict = [l.strip() for l in open(dict).readlines()]

    # create model
    model = ViT(num_class=21843).to(device)
    # load model weights
    assert os.path.exists(weight), "file: '{}' dose not exist.".format(weight)
    # model.load_state_dict(torch.load(weight, map_location=device))
    model.eval()
    for img in imgs:
        with torch.no_grad():
            # predict class
            output = torch.squeeze(model(img.to(device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()
            print("class: {}   prob: {:.3}".format(class_indict[predict_cla],
                                                   predict[predict_cla].numpy()))


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default="")
    parser.add_argument('--weight', type=str, default='./vit_base_patch16_224_in21k.pth',
                        help='initial weights path')
    parser.add_argument('--dict', type=str, default='./synset.txt')
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_opt()
    run(**vars(opt))

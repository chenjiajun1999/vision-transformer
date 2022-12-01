# Vision Transformer

语言：Python 3.7

环境：ubuntu 16.04

Dataset：ImageNet

~~~
imagenet/train/
├── n01440764
│   ├── n01440764_10026.JPEG
│   ├── n01440764_10027.JPEG
│   ├── ......
├── ......
imagenet/val/
├── n01440764
│   ├── ILSVRC2012_val_00000293.JPEG
│   ├── ILSVRC2012_val_00002138.JPEG
│   ├── ......
├── ......
~~~

单卡训练：
~~~
python train.py
~~~

多卡训练：
~~~
CUDA_VISIBLE_DEVICES=0,2 python -m torch.distributed.launch --nproc_per_node=2 --use_env train_multi_GPU.py
~~~
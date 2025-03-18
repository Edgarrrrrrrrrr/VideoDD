from functools import lru_cache
import random
import time
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset
from collections import defaultdict
import torch.utils.data
from torchvision import datasets, transforms
from scipy.ndimage.interpolation import rotate as scipyrotate
import tqdm
import torchvision.models.video as models
from distill_utils.dataset import Kinetics400, UCF101, HMDB51, miniUCF101, staticHMDB51, staticUCF101, staticUCF50, singleSSv2, singleKinetics400
from networks import MLP, ConvNet, LeNet, AlexNet, AlexNetBN, VGG11, VGG11BN, ResNet18, ResNet18BN_AP, ResNet18BN, VideoConvNetMean, VideoConvNetMLP, VideoConvNetLSTM, VideoConvNetRNN, VideoConvNetGRU, ConvNet3D
from loguru import logger

# @lru_cache()
def get_dataset(dataset, data_path, num_workers=0,img_size=(112,112),split_num=1,split_id=0,split_mode='mean'):
    if dataset == 'MNIST':
        channel = 1
        im_size = (28, 28)
        num_classes = 10
        mean = [0.1307]
        std = [0.3081]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.MNIST(data_path, train=True, download=True, transform=transform) # no augmentation
        dst_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)
        class_names = [str(c) for c in range(num_classes)]

    elif dataset == 'FashionMNIST':
        channel = 1
        im_size = (28, 28)
        num_classes = 10
        mean = [0.2861]
        std = [0.3530]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.FashionMNIST(data_path, train=True, download=True, transform=transform) # no augmentation
        dst_test = datasets.FashionMNIST(data_path, train=False, download=True, transform=transform)
        class_names = dst_train.classes

    elif dataset == 'SVHN':
        channel = 3
        im_size = (32, 32)
        num_classes = 10
        mean = [0.4377, 0.4438, 0.4728]
        std = [0.1980, 0.2010, 0.1970]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.SVHN(data_path, split='train', download=True, transform=transform)  # no augmentation
        dst_test = datasets.SVHN(data_path, split='test', download=True, transform=transform)
        class_names = [str(c) for c in range(num_classes)]

    elif dataset == 'CIFAR10':
        channel = 3
        im_size = (32, 32)
        num_classes = 10
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.CIFAR10(data_path, train=True, download=True, transform=transform) # no augmentation
        dst_test = datasets.CIFAR10(data_path, train=False, download=True, transform=transform)
        class_names = dst_train.classes

    elif dataset == 'CIFAR100':
        channel = 3
        im_size = (32, 32)
        num_classes = 100
        mean = [0.5071, 0.4866, 0.4409]
        std = [0.2673, 0.2564, 0.2762]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.CIFAR100(data_path, train=True, download=True, transform=transform) # no augmentation
        dst_test = datasets.CIFAR100(data_path, train=False, download=True, transform=transform)
        class_names = dst_train.classes

    elif dataset == 'TinyImageNet':
        channel = 3
        im_size = (64, 64)
        num_classes = 200
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        data = torch.load(os.path.join(data_path, 'tinyimagenet.pt'), map_location='cpu')

        class_names = data['classes']

        images_train = data['images_train']
        labels_train = data['labels_train']
        images_train = images_train.detach().float() / 255.0
        labels_train = labels_train.detach()
        for c in range(channel):
            images_train[:,c] = (images_train[:,c] - mean[c])/std[c]
        dst_train = TensorDataset(images_train, labels_train)  # no augmentation

        images_val = data['images_val']
        labels_val = data['labels_val']
        images_val = images_val.detach().float() / 255.0
        labels_val = labels_val.detach()

        for c in range(channel):
            images_val[:, c] = (images_val[:, c] - mean[c]) / std[c]

        dst_test = TensorDataset(images_val, labels_val)  # no augmentation
 
    elif dataset == 'ImageNet':
        channel = 3
        # im_size = (128, 128)
        im_size = (64, 64) 
        num_classes = 1000

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        
        resized_data_path = data_path+"/imagenet_%dx%d"%im_size
        if os.path.exists(resized_data_path):
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=mean, std=std),])
            # the images are already resized
            path = resized_data_path
        else:
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=mean, std=std),
                                            transforms.Resize(im_size),
                                            transforms.CenterCrop(im_size)])
            
            path = data_path+"/imagenet"

        dst_train = datasets.ImageNet(path, split="train", transform=transform) # no augmentation
        dst_test = datasets.ImageNet(path, split="val", transform=transform)
        class_names = None

    elif dataset in ['Kinetics400', 'Kinetics400_long']:
        # this is a video dataset
        channel = 3
        # im_size = (128, 128)
        im_size = (64,64) if dataset == 'Kinetics400' else (112,112)
        num_classes = 400

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]  # use imagenet transform
        
        path = data_path+"/Kinetics" if dataset == 'Kinetics400' else data_path+"/kinetics_112x112x16"
        assert os.path.exists(path)
        # the images are already resized
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=mean, std=std),])

        dst_train = Kinetics400(path, split="train", transform=transform) # no augmentation
        dst_test  = Kinetics400(path, split="val", transform=transform)
        class_names = None

    elif dataset == 'UCF101':
        # this is a video dataset
        channel = 3
        # im_size = (112, 112)
        im_size = img_size
        num_classes = 101

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]  # use imagenet transform
        
        path = data_path+"/UCF101"
        assert os.path.exists(path)
        if im_size != (112,112):
            transform = transforms.Compose([transforms.Resize((100,80)),
                                            transforms.RandomCrop(im_size),
                                            #transforms.Resize(im_size),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=mean, std=std)
                                            ])
        else:
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=mean, std=std)
                                            ])
        dst_train = UCF101(path, split="train", transform=transform) # no augmentation
        dst_test  = UCF101(path, split="test", transform=transform)
        print("UCF101 train: ", len(dst_train), "test: ", len(dst_test))
        class_names = None
    
    elif dataset == 'HMDB51':
        # this is a video dataset
        channel = 3
        im_size = img_size 
        num_classes = 51

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]  # use imagenet transform
        
        path = data_path+"/HMDB51"
        assert os.path.exists(path)
        if im_size != (112,112):
            print("do some transform")
            transform = transforms.Compose([transforms.Resize((100,80)),
                                            transforms.RandomCrop(im_size),
                                            #transforms.Resize(im_size),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=mean, std=std)
                                            ])
        else:
            logger.info("图片尺寸为(112,112)："+"do not do some augmentation")
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=mean, std=std)
                                            ])

        dst_train = HMDB51(path, split="train", transform=transform) # no augmentation
        dst_test  = HMDB51(path, split="test", transform=transform)
        #print(type(dst_test))
        #print(type(dst_train.labels))
        logger.info("数据集的大小： HMDB51 train: "+str(len(dst_train)) + " test: " + str(len(dst_test)))
        class_names = None

    elif dataset in ['miniUCF101', 'miniUCF101_long']:
        # this is a video dataset, only 50 classes of UCF101
        channel = 3
        im_size = img_size
        num_classes = 50

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]  # use imagenet transform
        
        path = data_path+"/UCF101"
        assert os.path.exists(path)
        if im_size != (112,112):
            transform = transforms.Compose([transforms.Resize((100,80)),
                                            transforms.RandomCrop(im_size),
                                            #transforms.Resize(im_size),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=mean, std=std)
                                            ])
        else:
            print("miniUCF im_size:", im_size)
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=mean, std=std)
                                            ])
        if dataset == 'miniUCF101':
            dst_train = miniUCF101(path, split="train", transform=transform) # no augmentation
            dst_test  = miniUCF101(path, split="test", transform=transform)
        print("UCF101 train: ", len(dst_train), "test: ", len(dst_test))
        class_names = None

    elif dataset == 'staticHMDB51':
        # this is a video dataset, get boring video of HMDB51
        channel = 3
        im_size = img_size
        num_classes = 51

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]  # use imagenet transform
        
        path = data_path+"/HMDB51"
        assert os.path.exists(path)
        if im_size != (112,112):
            transform = transforms.Compose([transforms.Resize((100,80)),
                                            transforms.RandomCrop(im_size),
                                            #transforms.Resize(im_size),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=mean, std=std)
                                            ])
        else:
            transform = transforms.Compose([#transforms.Resize((160,120)),
                                                #transforms.RandomCrop(im_size),
                                                #transforms.CenterCrop(im_size),
                                                #transforms.Resize(im_size),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=mean, std=std)
                                                ])

        dst_train = staticHMDB51(path, split="train", transform=transform) # no augmentation
        dst_test  = staticHMDB51(path, split="test", transform=transform)
        class_names = None

    elif dataset == 'staticUCF101':
        # this is a video dataset, get boring video of UCF101
        channel = 3
        im_size = img_size
        num_classes = 101

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]  # use imagenet transform
        
        path = data_path+"/UCF101"
        assert os.path.exists(path)
        if im_size != (112,112):
            transform = transforms.Compose([transforms.Resize((100,80)),
                                            transforms.RandomCrop(im_size),
                                            #transforms.Resize(im_size),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=mean, std=std)
                                            ])
        else:
            transform = transforms.Compose([#transforms.Resize((160,120)),
                                                #transforms.RandomCrop(im_size),
                                                #transforms.CenterCrop(im_size),
                                                #transforms.Resize(im_size),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=mean, std=std)
                                                ])

        dst_train = staticUCF101(path, split="train", transform=transform) # no augmentation
        dst_test  = staticUCF101(path, split="test", transform=transform)
        class_names = None
        
        print("UCF101 train: ", len(dst_train), "test: ", len(dst_test))

    elif dataset == 'staticUCF50':
        # this is a video dataset, get boring video of UCF50
        channel = 3
        im_size = img_size
        num_classes = 50

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]  # use imagenet transform
        
        path = data_path+"/UCF101"
        assert os.path.exists(path)
        if im_size != (112,112):
            transform = transforms.Compose([transforms.Resize((100,80)),
                                            transforms.RandomCrop(im_size),
                                            #transforms.Resize(im_size),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=mean, std=std)
                                            ])
        else:
            transform = transforms.Compose([#transforms.Resize((160,120)),
                                                #transforms.RandomCrop(im_size),
                                                #transforms.CenterCrop(im_size),
                                                #transforms.Resize(im_size),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=mean, std=std)
                                                ])

        dst_train = staticUCF50(path, split="train", transform=transform) # no augmentation
        dst_test  = staticUCF50(path, split="test", transform=transform)
        class_names = None
        
        print("UCF101 train: ", len(dst_train), "test: ", len(dst_test))

    elif dataset == 'singleHMDB51':
        # this is a img dataset for static learning
        channel = 3
        im_size = img_size 
        num_classes = 51

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]  # use imagenet transform
        
        path = data_path+"/HMDB51"
        assert os.path.exists(path)
        if im_size == (64,64):
            transform = transforms.Compose([transforms.Resize((100,80)),
                                            transforms.RandomCrop(im_size),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=mean, std=std)
                                            ])
        else:
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=mean, std=std)
                                            ])

        dst_train = staticHMDB51(path, split="train", transform=transform, frames = 1) # no augmentation
        dst_test  = staticHMDB51(path, split="test", transform=transform, frames = 1)
        print("HMDB51 train: ", len(dst_train), "test: ", len(dst_test))
        class_names = None

    elif dataset == 'singleUCF50':
        channel = 3
        im_size = img_size
        num_classes = 50

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]  # use imagenet transform
        
        path = data_path+"/UCF101"
        assert os.path.exists(path)
        if im_size != (112,112):
            transform = transforms.Compose([transforms.Resize((100,80)),
                                            transforms.RandomCrop(im_size),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=mean, std=std)
                                            ])
        else:
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=mean, std=std)
                                            ])

        dst_train = staticUCF50(path, split="train", transform=transform, frames = 1, split_num = split_num, split_id = split_id, split_mode=split_mode) # no augmentation
        dst_test  = staticUCF50(path, split="test", transform=transform, frames = 1, split_num = split_num, split_id = split_id, split_mode=split_mode)
        class_names = None
        
        print("UCF101 train: ", len(dst_train), "test: ", len(dst_test))

    elif dataset == 'singleUCF101':
        # this is a img dataset for static learning
        channel = 3
        im_size = img_size
        num_classes = 101

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]  # use imagenet transform
        
        path = data_path+"/UCF101"
        assert os.path.exists(path)
        if im_size != (112,112):
            transform = transforms.Compose([transforms.Resize((100,80)),
                                            transforms.RandomCrop(im_size),
                                            #transforms.Resize(im_size),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=mean, std=std)
                                            ])
        else:
            transform = transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=mean, std=std)
                                            ])

        dst_train = staticUCF101(path, split="train", transform=transform, frames = 1, split_num = split_num, split_id = split_id) # no augmentation
        dst_test  = staticUCF101(path, split="test", transform=transform, frames = 1, split_num = split_num, split_id = split_id)
        class_names = None
        
        print("UCF101 train: ", len(dst_train), "test: ", len(dst_test))
    
    elif dataset == 'singleKinetics400':
        # this is a img dataset for static learning
        channel = 3
        im_size = (64, 64)
        num_classes = 400

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]  # use imagenet transform
        
        path = data_path+"/Kinetics"
        assert os.path.exists(path)
        # the images are already resized
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=mean, std=std),])

        dst_train = singleKinetics400(path, split="train", transform=transform) # no augmentation
        dst_test  = singleKinetics400(path, split="val", transform=transform)
        class_names = None

    elif dataset == 'singleSSv2':
        # this is a img dataset for static learning
        channel = 3
        im_size = (64, 64)
        num_classes = 174

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]  # use imagenet transform
        
        path = data_path+"/SSv2"
        assert os.path.exists(path)
        # the images are already resized
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=mean, std=std),])

        dst_train = singleSSv2(path, split="train", transform=transform) # no augmentation
        dst_test  = singleSSv2(path, split="val", transform=transform)
        class_names = None

    else:
        exit('unknown dataset: %s'%dataset)

    testloader = torch.utils.data.DataLoader(dst_test, batch_size=64, shuffle=False, num_workers=num_workers)
    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader

class MultiStaticSharedDataset(Dataset):
    def __init__(self, static, dynamic, hallucinator):
        self.static = static.detach().float()
        self.dynamic = dynamic.detach().float()
        self.hallucinator = hallucinator 
        self.n_s, _, _, _ = static.shape
        self.n_c, self.dpc, _, _, _, _ = dynamic.shape
    def __getitem__(self, index):
        per_s = self.n_s // self.n_c
        if per_s == 10:
            label = index // 5 # test for vpc=5
            idx = index % 5
            static_idx = label * per_s + 2 * idx + random.randint(0, 1)
            dynamic_idx = 2 * idx + random.randint(0, 1)
        elif per_s == 2:
            label = index # test for vpc=1
            static_idx = random.randint(0, per_s - 1) + label * per_s
            dynamic_idx = random.randint(0, self.dpc - 1)
        else:
            print("error for multi-static-shared-dataset")
            exit()
        static = self.static[static_idx, :, :, :] #3, 112, 112
        hal_idx = random.randint(0, len(self.hallucinator) - 1)
        dynamic = self.dynamic[label, dynamic_idx, :, :, :, :] #16, 1, 112, 112
        hallucinator = self.hallucinator[hal_idx]
        video = hallucinator(static.unsqueeze(0), dynamic.unsqueeze(0))
        return video[0], label #frames,c,h,w
    def __len__(self):
        if self.n_s == self.n_c * 10:
            return self.n_c*5 # test for vpc=5
        elif self.n_s == self.n_c * 2:
            return self.n_c
        else:
            print("error for multi-static-shared-dataset")
            exit()


class TensorDataset(Dataset):
    def __init__(self, images, labels): # images: n x c x h x w tensor
        self.images = images.detach().float()
        self.labels = labels.detach()

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.images.shape[0]



def get_default_convnet_setting():
    net_width, net_depth, net_act, net_norm, net_pooling = 128, 3, 'relu', 'instancenorm', 'avgpooling'
    return net_width, net_depth, net_act, net_norm, net_pooling



def get_network(model, channel, num_classes, im_size=(32, 32), frames = 16, dist = True):
    torch.random.manual_seed(int(time.time() * 1000) % 100000)
    net_width, net_depth, net_act, net_norm, net_pooling = get_default_convnet_setting()

    if model == 'MLP':
        net = MLP(channel=channel, num_classes=num_classes)
    elif model == 'ConvNet':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'LeNet':
        net = LeNet(channel=channel, num_classes=num_classes)
    elif model == 'AlexNet':
        net = AlexNet(channel=channel, num_classes=num_classes)
    elif model == 'AlexNetBN':
        net = AlexNetBN(channel=channel, num_classes=num_classes)
    elif model == 'VGG11':
        net = VGG11( channel=channel, num_classes=num_classes)
    elif model == 'VGG11BN':
        net = VGG11BN(channel=channel, num_classes=num_classes)
    elif model == 'ResNet18':
        net = ResNet18(channel=channel, num_classes=num_classes)
    elif model == 'ResNet18BN_AP':
        net = ResNet18BN_AP(channel=channel, num_classes=num_classes)
    elif model == 'ResNet18BN':
        net = ResNet18BN(channel=channel, num_classes=num_classes)

    elif model == 'ConvNetD1':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=1, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetD2':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=2, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetD3':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=3, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetD4':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=4, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetD5':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=5, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetD6':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=6, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetD7':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=7, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetD8':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=8, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)

    elif model == 'ConvNetW32':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=32, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetW64':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=64, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetW128':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=128, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetW256':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=256, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)

    elif model == 'ConvNetAS':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act='sigmoid', net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetAR':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act='relu', net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetAL':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act='leakyrelu', net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetASwish':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act='swish', net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetASwishBN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act='swish', net_norm='batchnorm', net_pooling=net_pooling, im_size=im_size)

    elif model == 'ConvNetNN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm='none', net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetBN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm='batchnorm', net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetLN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm='layernorm', net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetIN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm='instancenorm', net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetGN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm='groupnorm', net_pooling=net_pooling, im_size=im_size)

    elif model == 'ConvNetNP':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling='none', im_size=im_size)
    elif model == 'ConvNetMP':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling='maxpooling', im_size=im_size)
    elif model == 'ConvNetAP':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling='avgpooling', im_size=im_size)

    elif model == 'VideoConvNetMean':
        net = VideoConvNetMean(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'VideoConvNetMLP':
        net = VideoConvNetMLP(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'VideoConvNetLSTM':
        net = VideoConvNetLSTM(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'VideoConvNetRNN':
        net = VideoConvNetRNN(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'VideoConvNetGRU':
        net = VideoConvNetGRU(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNet3D':
        net = ConvNet3D(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm='none', net_pooling='maxpooling', im_size=im_size,frames=frames)
    else:
        net = None
        exit('unknown model: %s'%model)


    if dist:
        gpu_num = torch.cuda.device_count()
        if gpu_num>0:
            device = 'cuda'
            if gpu_num>1:
                net = nn.DataParallel(net)
        else:
            device = 'cpu'
        net = net.to(device)

    return net



def get_time():
    return str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))



def distance_wb(gwr, gws):
    shape = gwr.shape
    if len(shape) == 4: # conv, out*in*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2] * shape[3])
        gws = gws.reshape(shape[0], shape[1] * shape[2] * shape[3])
    elif len(shape) == 3:  # layernorm, C*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2])
        gws = gws.reshape(shape[0], shape[1] * shape[2])
    elif len(shape) == 2: # linear, out*in
        tmp = 'do nothing'
    elif len(shape) == 1: # batchnorm/instancenorm, C; groupnorm x, bias
        gwr = gwr.reshape(1, shape[0])
        gws = gws.reshape(1, shape[0])
        return torch.tensor(0, dtype=torch.float, device=gwr.device)

    dis_weight = torch.sum(1 - torch.sum(gwr * gws, dim=-1) / (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 0.000001))
    dis = dis_weight
    return dis



def match_loss(gw_syn, gw_real, args):
    dis = torch.tensor(0.0).to(args.device)

    if args.dis_metric == 'ours':
        for ig in range(len(gw_real)):
            gwr = gw_real[ig]
            gws = gw_syn[ig]
            dis += distance_wb(gwr, gws)

    elif args.dis_metric == 'mse':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = torch.sum((gw_syn_vec - gw_real_vec)**2)

    elif args.dis_metric == 'cos':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = 1 - torch.sum(gw_real_vec * gw_syn_vec, dim=-1) / (torch.norm(gw_real_vec, dim=-1) * torch.norm(gw_syn_vec, dim=-1) + 0.000001)

    else:
        exit('unknown distance function: %s'%args.dis_metric)

    return dis



def get_loops(ipc, dataset=None):
    # Get the two hyper-parameters of outer-loop and inner-loop.
    # The following values are empirically good.
    if ipc == 1 or ipc ==5:
        outer_loop, inner_loop = 1, 1
    elif ipc == 10:
        outer_loop, inner_loop = 10, 50
    elif ipc == 20:
        outer_loop, inner_loop = 20, 25
    elif ipc == 30:
        outer_loop, inner_loop = 30, 20
    elif ipc == 40:
        outer_loop, inner_loop = 40, 15
    elif ipc == 50:
        outer_loop, inner_loop = 50, 10
    else:
        outer_loop, inner_loop = 0, 0
        exit('loop hyper-parameters are not defined for %d ipc'%ipc)
    return outer_loop, inner_loop



def epoch_old(mode, dataloader, net, optimizer, criterion, args, aug):
    loss_avg, acc_avg, num_exp = 0, 0, 0
    net = net.to(args.device)
    criterion = criterion.to(args.device)

    if mode == 'train':
        net.train()
    else:
        net.eval()

    for i_batch, datum in enumerate(dataloader):
        img = datum[0].float().to(args.device)
        if aug:
            if args.dsa:
                img = DiffAugment(img, args.dsa_strategy, param=args.dsa_param)
            else:
                img = augment(img, args.dc_aug_param, device=args.device)
        lab = datum[1].long().to(args.device)
        n_b = lab.shape[0]

        output = net(img)
        loss = criterion(output, lab)
        acc = np.sum(np.equal(np.argmax(output.cpu().data.numpy(), axis=-1), lab.cpu().data.numpy()))

        loss_avg += loss.item()*n_b
        acc_avg += acc
        num_exp += n_b

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    loss_avg /= num_exp
    acc_avg /= num_exp

    return loss_avg, acc_avg, None


# def epoch(mode, dataloader, net, optimizer, criterion, args):
#     loss_avg, acc_avg, num_exp = 0, 0, 0
#     top5_acc_avg, top3_acc_avg, top1_acc_avg= 0.0, 0.0, 0.0
#     net = net.to(args.device)
#     criterion = criterion.to(args.device)

#     if mode == 'train':
#         net.train()
#     else:
#         net.eval()

#     correct_per_class = defaultdict(list)

#     if mode == 'train':
#         #print("epoch in for")
#         #print("Device during train epoch:", args.device)
#         for i_batch, datum in enumerate(dataloader):
#             img = datum[0].float().to(args.device)
#             if 'Video' in args.model:
#                 img = img[:,:, :, 24:-24,24:-24]
#             img = (img - img.mean()) / img.std()
#             lab = datum[1].long().to(args.device)
#             n_b = lab.shape[0]

#             output = net(img)
#             loss = criterion(output, lab)
#             matched = np.equal(np.argmax(output.cpu().data.numpy(), axis=-1), lab.cpu().data.numpy())
#             acc = np.sum(matched)
#             top5_preds = np.argsort(output.cpu().data.numpy(), axis=-1)[:, -5:]
#             top5_matched = np.array([lab.cpu().data.numpy()[i] in top5_preds[i] for i in range(len(lab))])
#             top5_acc = np.sum(top5_matched)
#             #print("epoch in forrrrr")
#             for y, c in zip(lab.cpu().tolist(), matched.tolist()):
#                 correct_per_class[y].append(c)
#             #print("end epoch in forrrrr")
#             loss_avg += loss.item()*n_b
#             acc_avg += acc
#             num_exp += n_b
#             top5_acc_avg += top5_acc

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             #print("end epoch in for")
#     else :
#         #print("test epoch in for")
#         #print("Device during test epoch:", args.device)
#         for j_ in range(3):
#             for i_batch, datum in enumerate(dataloader):
#                 img = datum[0].float().to(args.device)
#                 if 'Video' in args.model:
#                     img = img[:,:, :, 24:-24,24:-24]
#                 img = (img - img.mean()) / img.std()
#                 lab = datum[1].long().to(args.device)
#                 n_b = lab.shape[0]
#                 #print("Image device:", img.device)
#                 #print("Model device:", next(net.parameters()).device)

#                 output = net(img)
#                 loss = criterion(output, lab)
#                 acc = np.sum(np.equal(np.argmax(output.cpu().data.numpy(), axis=-1), lab.cpu().data.numpy()))
#                 top1_preds = np.argsort(output.cpu().data.numpy(), axis=-1)[:, -1:]
#                 top1_matched = np.array([lab.cpu().data.numpy()[i] in top1_preds[i] for i in range(len(lab))])
#                 top1_acc = np.sum(top1_matched)
#                 top3_preds = np.argsort(output.cpu().data.numpy(), axis=-1)[:, -3:]
#                 top3_matched = np.array([lab.cpu().data.numpy()[i] in top3_preds[i] for i in range(len(lab))])
#                 top3_acc = np.sum(top3_matched)
#                 top5_preds = np.argsort(output.cpu().data.numpy(), axis=-1)[:, -5:]
#                 top5_matched = np.array([lab.cpu().data.numpy()[i] in top5_preds[i] for i in range(len(lab))])
#                 top5_acc = np.sum(top5_matched)
#                 #print("test epoch in forrrrr")
#                 for y, c in zip(lab.cpu().tolist(), np.equal(np.argmax(output.cpu().data.numpy(), axis=-1), lab.cpu().data.numpy()).tolist()):
#                     correct_per_class[y].append(c)
#                 #print("end test epoch in forrrrr")
#                 loss_avg += loss.item()*n_b
#                 acc_avg += acc
#                 top5_acc_avg += top5_acc
#                 top3_acc_avg += top3_acc
#                 top1_acc_avg += top1_acc
#                 num_exp += n_b
#             #print("end epoch in for")    

#     loss_avg /= num_exp
#     acc_avg /= num_exp
#     top5_acc_avg /= num_exp
#     top3_acc_avg /= num_exp
#     top1_acc_avg /= num_exp

#     top_acc_avg = [acc_avg, top1_acc_avg, top3_acc_avg, top5_acc_avg]

#     correct_per_class = dict(correct_per_class)
#     correct_per_class = [
#         np.mean(correct_per_class[i]) 
#             if i in correct_per_class 
#             else None 
#         for i in range(len(correct_per_class))]

#     if args.eval_mode == 'top5':
#         return loss_avg, top_acc_avg, correct_per_class
#     else:
#         return loss_avg, acc_avg, correct_per_class
def epoch(mode, dataloader, net, optimizer, criterion, args):
    loss_avg, acc_avg, num_exp = 0, 0, 0
    top5_acc_avg, top3_acc_avg, top1_acc_avg= 0.0, 0.0, 0.0
    net = net.to(args.device)
    criterion = criterion.to(args.device)

    if mode == 'train':
        net.train()
    else:
        net.eval()

    correct_per_class = defaultdict(list)

    if mode == 'train':
        #print("epoch in for")
        #print("Device during train epoch:", args.device)
        for i_batch, datum in enumerate(dataloader):
            # 加载数据并转移到 GPU
            img = datum[0].float().to(args.device)
            if 'Video' in args.model:
                img = img[:, :, :, 24:-24, 24:-24]
            img = (img - img.mean()) / img.std()
            lab = datum[1].long().to(args.device)
            n_b = lab.shape[0]

            # 模型前向计算
            output = net(img)
            loss = criterion(output, lab)

            ###### 使用 PyTorch 替代 NumPy #####
            # 计算 Top-1 准确率
            preds = torch.argmax(output, dim=-1)  # 获取预测类别
            matched = preds == lab  # 检查预测是否正确
            acc = torch.sum(matched).item()  # Top-1 正确数量

            # 计算 Top-5 准确率
            top5_preds = torch.topk(output, k=5, dim=-1).indices  # 获取 Top-5 预测类别
            top5_matched = (top5_preds == lab.unsqueeze(1)).any(dim=-1)  # 检查 Top-5 是否匹配
            top5_acc = torch.sum(top5_matched).item()  # Top-5 正确数量

            # 按类别统计正确率
            for y, c in zip(lab.tolist(), matched.tolist()):  # 将张量转换为 Python 列表
                correct_per_class[y].append(c)

            ###### 累加统计 #####
            loss_avg += loss.item() * n_b
            acc_avg += acc
            num_exp += n_b
            top5_acc_avg += top5_acc

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # for i_batch, datum in enumerate(dataloader):
        #     img = datum[0].float().to(args.device)
        #     if 'Video' in args.model:
        #         img = img[:,:, :, 24:-24,24:-24]
        #     img = (img - img.mean()) / img.std()
        #     lab = datum[1].long().to(args.device)
        #     n_b = lab.shape[0]

        #     output = net(img)
        #     loss = criterion(output, lab)
        #     matched = np.equal(np.argmax(output.cpu().data.numpy(), axis=-1), lab.cpu().data.numpy())
        #     acc = np.sum(matched)
        #     top5_preds = np.argsort(output.cpu().data.numpy(), axis=-1)[:, -5:]
        #     top5_matched = np.array([lab.cpu().data.numpy()[i] in top5_preds[i] for i in range(len(lab))])
        #     top5_acc = np.sum(top5_matched)
        #     #print("epoch in forrrrr")
        #     for y, c in zip(lab.cpu().tolist(), matched.tolist()):
        #         correct_per_class[y].append(c)
        #     #print("end epoch in forrrrr")
        #     loss_avg += loss.item()*n_b
        #     acc_avg += acc
        #     num_exp += n_b
        #     top5_acc_avg += top5_acc

        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()
            #print("end epoch in for")
    else :
        #print("test epoch in for")
        #print("Device during test epoch:", args.device)
        for j_ in range(3):
            for i_batch, datum in enumerate(dataloader):
                img = datum[0].float().to(args.device)
                if 'Video' in args.model:
                    img = img[:, :, :, 24:-24, 24:-24]
                img = (img - img.mean()) / img.std()
                lab = datum[1].long().to(args.device)
                n_b = lab.shape[0]

                # 模型前向计算
                output = net(img)
                loss = criterion(output, lab)

                ###### 自定义加速 #####
                # 计算 Top-1 准确率
                preds = torch.argmax(output, dim=-1)  # 获取预测类别
                acc = torch.sum(preds == lab).item()  # 计算 Top-1 正确的数量

                # 计算 Top-1 匹配
                top1_preds = torch.topk(output, k=1, dim=-1).indices  # 获取 Top-1 预测
                top1_matched = (top1_preds == lab.unsqueeze(1)).any(dim=-1)  # 检查是否匹配
                top1_acc = torch.sum(top1_matched).item()

                # 计算 Top-3 匹配
                top3_preds = torch.topk(output, k=3, dim=-1).indices  # 获取 Top-3 预测
                top3_matched = (top3_preds == lab.unsqueeze(1)).any(dim=-1)  # 检查是否匹配
                top3_acc = torch.sum(top3_matched).item()

                # 计算 Top-5 匹配
                top5_preds = torch.topk(output, k=5, dim=-1).indices  # 获取 Top-5 预测
                top5_matched = (top5_preds == lab.unsqueeze(1)).any(dim=-1)  # 检查是否匹配
                top5_acc = torch.sum(top5_matched).item()

                # 按类别统计正确率
                for y, c in zip(lab.tolist(), (preds == lab).tolist()):
                    correct_per_class[y].append(c)

                ###### 统计指标 #####
                loss_avg += loss.item() * n_b
                acc_avg += acc
                top5_acc_avg += top5_acc
                top3_acc_avg += top3_acc
                top1_acc_avg += top1_acc
                num_exp += n_b

        # for j_ in range(3):
        #     for i_batch, datum in enumerate(dataloader):
        #         img = datum[0].float().to(args.device)
        #         if 'Video' in args.model:
        #             img = img[:,:, :, 24:-24,24:-24]
        #         img = (img - img.mean()) / img.std()
        #         lab = datum[1].long().to(args.device)
        #         n_b = lab.shape[0]
        #         #print("Image device:", img.device)
        #         #print("Model device:", next(net.parameters()).device)

        #         output = net(img)
        #         loss = criterion(output, lab)
        #         ######自定义加速#####
                
        #         ####################
        #         acc = np.sum(np.equal(np.argmax(output.cpu().data.numpy(), axis=-1), lab.cpu().data.numpy()))
        #         top1_preds = np.argsort(output.cpu().data.numpy(), axis=-1)[:, -1:]
        #         top1_matched = np.array([lab.cpu().data.numpy()[i] in top1_preds[i] for i in range(len(lab))])
        #         top1_acc = np.sum(top1_matched)
        #         top3_preds = np.argsort(output.cpu().data.numpy(), axis=-1)[:, -3:]
        #         top3_matched = np.array([lab.cpu().data.numpy()[i] in top3_preds[i] for i in range(len(lab))])
        #         top3_acc = np.sum(top3_matched)
        #         top5_preds = np.argsort(output.cpu().data.numpy(), axis=-1)[:, -5:]
        #         top5_matched = np.array([lab.cpu().data.numpy()[i] in top5_preds[i] for i in range(len(lab))])
        #         top5_acc = np.sum(top5_matched)
        #         #print("test epoch in forrrrr")
        #         for y, c in zip(lab.cpu().tolist(), np.equal(np.argmax(output.cpu().data.numpy(), axis=-1), lab.cpu().data.numpy()).tolist()):
        #             correct_per_class[y].append(c)
        #         #print("end test epoch in forrrrr")
        #         loss_avg += loss.item()*n_b
        #         acc_avg += acc
        #         top5_acc_avg += top5_acc
        #         top3_acc_avg += top3_acc
        #         top1_acc_avg += top1_acc
        #         num_exp += n_b
            #print("end epoch in for")    

    loss_avg /= num_exp
    acc_avg /= num_exp
    top5_acc_avg /= num_exp
    top3_acc_avg /= num_exp
    top1_acc_avg /= num_exp

    top_acc_avg = [acc_avg, top1_acc_avg, top3_acc_avg, top5_acc_avg]

    correct_per_class = dict(correct_per_class)
    correct_per_class = [
        np.mean(correct_per_class[i]) 
            if i in correct_per_class 
            else None 
        for i in range(len(correct_per_class))]

    if args.eval_mode == 'top5':
        return loss_avg, top_acc_avg, correct_per_class
    else:
        return loss_avg, acc_avg, correct_per_class


def evaluate_synset(it_eval, net, images_train, labels_train, testloader, args, mode='hallucinator', return_loss=False, test_freq=None):
    lr = float(args.lr_net)
    Epoch = int(args.epoch_eval_train)
    lr_schedule = [Epoch//2+1]#, 3*Epoch//4+1]
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    criterion = nn.CrossEntropyLoss().to(args.device)

    if mode == 'none':
        dst_train = TensorDataset(images_train, labels_train)
    elif mode == 'multi-static':
        dst_train = MultiStaticSharedDataset(images_train[0], images_train[1], images_train[2])
    else:
        raise NotImplementedError
    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_train, shuffle=True, num_workers=0)

    start = time.time()

    print("start ep")
    for ep in tqdm.tqdm(range(Epoch + 1)):
        #print("start train epoch")
        loss_train, acc_train, _= epoch('train', trainloader, net, optimizer, criterion, args)
        #print("end train epoch")
        if (test_freq is None and ep == Epoch) or (test_freq is not None and ep % test_freq == 0 and ep != 0):
            with torch.no_grad():
                #print("start test epoch")
                loss_test, acc_test, acc_per= epoch('test', testloader, net, optimizer, criterion, args)
                #print("end test epoch")
                if args.eval_mode != 'top5':
                    print('%s Evaluate_%02d: Ep %d time = %ds loss = %.6f train acc = %.2f, test acc = %.2f' % (get_time(), it_eval, ep, int(time.time() - start), loss_train, acc_train*100, acc_test*100))
                    #print('acc_per', acc_per)
        if ep in lr_schedule:
            lr *= 0.1
            print('lr = %.6f'%lr)
            optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
        if ep % 10 == 0 and args.eval_mode == 'test':
            print("Epoch: %d, loss: %.6f, acc_train: %.2f" % (ep, loss_train, acc_train*100))
    #print("end ep")
    time_train = time.time() - start
    if args.eval_mode != 'top5':
        print('%s Evaluate_%02d: Ep %d time = %ds loss = %.6f train acc = %.2f, test acc = %.2f' % (get_time(), it_eval, Epoch, int(time_train), loss_train, acc_train*100, acc_test*100))

    if mode == 'none' or mode == 'hallucinator' or mode == 'multi-static' or mode =='S1D1':
        return net, acc_train, acc_test, acc_per
    return net, acc_train, acc_test



def augment(images, dc_aug_param, device):
    # This can be sped up in the future.

    if dc_aug_param != None and dc_aug_param['strategy'] != 'none':
        scale = dc_aug_param['scale']
        crop = dc_aug_param['crop']
        rotate = dc_aug_param['rotate']
        noise = dc_aug_param['noise']
        strategy = dc_aug_param['strategy']

        shape = images.shape
        mean = []
        for c in range(shape[1]):
            mean.append(float(torch.mean(images[:,c])))

        def cropfun(i):
            im_ = torch.zeros(shape[1],shape[2]+crop*2,shape[3]+crop*2, dtype=torch.float, device=device)
            for c in range(shape[1]):
                im_[c] = mean[c]
            im_[:, crop:crop+shape[2], crop:crop+shape[3]] = images[i]
            r, c = np.random.permutation(crop*2)[0], np.random.permutation(crop*2)[0]
            images[i] = im_[:, r:r+shape[2], c:c+shape[3]]

        def scalefun(i):
            h = int((np.random.uniform(1 - scale, 1 + scale)) * shape[2])
            w = int((np.random.uniform(1 - scale, 1 + scale)) * shape[2])
            tmp = F.interpolate(images[i:i + 1], [h, w], )[0]
            mhw = max(h, w, shape[2], shape[3])
            im_ = torch.zeros(shape[1], mhw, mhw, dtype=torch.float, device=device)
            r = int((mhw - h) / 2)
            c = int((mhw - w) / 2)
            im_[:, r:r + h, c:c + w] = tmp
            r = int((mhw - shape[2]) / 2)
            c = int((mhw - shape[3]) / 2)
            images[i] = im_[:, r:r + shape[2], c:c + shape[3]]

        def rotatefun(i):
            im_ = scipyrotate(images[i].cpu().data.numpy(), angle=np.random.randint(-rotate, rotate), axes=(-2, -1), cval=np.mean(mean))
            r = int((im_.shape[-2] - shape[-2]) / 2)
            c = int((im_.shape[-1] - shape[-1]) / 2)
            images[i] = torch.tensor(im_[:, r:r + shape[-2], c:c + shape[-1]], dtype=torch.float, device=device)

        def noisefun(i):
            images[i] = images[i] + noise * torch.randn(shape[1:], dtype=torch.float, device=device)


        augs = strategy.split('_')

        for i in range(shape[0]):
            choice = np.random.permutation(augs)[0] # randomly implement one augmentation
            if choice == 'crop':
                cropfun(i)
            elif choice == 'scale':
                scalefun(i)
            elif choice == 'rotate':
                rotatefun(i)
            elif choice == 'noise':
                noisefun(i)

    return images



def get_daparam(dataset, model, model_eval, ipc):
    # We find that augmentation doesn't always benefit the performance.
    # So we do augmentation for some of the settings.

    dc_aug_param = dict()
    dc_aug_param['crop'] = 4
    dc_aug_param['scale'] = 0.2
    dc_aug_param['rotate'] = 45
    dc_aug_param['noise'] = 0.001
    dc_aug_param['strategy'] = 'none'

    if dataset == 'MNIST':
        dc_aug_param['strategy'] = 'crop_scale_rotate'

    if model_eval in ['ConvNetBN']: # Data augmentation makes model training with Batch Norm layer easier.
        dc_aug_param['strategy'] = 'crop_noise'

    return dc_aug_param


def get_eval_pool(eval_mode, model, model_eval):
    if eval_mode == 'M': # multiple architectures
        model_eval_pool = ['MLP', 'ConvNet', 'LeNet', 'AlexNet', 'VGG11', 'ResNet18']
    elif eval_mode == 'B':  # multiple architectures with BatchNorm for DM experiments
        model_eval_pool = ['ConvNetBN', 'ConvNetASwishBN', 'AlexNetBN', 'VGG11BN', 'ResNet18BN']
    elif eval_mode == 'W': # ablation study on network width
        model_eval_pool = ['ConvNetW32', 'ConvNetW64', 'ConvNetW128', 'ConvNetW256']
    elif eval_mode == 'D': # ablation study on network depth
        model_eval_pool = ['ConvNetD1', 'ConvNetD2', 'ConvNetD3', 'ConvNetD4']
    elif eval_mode == 'A': # ablation study on network activation function
        model_eval_pool = ['ConvNetAS', 'ConvNetAR', 'ConvNetAL', 'ConvNetASwish']
    elif eval_mode == 'P': # ablation study on network pooling layer
        model_eval_pool = ['ConvNetNP', 'ConvNetMP', 'ConvNetAP']
    elif eval_mode == 'N': # ablation study on network normalization layer
        model_eval_pool = ['ConvNetNN', 'ConvNetBN', 'ConvNetLN', 'ConvNetIN', 'ConvNetGN']
    elif eval_mode == 'S': # itself
        if 'BN' in model:
            print('Attention: Here I will replace BN with IN in evaluation, as the synthetic set is too small to measure BN hyper-parameters.')
        model_eval_pool = [model[:model.index('BN')]] if 'BN' in model else [model]
    elif eval_mode == 'SS':  # itself
        model_eval_pool = [model]
    else:
        model_eval_pool = [model_eval]
    return model_eval_pool


class ParamDiffAug():
    def __init__(self):
        self.aug_mode = 'S' #'multiple or single'
        self.prob_flip = 0.5
        self.ratio_scale = 1.2
        self.ratio_rotate = 15.0
        self.ratio_crop_pad = 0.125
        self.ratio_cutout = 0.5 # the size would be 0.5x0.5
        self.brightness = 1.0
        self.saturation = 2.0
        self.contrast = 0.5


def set_seed_DiffAug(param):
    if param.latestseed == -1:
        return
    else:
        torch.random.manual_seed(param.latestseed)
        param.latestseed += 1


def DiffAugment(x, strategy='', seed = -1, param = None):
    if strategy == 'None' or strategy == 'none' or strategy == '':
        return x

    if seed == -1:
        param.Siamese = False
    else:
        param.Siamese = True

    param.latestseed = seed

    if strategy:
        if param.aug_mode == 'M': # original
            for p in strategy.split('_'):
                for f in AUGMENT_FNS[p]:
                    x = f(x, param)
        elif param.aug_mode == 'S':
            pbties = strategy.split('_')
            set_seed_DiffAug(param)
            p = pbties[torch.randint(0, len(pbties), size=(1,)).item()]
            for f in AUGMENT_FNS[p]:
                x = f(x, param)
        else:
            exit('unknown augmentation mode: %s'%param.aug_mode)
        x = x.contiguous()
    return x


# We implement the following differentiable augmentation strategies based on the code provided in https://github.com/mit-han-lab/data-efficient-gans.
def rand_scale(x, param):
    # x>1, max scale
    # sx, sy: (0, +oo), 1: orignial size, 0.5: enlarge 2 times
    ratio = param.ratio_scale
    set_seed_DiffAug(param)
    sx = torch.rand(x.shape[0]) * (ratio - 1.0/ratio) + 1.0/ratio
    set_seed_DiffAug(param)
    sy = torch.rand(x.shape[0]) * (ratio - 1.0/ratio) + 1.0/ratio
    theta = [[[sx[i], 0,  0],
            [0,  sy[i], 0],] for i in range(x.shape[0])]
    theta = torch.tensor(theta, dtype=torch.float)
    if param.Siamese: # Siamese augmentation:
        theta[:] = theta[0].clone()
    grid = F.affine_grid(theta, x.shape).to(x.device)
    x = F.grid_sample(x, grid)
    return x


def rand_rotate(x, param): # [-180, 180], 90: anticlockwise 90 degree
    ratio = param.ratio_rotate
    set_seed_DiffAug(param)
    theta = (torch.rand(x.shape[0]) - 0.5) * 2 * ratio / 180 * float(np.pi)
    theta = [[[torch.cos(theta[i]), torch.sin(-theta[i]), 0],
        [torch.sin(theta[i]), torch.cos(theta[i]),  0],]  for i in range(x.shape[0])]
    theta = torch.tensor(theta, dtype=torch.float)
    if param.Siamese: # Siamese augmentation:
        theta[:] = theta[0].clone()
    grid = F.affine_grid(theta, x.shape).to(x.device)
    x = F.grid_sample(x, grid)
    return x


def rand_flip(x, param):
    prob = param.prob_flip
    set_seed_DiffAug(param)
    randf = torch.rand(x.size(0), 1, 1, 1, device=x.device)
    if param.Siamese: # Siamese augmentation:
        randf[:] = randf[0].clone()
    return torch.where(randf < prob, x.flip(3), x)


def rand_brightness(x, param):
    ratio = param.brightness
    set_seed_DiffAug(param)
    randb = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.Siamese:  # Siamese augmentation:
        randb[:] = randb[0].clone()
    x = x + (randb - 0.5)*ratio
    return x


def rand_saturation(x, param):
    ratio = param.saturation
    x_mean = x.mean(dim=1, keepdim=True)
    set_seed_DiffAug(param)
    rands = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.Siamese:  # Siamese augmentation:
        rands[:] = rands[0].clone()
    x = (x - x_mean) * (rands * ratio) + x_mean
    return x


def rand_contrast(x, param):
    ratio = param.contrast
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    set_seed_DiffAug(param)
    randc = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.Siamese:  # Siamese augmentation:
        randc[:] = randc[0].clone()
    x = (x - x_mean) * (randc + ratio) + x_mean
    return x


def rand_crop(x, param):
    # The image is padded on its surrounding and then cropped.
    ratio = param.ratio_crop_pad
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    set_seed_DiffAug(param)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    set_seed_DiffAug(param)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    if param.Siamese:  # Siamese augmentation:
        translation_x[:] = translation_x[0].clone()
        translation_y[:] = translation_y[0].clone()
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
    return x


def rand_cutout(x, param):
    ratio = param.ratio_cutout
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    set_seed_DiffAug(param)
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    set_seed_DiffAug(param)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    if param.Siamese:  # Siamese augmentation:
        offset_x[:] = offset_x[0].clone()
        offset_y[:] = offset_y[0].clone()
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x


AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'crop': [rand_crop],
    'cutout': [rand_cutout],
    'flip': [rand_flip],
    'scale': [rand_scale],
    'rotate': [rand_rotate],
}


class Conv3DNet(nn.Module):
    def __init__(self, in_channel=4, mid_channel=3, out_channel=3, img_size=112, kernel_size=3, mode='concat'):
        super().__init__()
        self.mode = mode
        if mode == 'add':
            in_channel = 3
        self.encoder = nn.Conv3d(in_channel, mid_channel, kernel_size, padding=1)

    def forward(self, static, dynamic):
        b, f, _, h, w = dynamic.shape # bz, 16, 1, 112, 112
        static = static.repeat(f, 1, 1, 1, 1).permute(1, 2, 0, 3, 4) #bz, 3, 16, h, w
        dynamic = dynamic.permute(0, 2, 1, 3, 4) #bz, 1, 16, h, w
        if self.mode == 'concat':
            x = torch.cat([static, dynamic], dim=1) #bz, 4, f, h, w
        elif self.mode == 'add':
            x = static + dynamic #bz, 3, f, h, w
        else:
            raise NotImplementedError
        x = self.encoder(x)
        return x.permute(0, 2, 1, 3, 4) 
    

def interpolate_models(model_init, model_final, model_interval, a=0, b=1):
    """
    在两个模型之间进行插值，得到新的模型。
    
    :param model_init: 随机初始化的模型
    :param model_final: 预训练的模型
    :param model_interval: 插值后的新模型
    :param a: 插值范围下界
    :param b: 插值范围上界
    :return: 插值后的模型
    """
    # 生成插值系数 l 在 [a, b] 区间内
    l = (b - a) * torch.rand(1).item() + a
    
    # 对每一层的参数进行插值
    for model_interval_param, model_init_param, model_final_param in zip(
        model_interval.parameters(),
        model_init.parameters(),
        model_final.parameters(),
    ):
        model_interval_param.data.copy_(
            l * model_init_param.data + (1 - l) * model_final_param.data
        )
    
    return model_interval


class SampleNet(nn.Module):
    """
    TNet module for adversarial networks with fixed activation layers and predefined parameters.
    """

    def __init__(self, feature_dim=64, t_batchsize=64, t_var=1):
        super(SampleNet, self).__init__()
        self.feature_dim = feature_dim  # Feature dimension
        #####################
        #####     Q1    #####
        #####################
        self.t_sigma_num = t_batchsize // 16  # Number of sigmas for t_net
        self._input_adv_t_net_dim = feature_dim  # Input noise dimension
        self._input_t_dim = feature_dim  # t_net input dimension
        self._input_t_batchsize = t_batchsize  # Batch size
        self._input_t_var = t_var  # Variance of input noise

        # Fixed activation layers
        self.activation_1 = nn.LeakyReLU(negative_slope=0.2)
        self.activation_2 = nn.Tanh()

        # Create a simple 3-layer fully connected network using fixed activation layers
        self.t_layers_list = nn.ModuleList()
        ch_in = self.feature_dim
        num_layer = 3
        for i in range(num_layer):
            self.t_layers_list.append(nn.Linear(ch_in, ch_in))
            self.t_layers_list.append(nn.BatchNorm1d(ch_in))
            # Use activation_1 for the first two layers, and activation_2 for the last layer
            self.t_layers_list.append(
                self.activation_1 if i < (num_layer - 1) else self.activation_2
            )

    def forward(self, device):
        # Generate white noise
        if self.t_sigma_num > 0:
            # Initialize the white noise input
            self._t_net_input = torch.randn(
                self.t_sigma_num, self._input_adv_t_net_dim
            ) * (self._input_t_var**0.5)
            #self._t_net_input = self._t_net_input.to(device).detach()
            self._t_net_input = self._t_net_input.to(device)

            # Forward pass
            a = self._t_net_input
            for layer in self.t_layers_list:
                a = layer(a)

            a = a.repeat(int(self._input_t_batchsize / self.t_sigma_num), 1)

            # Generate the final t value
            # self._t = torch.randn(self._input_t_batchsize, self._input_t_dim) * ((self._input_t_var / self._input_t_dim) ** 0.5)
            # self._t = self._t.to(device).detach()
            self._t = a
        else:
            # When t_sigma_num = 0, generate standard Gaussian noise as t
            self._t = torch.randn(self._input_t_batchsize, self._input_t_dim) * (
                (self._input_t_var / self._input_t_dim) ** 0.5
            )
            self._t = self._t.to(device)
            #self._t = self._t.to(device).detach()
        return self._t



def sync_distributed_metric(metric, mode="avg"):
    # if dist.is_initialized():
    #     device = torch.device(f'cuda:{dist.get_rank()}' if torch.cuda.is_available() else 'cpu')
    # else:
    #     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device(f'cuda:{dist.get_rank()}' if torch.cuda.is_available() else 'cpu')
    if isinstance(metric, list):
        # Convert metric to tensor if it isn't already
        metric_tensors = [torch.tensor(m, device=device) if not isinstance(m, torch.Tensor) else m for m in metric]
        # Use all_reduce to synchronize each tensor across ranks
        for m in metric_tensors:
            dist.all_reduce(m, op=dist.ReduceOp.SUM)
        # Return average for each metric if mode is "avg", otherwise return the sum
        if mode == "avg":
            return [m.item() / dist.get_world_size() for m in metric_tensors]
        else:
            return [m.item() for m in metric_tensors]
    else:
        # Single metric
        if not isinstance(metric, torch.Tensor):
            metric = torch.tensor(metric, device=device)
        # Use all_reduce to synchronize the metric
        dist.all_reduce(metric, op=dist.ReduceOp.SUM)
        # Return the average value if mode is "avg", otherwise return the sum
        return metric.item() / dist.get_world_size() if mode == "avg" else metric.item()


def interloss(img_syn, label_syn, trained_model):
    #logits = trained_model(img_syn, return_features=False)
    logits = trained_model(img_syn,return_features=False)
    #print(logits)
    #print(type(logits))
    #print(label_syn.shape)
    #print("label_syn",label_syn)
    #print(logits.shape)
    
    loss = F.cross_entropy(logits, label_syn)
    return loss



class R2Plus1D_Custom(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super(R2Plus1D_Custom, self).__init__()
        
        # 根据是否预训练加载模型
        self.model = models.r2plus1d_18(pretrained=pretrained)
        self.model.eval()

        # 修改分类层
        self.model.fc = nn.Linear(512, num_classes)
    
    def forward(self, x, return_features=False):
        self.model.eval()
        if return_features:
            # 提取 fc 之前的 512 维特征
            x = self.model.stem(x)
            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)
            x = self.model.avgpool(x)
            x = torch.flatten(x, 1)  # (B, 512)
            return x
        else:
            # 直接分类
            return self.model(x)
    
def preload_dataset(dataset, batch_size=64, num_workers=0):
    default_num_workers = max(0, os.cpu_count() - 1)  # 自动设置为 CPU 核心数 - 1
    num_workers = default_num_workers  # 你也可以手动覆盖这个值
    logger.info("Preloading dataset...")
    
    # 使用 DataLoader 并行加载数据
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    # 注意，使用DataLoader加载数据之后，DataLoader中的元素会多一维Batch出来，因此最后使用cat而不是stack
    
    video_all = []
    label_all = []
    
    # 遍历 DataLoader 并收集所有数据
    for batch in tqdm.tqdm(dataloader, desc="Loading data"):
        videos, labels = batch
        video_all.append(videos)
        label_all.append(labels)
    
    # 将所有数据堆叠到一起
    video_all = torch.cat(video_all, dim=0)
    label_all = torch.cat(label_all, dim=0)
    
    # 返回一个新的 TensorDataset
    return TensorDataset(video_all, label_all), video_all, label_all

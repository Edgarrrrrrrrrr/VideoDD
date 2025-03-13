import os
import time
import random
import csv
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tdata
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import wandb


# ---------------------------
# Tool Function
# ---------------------------
def get_time():
    return str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))


def get_dataset(dataset, data_path, batch_size=128, num_workers=8, preload=False):
    assert dataset == 'HMDB51', "Only HMDB51 is supported in this baseline implementation."

    # HMDB51 data preprocessing
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])


    train_set = HMDB51(path=data_path, split="train", transform=transform, preload=preload)
    test_set = HMDB51(path=data_path, split="test", transform=transform, preload=preload)

    print(f"Loaded HMDB51: {len(train_set)} train samples, {len(test_set)} test samples.")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True, prefetch_factor=4, persistent_workers=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                             pin_memory=True, prefetch_factor=4, persistent_workers=True)
    return train_loader, test_loader


def evaluate_synset(it_eval, net, train_loader, test_loader, args, mode='none'):
    assert mode == 'none', "Only full dataset training is supported in this baseline."

    device = args.device
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(net.parameters(), lr=args.lr_net, momentum=0.9, weight_decay=0.0005)

    total_train_samples = len(train_loader.dataset)
    total_test_samples = len(test_loader.dataset)

    # Training and evaluation
    for epoch in range(args.epoch_eval_train):
        net.train()
        train_loss = 0.0
        correct_train, total_train = 0, 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epoch_eval_train}", leave=False):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            correct_train += predicted.eq(labels).sum().item()
            total_train += labels.size(0)

        avg_train_loss = train_loss / total_train
        train_acc = correct_train / total_train

        net.eval()
        test_loss = 0.0
        correct_test, total_test = 0, 0
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Evaluating", leave=False):
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * labels.size(0)
                _, predicted = outputs.max(1)
                correct_test += predicted.eq(labels).sum().item()
                total_test += labels.size(0)
        avg_test_loss = test_loss / total_test
        test_acc = correct_test / total_test

        wandb.log({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'train_acc': train_acc,
            'test_loss': avg_test_loss,
            'test_acc': test_acc
        })

        print(
            f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Train Acc = {train_acc * 100:.2f}% | Test Loss = {avg_test_loss:.4f}, Test Acc = {test_acc * 100:.2f}%")

    return net, train_acc, test_acc


def get_network(model, num_classes, im_size=(112, 112), frames=16):
    assert model == 'ConvNet3D', "Only ConvNet3D is supported in this baseline implementation."
    net_width = 128  
    net_depth = 3  
    net_act = 'relu'  
    net_norm = 'instancenorm'  
    net_pooling = 'avgpooling'  
    net = ConvNet3D(
        channel=3, num_classes=num_classes,
        im_size=im_size, frames=frames, net_width=net_width, net_depth=net_depth,
        net_act=net_act, net_norm=net_norm,
        net_pooling=net_pooling
    )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = net.to(device)
    print(f"Using {model} model for training.")
    return net


# ---------------------------
# Dataset class definition
# ---------------------------
NUM_FRAMES = 16
FRAME_GAP = 4


class HMDB51(tdata.Dataset):
    def __init__(self, path, split, transform, preload=False):
        
        self.data_path = os.path.join(path, "jpegs_112")
        begin_frame, end_frame, skip_frame = 1, 24, 3
        self.frames = list(range(begin_frame, end_frame, skip_frame))

        self.transform = transform
        self.split = split
        self.preload = preload  

        csv_path = os.path.join(path, "hmdb51_splits.csv")
        self.video_dirs = []
        self.label_strs = []
        self.class_strs = set()

        with open(csv_path) as fp:
            reader = csv.DictReader(fp)
            for item in reader:
                if item["split"] != split:
                    continue
                name = item["folder_name"]
                sample_dir = os.path.join(self.data_path, name)
                if not os.path.isdir(sample_dir):
                    print(f"Warning: Folder {sample_dir} not found, skipping this sample.")
                    continue
                self.label_strs.append(item["label"])
                self.class_strs.add(item["label"])
                self.video_dirs.append(sample_dir)

        self.class_strs = sorted(self.class_strs)
        self.class_2_idx = {x: i for i, x in enumerate(self.class_strs)}
        self.labels = [self.class_2_idx[l] for l in self.label_strs]
        self.targets = self.labels  # alias

        length = len(self.video_dirs)
        self.start = [-1 for i in range(length)]

        if self.preload:
            print("Preloading dataset images into memory...")
            self.cache = {}  
            for video_dir in self.video_dirs:
                frame_images = []
                frame_files = sorted([f for f in os.listdir(video_dir) if f.endswith('.jpg')])
                for fname in frame_files:
                    img_path = os.path.join(video_dir, fname)
                    image = Image.open(img_path).convert('RGB').copy()
                    frame_images.append(image)
                self.cache[video_dir] = frame_images

    def __len__(self):
        return len(self.video_dirs)

    def read_images(self, path, use_transform):
        X = []
        flip = (random.random() > 0.5)
        for i in self.frames:
            if self.preload:
                image = self.cache[path][i - 1]  
            else:
                image_path = os.path.join(path, "frame{:06d}.jpg".format(i))
                image = Image.open(image_path).convert('RGB')
            if flip:
                image = transforms.functional.hflip(image)
            if use_transform is not None:
                image = use_transform(image)
            X.append(image)
        X = torch.stack(X, dim=0)
        return X

    def __getitem__(self, index):
        path = self.video_dirs[index]
        label = self.labels[index]
        length = len(os.listdir(path))
        if length < NUM_FRAMES * FRAME_GAP:
            skip = length // NUM_FRAMES
        else:
            skip = FRAME_GAP
        if self.start[index] == -1 or self.split == "test":
            self.start[index] = random.randint(1, max(1, length - (NUM_FRAMES - 1) * skip))
        self.frames = list(range(self.start[index], self.start[index] + NUM_FRAMES * skip, skip))
        X = self.read_images(path, self.transform)
        return X, label

    def get_all_frames(self, index):
        X = []
        path = self.video_dirs[index]
        if self.preload:
            for image in self.cache[path]:
                X.append(self.transform(image))
        else:
            length = len(os.listdir(path))
            for i in range(1, length + 1):
                image_path = os.path.join(path, "frame{:06d}.jpg".format(i))
                image = Image.open(image_path).convert('RGB')
                image = self.transform(image)
                X.append(image)
        X = torch.stack(X, dim=0)
        return X, len(X)


# ---------------------------
# ConvNet3D
# ---------------------------
class Swish(nn.Module):  # Swish(x) = x * σ(x)
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.sigmoid(input)


class ConvNet3D(nn.Module):
    def __init__(self, channel, num_classes, net_width, net_depth, net_act, net_norm, net_pooling, frames,
                 im_size=(32, 32), dropout_keep_prob=0.5):
        super(ConvNet3D, self).__init__()
        self.features, shape_feat = self._make_layers(channel, net_width, net_depth, net_norm, net_act, net_pooling,
                                                      im_size, frames)
        num_feat = shape_feat[0] * shape_feat[1] * shape_feat[2] * shape_feat[3]
        self.avg_pool = nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(1, 1, 1)) if (im_size[0] > 64) else nn.AvgPool3d(
            kernel_size=(2, 1, 1), stride=(1, 1, 1))
        self.dropout = nn.Dropout(dropout_keep_prob)
        self.logit = nn.Conv3d(net_width, num_classes, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=True)

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        out = self.features(x)
        out = self.logit(self.dropout(self.avg_pool(out)))
        logits = out.squeeze(3).squeeze(3)
        logits = torch.max(logits, 2)[0]
        return logits

    def embed(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        out = self.features(x)
        out = out.view(out.size(0), -1)
        return out

    def _get_activation(self, net_act):
        if net_act == 'sigmoid':
            return nn.Sigmoid()
        elif net_act == 'relu':
            return nn.ReLU(inplace=True)
        elif net_act == 'leakyrelu':
            return nn.LeakyReLU(negative_slope=0.01)
        elif net_act == 'swish':
            return Swish()
        else:
            exit('unknown activation function: %s' % net_act)

    def _get_pooling(self, net_pooling, flag):
        if net_pooling == 'maxpooling':
            if flag == 1:
                return nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
            else:
                return nn.MaxPool3d(kernel_size=2, stride=2)
        elif net_pooling == 'avgpooling':
            return nn.AvgPool3d(kernel_size=2, stride=2)
        elif net_pooling == 'none':
            return None
        else:
            exit('unknown net_pooling: %s' % net_pooling)

    def _get_normlayer(self, net_norm, shape_feat):
        if net_norm == 'batchnorm':
            return nn.BatchNorm3d(shape_feat[0], affine=True)
        elif net_norm == 'layernorm':
            return nn.LayerNorm(shape_feat, elementwise_affine=True)
        elif net_norm == 'instancenorm':
            return nn.GroupNorm(shape_feat[0], shape_feat[0], affine=True)
        elif net_norm == 'groupnorm':
            return nn.GroupNorm(4, shape_feat[0], affine=True)
        elif net_norm == 'none':
            return None
        else:
            exit('unknown net_norm: %s' % net_norm)

    def _make_layers(self, channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size, frames):
        layers = []
        in_channels = channel
        if im_size[0] == 28:
            im_size = (32, 32)
        shape_feat = [in_channels, frames, im_size[0], im_size[1]]
        for d in range(net_depth):
            layers += [nn.Conv3d(in_channels, 64 if d == 0 else net_width,
                                 kernel_size=(3, 7, 7), padding=(1, 3, 3), stride=(1, 2, 2))]
            shape_feat[2] //= 2
            shape_feat[3] //= 2
            shape_feat[0] = 64 if d == 0 else net_width
            if net_norm != 'none':
                layers += [self._get_normlayer(net_norm, shape_feat)]
            layers += [self._get_activation(net_act)]
            in_channels = shape_feat[0]
            if net_pooling != 'none':
                layers += [self._get_pooling(net_pooling, 1 if d == 0 else 0)]
                if d != 0:
                    shape_feat[1] //= 2
                shape_feat[2] //= 2
                shape_feat[3] //= 2
        return nn.Sequential(*layers), shape_feat

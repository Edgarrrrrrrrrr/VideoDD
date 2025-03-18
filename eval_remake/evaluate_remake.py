# 重写的评估函数喵喵喵
# 原来的：def evaluate_synset(it_eval, net, images_train, labels_train, testloader, args, mode='hallucinator', return_loss=False, test_freq=None):
import time
import os
from loguru import logger
import torch
import torch.nn as nn
from .evaluate_loss import SoftCrossEntropy, HingeLoss

import torch.multiprocessing as mp
import torch.distributed as dist
    
# 初始化分布式环境
def setup(rank, world_size):
    dist.init_process_group(
        backend='nccl',  # 使用 NCCL 后端进行 GPU 通信
        init_method='tcp://localhost:12355',  # 初始化方法
        rank=rank,  # 当前进程的 rank
        world_size=world_size  # 总进程数
    )

# 清理分布式环境
def cleanup():
    dist.destroy_process_group()

# 定义需要在多进程中运行的函数
def train(rank, world_size):
    print(f"Training on GPU {rank} out of {world_size}")

# def train(rank, world_size, model, train_loader, test_loader, criterion, optimizer, scheduler, args):
#     """
#     rank: 当前进程的 rank
#     world_size: 总进程数
#     model: 传入的模型
#     train_loader: 训练集数据加载器
#     test_loader: 测试集数据加载器
#     criterion: 损失函数
#     optimizer: 优化器
#     scheduler: 学习率调度器
#     args: 其他参数
#     """
    # 初始化分布式环境
    # setup(rank, world_size)

    # # 设置当前进程的 GPU 设备
    # torch.cuda.set_device(rank)
    # device = torch.device(f"cuda:{rank}")
    # model = model.to(device)

    # # 包装模型为分布式模型
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    # # 设置训练数据的分布式采样器
    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_loader.dataset, num_replicas=world_size, rank=rank)
    # train_loader = torch.utils.data.DataLoader(
    #     train_loader.dataset,
    #     batch_size=args.batch_size,
    #     sampler=train_sampler,
    #     num_workers=args.num_workers,
    #     pin_memory=True
    # )

    # # 开始训练
    # for epoch in range(args.eval_epochs):
    #     model.train()
    #     train_sampler.set_epoch(epoch)  # 设置分布式采样器的 epoch，确保每个 epoch 数据不同
    #     running_loss = 0.0

    #     for batch_idx, (inputs, targets) in enumerate(train_loader):
    #         # 将数据移动到当前 GPU
    #         inputs, targets = inputs.to(device), targets.to(device)

    #         # 前向传播
    #         outputs = model(inputs)
    #         loss = criterion(outputs, targets)

    #         # 反向传播
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    #         # 统计损失
    #         running_loss += loss.item()

    #         # 打印日志（仅 rank=0 的进程打印）
    #         if rank == 0 and batch_idx % args.log_interval == 0:
    #             logger.info(f"Epoch [{epoch+1}/{args.eval_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    #     # 学习率调度器更新
    #     scheduler.step()

    #     # 打印 epoch 结束信息（仅 rank=0 的进程打印）
    #     if rank == 0:
    #         logger.info(f"Epoch [{epoch+1}/{args.eval_epochs}] completed. Average Loss: {running_loss / len(train_loader):.4f}")

    #     # 在测试集上评估（仅 rank=0 的进程评估）
    #     if rank == 0:
    #         evaluate(model, test_loader, criterion, device, epoch)

    # 清理分布式环境
    cleanup()

# 测试集评估函数
def evaluate(model, test_loader, criterion, device, epoch):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    logger.info(f"Epoch [{epoch+1}] Test Loss: {test_loss / len(test_loader):.4f}, Accuracy: {accuracy:.2f}%")


    




# 输入：args配置字典，模型，训练的Dataloader，测试的Dataloader，日志详细等级(可以不管)
def evaluate_synset_ours(args, model, train_loader, test_loader, log_level=1):
    # 配置日志保存到文件
    log_path = args.eval_log_path
    eval_logger = logger.add(log_path, format="{time} {level} {message}", level="INFO")
    if log_level==1:
        logger.info("开始进行评估，评估日志将会保存在", str(log_path))
        logger.info("评估数据质量使用的模型："+str(args.model)+" 使用的损失函数是"+str(args.eval_criterion)+" 使用的优化器是"+str(args.eval_optimizer))
    
    # 如果使用软标签
    if args.softlabel:
        pass
    # 使用硬标签
    #选择Loss喵
    if args.eval_criterion.lower() == 'crossentropyloss':
        criterion = nn.CrossEntropyLoss().cuda() #传入的是logits，而不要是概率分布
    elif args.eval_criterion.lower() == 'hingeloss':
        criterion = HingeLoss
    else:
        logger.error("非常抱歉我们目前不支持您的"+str(args.eval_criterion)+"用于评估喔，请等待后续QAQ")
    #选择优化器喵
    if args.eval_optimizer.lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.eval_lr, momentum=args.eval_momentum)
    elif args.eval_optimizer.lower() == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.eval_lr)
    else:
        logger.error("非常抱歉我们目前不支持您的"+str(args.eval_optimizer)+"用于评估喔，请等待后续QAQ")
    #选择lr-scheduler喵    
    if args.eval_lr_scheduler_slices == 1:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
         optimizer, milestones=[args.eval_epochs//2], gamma=0.1)
    elif args.eval_lr_scheduler_slices == 5:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[args.eval_epochs // 5,2*args.eval_epochs // 5,3*args.eval_epochs // 5,4*args.eval_epochs // 5], gamma=0.5
    )
    else:
        logger.error("非常抱歉我们目前不支持您的"+str(args.eval_lr_scheduler_slices)+"用于评估喔，请等待后续QAQ")

    #开启多进程计算喔喵
    print(f"Available GPUs: {torch.cuda.device_count()}")
    world_size = torch.cuda.device_count()  # 获取 GPU 数量
    if log_level == 1:
        logger.info(f"那么现在开始使用多GPU进行计算了喔喵，您当前将使用{world_size}个GPU并行训练喔喵")
    # 启动多进程训练
    
    mp.spawn(train, args=(world_size,), nprocs=world_size)
    # mp.spawn(
    #     train,
    #     args=(world_size, model, train_loader, test_loader, criterion, optimizer, scheduler, args),
    #     nprocs=8,
    #     join=True
    # )
    logger.remove(eval_logger)


# def cat():
# cat_frames = [
#     r'''
#        /\_/\  
#       ( o.o ) 
#        > ^ <
#     ''',
#     r'''
#        /\_/\  
#       ( -.- ) 
#        > ^ <
#     ''',
#     r'''
#        /\_/\  
#       ( o.o ) 
#        > ^ <
#     '''
# ]

# while True:
#     for frame in cat_frames:
#         os.system('clear')  # 在 Linux 上清屏
#         print(frame)
#         time.sleep(0.5)

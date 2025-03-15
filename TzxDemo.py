import datetime
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datetime import datetime
import torchvision.utils
from tqdm import tqdm, trange
from utils import get_dataset, get_network, get_eval_pool, evaluate_synset, get_time, DiffAugment, TensorDataset,\
                    epoch, get_loops, match_loss, ParamDiffAug, Conv3DNet,interpolate_models,SampleNet,sync_distributed_metric,\
                    interloss,R2Plus1D_Custom,preload_dataset
from NCFM import innerloss,mutil_layer_innerloss,CFLossFunc,cf_loss
from ComputeLoss import compute_calib_loss
import wandb
import torchvision.models.video as models
import copy
import random
from reparam_module import ReparamModule
import warnings
import time
from loguru import logger

# 忽略所有警告
warnings.filterwarnings("ignore")

# 创建目录保存数据
save_dir = "exp_results"
os.makedirs(save_dir, exist_ok=True)


def main(args):
    

    # 假设视频的学习率、init noise 和 IPC 存储在 args 中
    video_lr = args.lr_video  # 获取视频学习率
    init_ways = args.init # 获取初始化噪声
    ipc = args.ipc  # 获取 IPC 值
    use_samplingnet = args.sampling_net  # 是否使用采样网络
    use_calib = args.iter_calib  # 是否使用校准
    
    
    use_interp = 0  # 是否使用插值网络去提取特征，0表示每次都是random的，而1表示有插值策略的

    # 生成唯一文件名，包含时间戳、学习率、init_noise、ipc、use_samplingnet、use_calib 和 use_interp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    loss_png_path = os.path.join(save_dir, f"loss_{timestamp}_loss_curve_lr{video_lr}_noise{init_ways}_ipc{ipc}_sampling{use_samplingnet}_calib{use_calib}_featureNet{use_interp}.png")
    match_loss_png_path = os.path.join(save_dir, f"match_{timestamp}_match_loss_curve_lr{video_lr}_noise{init_ways}_ipc{ipc}_sampling{use_samplingnet}_calib{use_calib}_featureNet{use_interp}.png")
    calib_loss_png_path = os.path.join(save_dir, f"calib_{timestamp}_calib_loss_curve_lr{video_lr}_noise{init_ways}_ipc{ipc}_sampling{use_samplingnet}_calib{use_calib}_featureNet{use_interp}.png")


    # 记录 loss 变化
    loss_history = []
    match_loss_history = []
    calib_loss_history = []

    
    
    ####################################
    #######   STEP1 处理数据    #########
    ####################################
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_it_pool = np.arange(0, args.Iteration + 1, args.eval_it).tolist()
    logger.info('Evaluation iterations: '+ str(eval_it_pool))

    ###################################################################
    ####TO DO  Trick 1 data augmentation 可以在内存循环中加这个trick ####
    ###################################################################
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader= get_dataset(args.dataset, args.data_path)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)


    if args.preload:
        logger.info("预加载训练数据集")
        dst_train,video_all,label_all = preload_dataset(dst_train)


    # record performances of all experiments
    accs_all_exps = dict()  
    for key in model_eval_pool:
        accs_all_exps[key] = []
    
    #usage WandB 
    project_name = "Baseline_{}".format(args.method)
    wandb.init(sync_tensorboard=False,
               project=project_name,
               job_type="CleanRepo",
               config=args,
               name = f'{args.dataset}_ipc{args.ipc}_{args.lr_video}_{timestamp}'
               )
    
    args = type('', (), {})()
    for key in wandb.config._items:
        setattr(args, key, wandb.config._items[key])
        
    if args.batch_syn is None:
        args.batch_syn = num_classes * args.ipc
    args.distributed = torch.cuda.device_count() > 1
    logger.info('Hyper-parameters: \n'+ str(args.__dict__))
    logger.info('Evaluation model pool: '+ str(model_eval_pool))


    ''' organize the real dataset '''
    logger.info("BUILDING DATASET")
    # labels_all is train data's label i.e gt label
    labels_all = label_all if args.preload else dst_train.labels

    
    # 索引列表，若三个类 则indices_class = [[], [], []]
    indices_class = [[] for c in range(num_classes)]
    for i, lab in tqdm(enumerate(labels_all)):
        indices_class[lab].append(i)
    labels_all = torch.tensor(labels_all, dtype=torch.long, device="cpu")




    ###############################################
    ####   TO DO  Trick 2 可学习的 soft label   ####
    ###############################################


    #(num_classes * args.ipc, args.frames, channel, h, w)
    video_syn = torch.randn(size=(num_classes*args.ipc, args.frames, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=args.device)
    #num_classes = 3，args.ipc = 2，则生成 [0, 0, 1, 1, 2, 2],现在是不可学习的hard label
    label_syn = torch.tensor(np.stack([np.ones(args.ipc)*i for i in range(0, num_classes)]), dtype=torch.long, requires_grad=False, device=args.device).view(-1)
    #syn_lr = torch.tensor(args.lr_teacher).to(args.device) if args.method == 'MTT' else None
    
    def get_videos(c, n):  # get random n videos from class c 这个函数有点儿慢啊
        #idx_shuffle = np.random.permutation(indices_class[c])[:n]
        idx_shuffle = np.random.choice(indices_class[c], size=n, replace=False)
        if n == 1:
            imgs = dst_train[idx_shuffle[0]][0].unsqueeze(0)
        else:
            imgs = torch.stack([dst_train[i][0] for i in idx_shuffle])
        return imgs.to(args.device)
    

    ##############################################################
    ######  TO DO trick3 cf methods has more ways to init   ######
    ##############################################################
    if args.init == 'real':
        logger.info('initialize synthetic data from random real video')
        for c in range(0, num_classes):
            i = c 
            video_syn.data[i*args.ipc:(i+1)*args.ipc] = get_videos(c, args.ipc).detach().data
    else:
        logger.info('initialize synthetic data from random noise')


    ####################################
    #######   STEP2 配置训练    #########
    ####################################

    # 调整维度为 [B, C, T, H, W]
    video_syn = video_syn.permute(0, 2, 1, 3, 4).contiguous().detach()
    video_syn.requires_grad_(True)  # 让它重新成为叶子张量
    print(video_syn.shape) #torch.Size([class*ipc, 3, 16, 112, 112])

    # syn_lr = syn_lr.detach().to(args.device).requires_grad_(args.train_lr) if args.method == 'MTT' else None
    # optimizer_lr = torch.optim.Adam([syn_lr], lr=args.lr_lr) if args.train_lr else None
    #optimizer_video = torch.optim.Adam([video_syn], lr=args.lr_video) # optimizer_video for synthetic data
    
    optimizer_video = torch.optim.SGD([video_syn], lr=args.lr_video, momentum=0.9)  # SGD optimizer for synthetic data
    optimizer_video.zero_grad()

    criterion = nn.CrossEntropyLoss().to(args.device)
    logger.info('training begins')

    best_acc = {m: 0 for m in model_eval_pool}
    best_std = {m: 0 for m in model_eval_pool}

    logger.info("蒸馏数据集的尺寸："+str(video_syn.shape))
    logger.info("蒸馏数据集标签的尺寸："+str(label_syn.shape))


    ###################################
    ######  TO DO more models    ######
    ###################################
    # 自己定义类，加载预训练model
    model_final = R2Plus1D_Custom(num_classes=num_classes, pretrained=True).to(args.device)
    model_init = R2Plus1D_Custom(num_classes=num_classes, pretrained=False).to(args.device)
    model_interval = R2Plus1D_Custom(num_classes=num_classes, pretrained=False).to(args.device)

    cfloss=CFLossFunc(alpha_for_loss=args.alpha_for_loss, beta_for_loss=args.beta_for_loss)
    
    if args.sampling_net:
        logger.info("use the sampling_net")
        ####这里的feature_dim是特征提取网络提取的feature维度
        sampling_net=SampleNet(feature_dim=512).to(args.device)
         # 定义优化器
        optimizer_sampling_net = torch.optim.SGD(sampling_net.parameters(), lr=args.lr_sampling_net)
        # 定义学习率调度器
        scheduler_sampling_net = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_sampling_net,
            mode='min',   # 监测的指标是最小化的
            factor=0.5,   # 学习率降低的倍数
            patience=500, # 如果500个epoch没有提升，就降低学习率
            verbose=False
        )
    else:
        sampling_net=None
        optimizer_sampling_net=None
        scheduler_sampling_net=None

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer_video,
                    mode='min',
                    factor=0.5,    
                    patience=500,    
                    verbose=False)
    
    


    #######################################
    #######   STEP3 训练(评估)    #########
    #######################################
    if args.method == "CF":
        logger.info("Use CF method")
        for it in trange(0, args.Iteration+1):   

            logger.info(f"当前学习率: {optimizer_video.param_groups[0]['lr']}")
            save_this_it = False
            wandb.log({"Progress": it}, step=it)
            #########################
            ######   TO DO 评估  ####
            #########################
            if it==10000:
                eval_video_syn = video_syn.clone().permute(0, 2, 1, 3, 4).contiguous().detach()
                for model_eval in model_eval_pool:
                        logger.info('Evaluation model_train = %s, model_eval = %s, iteration = %d'%(args.model, model_eval, it))
                        accs_test = []
                        accs_train = []
                        for it_eval in range(args.num_eval):
                            net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device)  # get a random model
                            image_syn_eval, label_syn_eval = eval_video_syn.detach().clone(), label_syn.detach().clone() # avoid any unaware modification
                            _, acc_train, acc_test, acc_per_cls = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args, mode='none',test_freq=100)

                            accs_test.append(acc_test)
                            accs_train.append(acc_train)
                            print("acc_per_cls:",acc_per_cls)
                        accs_test = np.array(accs_test)
                        accs_train = np.array(accs_train)
                        acc_test_mean = np.mean(accs_test)
                        acc_test_std = np.std(accs_test)
                        if acc_test_mean > best_acc[model_eval]:
                            best_acc[model_eval] = acc_test_mean
                            best_std[model_eval] = acc_test_std
                            save_this_best_ckpt = True
                        print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------' % (
                            len(accs_test), model_eval, acc_test_mean, acc_test_std))
                        wandb.log({'Accuracy/{}'.format(model_eval): acc_test_mean}, step=it)
                        wandb.log({'Max_Accuracy/{}'.format(model_eval): best_acc[model_eval]}, step=it)
                        wandb.log({'Std/{}'.format(model_eval): acc_test_std}, step=it)
                        wandb.log({'Max_Std/{}'.format(model_eval): best_std[model_eval]}, step=it)

            logger.info("这次训练轮数是:"+str(it))
            ###############################################################################################
            ####    Trick 3 插值预训练的model去捕获特征  但与cf实现略有出入 可以做实验  一个想法是去做序列化 ####
            ###############################################################################################

            #model_interval = interpolate_models(model_init, model_final, model_interval, a=0, b=1)
            #model_interval = R2Plus1D_Custom(num_classes=num_classes, pretrained=False).to(args.device)
            #model_interval.eval()

            start_time = time.time()  # 记录开始时间
            match_loss_total,match_grad_mean,calib_loss_total,calib_grad_mean=0,0,0,0
            for c in range(0,num_classes):
                # if(c%10==0):
                #     logger.info("现在是第"+str(it)+"次更新syn_video"+",处理第"+str(c)+"类")
                #每一个类ipc个样本视频
                class_c_video_syn = video_syn[c*args.ipc:(c+1)*args.ipc].reshape((args.ipc, channel, args.frames, im_size[0], im_size[1]))
                class_c_video_real = get_videos(c, args.batch_real).detach()
                class_c_video_real=class_c_video_real.permute(0, 2, 1, 3, 4)

                ###################
                ###  TO DO Loss ###
                ###################
                #logger.info("当前类的蒸馏训练集尺寸："+str(class_c_video_syn.shape))
                #logger.info("当前类的真实训练集尺寸："+str(class_c_video_real.shape))
               
                #with torch.no_grad():
                feat = model_interval(class_c_video_syn,return_features=True)
                with torch.no_grad():
                    feat_tg=model_interval(class_c_video_real,return_features=True)
                
                feat=F.normalize(feat,dim=1)
                feat_tg=F.normalize(feat_tg,dim=1)
                if sampling_net is not None:
                    #logger.info("做了采样t的操作")
                    ##########################
                    ###  TO Understand it  ###
                    ##########################
                    t=sampling_net(args.device)
                    #logger.info("采样得到的t的类型是:"+str(type(t)))
                    #logger.info("t的shape是:"+str(t.shape))
                else:
                    t=None
                
                ##########################
                ###  TO Understand it  ###
                ##########################
                loss = 1000 * cfloss(feat_tg, feat, t,args)
                #loss = 100*torch.sqrt(cf_loss(feat_tg,feat,t=t))
                #loss = 3000*torch.sum((torch.mean(feat_tg, dim=0) - torch.mean(feat, dim=0))**2)

                # logger.info("Shape of synthetic video features:"+ str(feat.shape))
                # logger.info("Shape of real video features:"+ str(feat_tg.shape))
                match_loss_total += loss.item()
                ##############################################################################################################
                ## To do 更新策略到底有没有效呢 而且我更新synthetic是更新多少呢全部or对应类 (那就涉及到每个c都去更新还是一起更新了) ##
                ##############################################################################################################
                optimizer_video.zero_grad()
                if optimizer_sampling_net is not None:
                    #logger.info("开始更新核心集参数以及对抗网络参数捏")
                    optimizer_sampling_net.zero_grad()
                    loss.backward()
                    # 反转sampling_net梯度方向
                    for name, param in sampling_net.named_parameters():
                        #print(f"参数 {name} 的梯度:\n", param.grad is None)
                        param.grad *= -1  
                  
                    optimizer_video.step()
                    optimizer_sampling_net.step()
                else:
                    loss.backward()
                    #logger.info("计算Loss前video_syn.grad 是否为 None:"+ str(video_syn.grad is None)) 
                    optimizer_video.step()

                video_syn_grad=video_syn.grad
                if video_syn_grad is not None:
                    #####################################################################
                    ######  DONE!!!!! TO DO 这个梯度是干什么呢  研究一下梯度是如何传的  #####
                    #####################################################################
                    match_grad_mean += torch.norm(video_syn_grad).item()
                    


            ################################
            ### TO DO trick4 calib_loss  ###
            ################################
            for i in range(0,args.iter_calib):
                for c in range(0,num_classes): #num_classes
                    calib_class_c_video_syn = video_syn[c*args.ipc:(c+1)*args.ipc].reshape((args.ipc, channel, args.frames, im_size[0], im_size[1]))
                    calib_label_c_video_syn = label_syn[c * args.ipc : (c + 1) * args.ipc]

                    ##################################
                    ### DONE!!!!! TO DO bug logits ###
                    ##################################
                    calib_loss = args.calib_weight * interloss(calib_class_c_video_syn, calib_label_c_video_syn, model_final)
                    calib_loss_total += calib_loss.item()

                    #print("calib_loss是:",calib_loss)
                    optimizer_video.zero_grad()
                    calib_loss.backward()
                    optimizer_video.step()

                    video_syn_grad=video_syn.grad
                    if video_syn_grad is not None:
                        #print("3333333")
                        calib_grad_norm = torch.norm(video_syn_grad).item()
                        calib_grad_mean=calib_grad_norm
                        #print("calib_gard_mean:",calib_grad_mean)
                   
                  

            ################################
            ### TO DO distributed_metric ###
            ################################
            #calib_loss_total,match_loss_total,match_grad_mean,calib_grad_mean = sync_distributed_metric([calib_loss_total,match_loss_total,match_grad_mean,calib_grad_mean],mode="all")
            #total_grad_mean = match_grad_mean + calib_grad_mean if args.iter_calib>0 else match_grad_mean
            
            

            print("match_loss_total是",match_loss_total)
            print("calib_loss_total是",calib_loss_total)
            current_loss = (match_loss_total + calib_loss_total) / num_classes if args.iter_calib>0 else (match_loss_total) / num_classes
            print("current_loss是",current_loss)
            
            loss_history.append(current_loss)  
            match_loss_history.append(match_loss_total)
            calib_loss_history.append(calib_loss_total)

            ###调整学习率
            scheduler.step(current_loss)
            # if scheduler_sampling_net is not None:
            #     scheduler_sampling_net.step(current_loss)
                        
            end_time = time.time()  # 记录开始时间
            #print(f"循环执行时间: {end_time - start_time:.4f} 秒")  # 计算并打印时间
    
        
    
        # 画出综合损失曲线
        plt.figure(figsize=(8, 6))
        plt.plot(loss_history, label="Total Loss", color='blue', linewidth=2)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training Total Loss Curve")
        plt.legend()
        plt.grid(True)
        plt.savefig(loss_png_path, dpi=300)
        plt.close()
        print(f"Total loss curve saved to: {loss_png_path}")

        # 画出 match_loss_total 曲线
        plt.figure(figsize=(8, 6))
        plt.plot(match_loss_history, label="Match Loss", color='red', linewidth=2)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Match Loss Curve")
        plt.legend()
        plt.grid(True)
        plt.savefig(match_loss_png_path, dpi=300)
        plt.close()
        print(f"Match loss curve saved to: {match_loss_png_path}")

        # 画出 calib_loss_total 曲线
        plt.figure(figsize=(8, 6))
        plt.plot(calib_loss_history, label="Calib Loss", color='green', linewidth=2)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Calibration Loss Curve")
        plt.legend()
        plt.grid(True)
        plt.savefig(calib_loss_png_path, dpi=300)
        plt.close()
        print(f"Calibration loss curve saved to: {calib_loss_png_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='HMDB51', help='dataset')
    parser.add_argument('--method', type=str, default='CF', help='MTT or DM  or CF')
    parser.add_argument('--model', type=str, default='ConvNet3D', help='model')
    parser.add_argument('--data_path', type=str, default='/opt/data/private/video_distillation-old/distill_utils/data/', help='dataset path')
    parser.add_argument('--Iteration', type=int, default=1000, help='how many distillation steps to perform')
    parser.add_argument('--eval_it', type=int, default=50, help='how often to evaluate')
    parser.add_argument('--eval_mode', type=str, default='SS',
                        help='use top5 to eval top5 accuracy, use S to eval single accuracy')
    parser.add_argument('--batch_syn', type=int, default=None, help='batch size for syn')
    parser.add_argument('--ipc', type=int, default=1, help='instance per class i.e. video per class')
    parser.add_argument('--frames', type=int, default=16, help='')
    parser.add_argument('--lr_teacher', type=float, default=0.001, help='MTT method learning rate for teacher')
    parser.add_argument('--init', type=str, default='real', choices=['noise', 'real', 'real-all'], help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
    parser.add_argument('--lr_video', type=float, default=0.0, help='learning rate for synthetic data')
    parser.add_argument('--sampling_net',type=int,default=0,help='use or not use sampling_net')
    parser.add_argument('--lr_sampling_net',type=float,default=0.01,help='the lr of sampling net')
    parser.add_argument('--alpha_for_loss',type=float,default=0.5,help='')
    parser.add_argument('--beta_for_loss',type=float,default=0.5,help='')
    parser.add_argument('--iter_calib',type=int,default=0,help='') 
    parser.add_argument('--calib_weight',type=float,default=1,help='')
    parser.add_argument('--num_freqs',type=int,default=1024,help='')

    
 


  
    parser.add_argument('--outer_loop', type=int, default=None, help='')
    parser.add_argument('--inner_loop', type=int, default=None, help='')
    parser.add_argument('--num_eval', type=int, default=5, help='how many networks to evaluate on')
    #parser.add_argument('--eval_it', type=int, default=50, help='how often to evaluate')
    parser.add_argument('--epoch_eval_train', type=int, default=1000,
                        help='epochs to train a model with synthetic data')
    #parser.add_argument('--Iteration', type=int, default=1000, help='how many distillation steps to perform')
    parser.add_argument('--lr_net', type=float, default=0.001, help='learning rate for network')
    parser.add_argument('--lr_lr', type=float, default=1e-5, help='MTT trick learning rate for synthetic data')
    parser.add_argument('--train_lr', action='store_true', help='train synthetic lr')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    #parser.add_argument('--batch_syn', type=int, default=None, help='batch size for syn')
    parser.add_argument('--expert_epochs', type=int, default=3, help='how many expert epochs the target params are')
    parser.add_argument('--syn_steps', type=int, default=64, help='how many steps to take on synthetic data')
    parser.add_argument('--max_start_epoch', type=int, default=25, help='max epoch we can start at')
    parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')
    parser.add_argument('--buffer_path', type=str, default=None, help='buffer path')
    parser.add_argument('--num_workers', type=int, default=8, help='')
    parser.add_argument('--preload', action='store_true', help='preload dataset')
    parser.add_argument('--save_path',type=str, default='./logged_files', help='path to save')
    


    args = parser.parse_args()
    torch.autograd.set_detect_anomaly(True)
    main(args)
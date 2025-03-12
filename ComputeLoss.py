import torch 
def compute_match_loss(
    args,
    loader_real, 
    sample_fn, 
    aug_fn, 
    inner_loss_fn, 
    optim_img,  
    class_list, 
    timing_tracker, 
    model_interval, 
    data_grad,
    optim_sampling_net = None,
    sampling_net =None

):
    """
    计算 match_loss。

    Args:
        loader_real: 实际数据的加载器。
        sample_fn: 从生成数据中采样的函数。
        aug_fn: 数据增强函数。
        inner_loss_fn: 用于计算损失的内部函数。
        optim_img: 优化器。
        class_list: 数据集的类别列表。
        timing_tracker: 用于记录时间的跟踪器。
        model_interval: 模型间隔。
        data_grad: 用于存储梯度信息的变量。
    
    Returns:
        Tuple:
        - loss_total: 总损失。
        - grad_mean: 所有类别的梯度范数的平均值。
    """
    match_loss_total = 0
    match_grad_mean = 0

    for c in class_list:
        timing_tracker.start_step()

        # 1. 数据加载
        img, _ = loader_real.class_sample(c)
        timing_tracker.record("data")
        img_syn, _ = sample_fn(c)

        # 2. 数据增强
        img_aug = aug_fn(torch.cat([img, img_syn]))
        timing_tracker.record("aug")
        n = img.shape[0]
        # 3. 计算损失
        loss = inner_loss_fn(img_aug[:n], img_aug[n:], model_interval,sampling_net,args)
        match_loss_total += loss.item()
        timing_tracker.record("loss")

        # 4. 反向传播和优化
        optim_img.zero_grad()
        if optim_sampling_net is not None:
            optim_sampling_net.zero_grad()
            loss.backward(retain_graph=True)
            optim_img.step()
            (-loss).backward()
            optim_sampling_net.step()
        else:
            loss.backward()
            optim_img.step()

            
        if data_grad is not None:
            match_grad_mean += torch.norm(data_grad).item()
        timing_tracker.record("backward")

    return match_loss_total, match_grad_mean
def compute_calib_loss(
    sample_fn, 
    aug_fn, 
    inter_loss_fn, 
    optim_img, 
    iter_calib, 
    class_list, 
    timing_tracker, 
    model_final, 
    calib_weight,
    data_grad
):
    """
    计算校准损失（calib_loss）。

    Args:
        sample_fn: 从生成数据中采样的函数。
        aug_fn: 数据增强函数。
        inter_loss_fn: 用于计算校准损失的函数。
        optim_img: 优化器。
        iter_calib: 校准迭代次数。
        class_list: 数据集的类别列表。
        timing_tracker: 用于记录时间的跟踪器。
        model_final: 最终的模型。
        data_grad: 用于存储梯度信息的变量。

    Returns:
        calib_loss_total: 校准损失总值。
    """
    calib_loss_total = 0
    calib_grad_norm = 0
    for i in range(0,iter_calib):
        for c in class_list:
            timing_tracker.start_step()

            # 1. 数据加载
            img_syn, label_syn = sample_fn(c)
            timing_tracker.record("data")

            # 2. 数据增强
            img_aug = aug_fn(torch.cat([img_syn]))
            timing_tracker.record("aug")

            # 3. 计算损失
            loss = calib_weight * inter_loss_fn(img_aug, label_syn, model_final)
            calib_loss_total += loss.item()
            timing_tracker.record("loss")

            # 4. 反向传播和优化
            optim_img.zero_grad()
            loss.backward()
            if data_grad is not None:
                calib_grad_norm = torch.norm(data_grad).item()
            optim_img.step()
            timing_tracker.record("backward")

    return calib_loss_total,calib_grad_norm
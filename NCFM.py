import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
def cf_loss(X, Y, num_freqs=4096,t=None):
    sigmas  = torch.ones((1, 2048), requires_grad=True).to(X.device)
    wX, wY = 1.0, 1.0
    # 确保输入张量形状正确
    X = X.view(X.size(0), -1)
    Y = Y.view(Y.size(0), -1)
    batch_size, dim = X.size()
    if t is None:
        t = torch.randn((num_freqs, dim), device=X.device)
    tX = torch.matmul(t, X.t())
    cos_tX = (torch.cos(tX) * wX).mean(1)
    sin_tX = (torch.sin(tX) * wX).mean(1)
    # 计算 Y 的特征函数
    tY = torch.matmul(t, Y.t())
    cos_tY = (torch.cos(tY) * wY).mean(1)
    sin_tY = (torch.sin(tY) * wY).mean(1)
    # 计算损失
    loss = (cos_tX - cos_tY) ** 2 + (sin_tX - sin_tY) ** 2
    return loss.mean()/torch.norm(sigmas, p=2)

def calculate_norm(x_r, x_i):
    return torch.sqrt(torch.mul(x_r, x_r) + torch.mul(x_i, x_i))

def calculate_imag(x):
    return torch.mean(torch.sin(x), dim=1)

def calculate_real(x):
    return torch.mean(torch.cos(x), dim=1)

class CFLossFunc(nn.Module):
    """
    CF loss function in terms of phase and amplitude difference.
    Args:
        alpha_for_loss: the weight for amplitude in CF loss, from 0-1
        beta_for_loss: the weight for phase in CF loss, from 0-1
    """
    def __init__(self, alpha_for_loss=0.5, beta_for_loss=0.5):
        super(CFLossFunc, self).__init__()
        self.alpha = alpha_for_loss
        self.beta = beta_for_loss

    def forward(self, feat_tg, feat, t=None, args=None):
        """
        Calculate CF loss between target and synthetic features.
        Args:
            feat_tg: target features from real data [B1 x D]
            feat: synthetic features [B2 x D]
            args: additional arguments containing num_freqs
        """
        # Generate random frequencies
        if t is None:
            t = torch.randn((args.num_freqs, feat.size(1)), device=feat.device)
            #print("t的shape是:"+str(t.shape))
        t_x_real = calculate_real(torch.matmul(t, feat.t()))
        t_x_imag = calculate_imag(torch.matmul(t, feat.t()))
        t_x_norm = calculate_norm(t_x_real, t_x_imag)

        t_target_real = calculate_real(torch.matmul(t, feat_tg.t()))
        t_target_imag = calculate_imag(torch.matmul(t, feat_tg.t()))
        t_target_norm = calculate_norm(t_target_real, t_target_imag)

        # Calculate amplitude difference and phase difference
        amp_diff = t_target_norm - t_x_norm
        loss_amp = torch.mul(amp_diff, amp_diff)

        loss_pha = 2 * (torch.mul(t_target_norm, t_x_norm) -
                       torch.mul(t_x_real, t_target_real) -
                       torch.mul(t_x_imag, t_target_imag))

        loss_pha = loss_pha.clamp(min=1e-12)  # Ensure numerical stability

        # Combine losses
        loss = torch.mean(torch.sqrt(self.alpha * loss_amp + self.beta * loss_pha))
        return loss
    
def innerloss(img_real, img_syn, model,sampling_net=None, args=None):
    """Matching losses (feature or gradient)
    """
    with torch.no_grad():
        _, feat_tg = model(img_real, return_features=True)
    _, feat = model(img_syn, return_features=True)

    if args.dis_metrics == "MMD": 
        loss = torch.sum((feat.mean(0) - feat_tg.mean(0))**2)
    else:
        feat = F.normalize(feat, dim=1)   
        feat_tg = F.normalize(feat_tg, dim=1)
        if sampling_net is not None:
            t = sampling_net(args.device)
        else:
            t=None
        # loss = 30 * args.cf_loss_func(feat_tg, feat, t,args)
        loss = 100*torch.sqrt(cf_loss(feat_tg,feat,t=t))
    return loss


def mutil_layer_innerloss(img_real, img_syn, model, args=None):
    
    # Ensure layer_index is a list
    assert isinstance(args.layer_index, list), "args.layer_index must be a list of layer indices"
    
    # Initialize loss as a tensor on the correct device
    loss = torch.tensor(0.0).to(img_real.device)

    # Extract features for both real and synthetic images
    with torch.no_grad():
        feat_tg_list = model.get_feature_mutil(img_real)  # Real image features
    feat_list = model.get_feature_mutil(img_syn)  # Synthetic image features

    for layer_index in args.layer_index:
        assert 0 <= layer_index <= 6, f"layer_index {layer_index} must be between 0 and 6"
        if args.dis_metrics == "MMD":
            # If the metric is MMD, calculate the MMD loss for the selected layer
            feat = feat_list[layer_index]
            feat_tg = feat_tg_list[layer_index]
            loss += torch.sum((feat.mean(0) - feat_tg.mean(0))**2)
        else:
            # Otherwise, calculate the feature matching loss for the selected layer
            feat = feat_list[layer_index]
            feat_tg = feat_tg_list[layer_index]
            feat = F.normalize(feat, dim=1)  # Normalize the feature
            feat_tg = F.normalize(feat_tg, dim=1)  # Normalize the target feature
            t = None  # Adjust this based on your CFLossFunc usage
            loss += 300 * args.cf_loss_func(feat_tg, feat, t, args)

    return loss
def interloss(img_syn, label_syn, trained_model):
    logits = trained_model(img_syn, return_features=False)
    loss = F.cross_entropy(logits, label_syn)
    return loss
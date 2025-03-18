import torch.nn.functional as F
import torch

def SoftCrossEntropy(inputs, target,temperature=1.0,reduction='average'):
    input_log_likelihood = -F.log_softmax(inputs/temperature, dim=1)
    target_log_likelihood = F.softmax(target/temperature, dim=1)
    batch = inputs.shape[0]
    loss = torch.sum(torch.mul(input_log_likelihood, target_log_likelihood)) / batch
    return loss

# 自定义 Hinge Loss
def HingeLoss(logits, one_hot_labels):
    margins = 1 - one_hot_labels * logits
    loss = torch.clamp(margins, min=0).mean()
    return loss


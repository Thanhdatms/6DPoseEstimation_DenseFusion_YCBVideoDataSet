from torch.nn.modules.loss import _Loss
from torch.autograd import Variable
import torch
import torch.nn as nn
import random
import copy
import math

CEloss = nn.CrossEntropyLoss()

def loss_calculation (semantic, target):

    # sematic has shape [batch_sizem, 22, 480, 640]
    pixel_num = semantic.size()[3] * semantic.size()[2]
    batch_size = semantic.size()[0]

    target = target.view(batch_size, -1).view(-1).contiguous() # flatten to [batch_size * 480 * 640]
    semantic = semantic.view(batch_size, 22, pixel_num).permute(0,2,1).contiguous().view(batch_size * pixel_num, 22).contiguous()

    semantic_loss = CEloss(semantic, target)

    return semantic_loss

class Loss(_Loss):
    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, semantic, target):
        loss = loss_calculation(semantic, target)
        return loss
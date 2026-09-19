from abc import ABC
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module, ABC):
    def __init__(self, alpha=2, beta=4):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, prediction, target):

        positive_index = target.eq(1).float()


        negative_index = target.lt(1).float()


        negative_weights = torch.pow(1 - target, self.beta)


        prediction = torch.clamp(prediction, 1e-12)


        positive_loss = torch.log(prediction) * torch.pow(1 - prediction, self.alpha) * positive_index


        negative_loss = torch.log(1 - prediction) * \
                        torch.pow(prediction, self.alpha) * \
                        negative_weights * \
                        negative_index


        num_positive = positive_index.float().sum()


        positive_loss_sum = positive_loss.sum()
        negative_loss_sum = negative_loss.sum()


        if num_positive == 0:


            loss = -negative_loss_sum
        else:


            loss = -(positive_loss_sum + negative_loss_sum) / num_positive

        return loss


class LBHinge(nn.Module):
    def __init__(self, error_metric=nn.MSELoss(), threshold=None, clip=None):
        super().__init__()
        self.error_metric = error_metric


        self.threshold = threshold if threshold is not None else -100
        self.clip = clip

    def forward(self, prediction, label, target_bb=None):

        negative_mask = (label < self.threshold).float()


        positive_mask = (1.0 - negative_mask)


        prediction_modified = negative_mask * F.relu(prediction) + positive_mask * prediction


        target_modified = positive_mask * label


        loss = self.error_metric(prediction_modified, target_modified)


        if self.clip is not None:
            loss = torch.min(loss, torch.tensor([self.clip], device=loss.device))

        return loss

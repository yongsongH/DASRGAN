import torch
from torch import nn as nn
from torch.nn import functional as F
from torchvision.transforms import functional as TF

from basicsr.archs.vgg_arch import VGGFeatureExtractor
from basicsr.utils.registry import LOSS_REGISTRY
from .loss_util import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']


@LOSS_REGISTRY.register()
class NoisesPerceptualLoss(nn.Module):
    def __init__(self,
                 layer_weights,
                 vgg_type='vgg19',
                 vgg_weights=None,
                 use_input_norm=True,
                 range_norm=False,
                 perceptual_weight=0.,
                 style_weight=0.,
                 noise_weight=1.0,
                 criterion='fro'):
        super(NoisesPerceptualLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.noise_weight = noise_weight
        self.layer_weights = layer_weights
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            range_norm=range_norm)

     

        self.criterion_type = criterion
        if self.criterion_type == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == 'l2':
            self.criterion = torch.nn.L2loss()
        elif self.criterion_type == 'fro':
            self.criterion = None
        else:
            raise NotImplementedError(f'{criterion} criterion has not been supported.')

    def forward(self, x, gt):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        # calculate perceptual loss
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    percep_loss += torch.norm(x_features[k] - gt_features[k], p='fro') * self.layer_weights[k]
                else:
                    percep_loss += self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None

        # calculate style loss
        if self.style_weight > 0:
            style_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    style_loss += torch.norm(
                        self._gram_mat(x_features[k]) - self._gram_mat(gt_features[k]), p='fro') * self.layer_weights[k]
                else:
                    style_loss += self.criterion(self._gram_mat(x_features[k]), self._gram_mat(
                        gt_features[k])) * self.layer_weights[k]
            style_loss *= self.style_weight
        else:
            style_loss = None

        # calculate noise perceptual loss
        if self.noise_weight > 0:
            noise = torch.randn_like(x)
            noise_features = self.vgg(noise)
            noise_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    noise_loss -= torch.norm(x_features[k] - noise_features[k], p='fro') * self.layer_weights[k]
                else:
                    noise_loss -= self.criterion(x_features[k], noise_features[k]) * self.layer_weights[k]
            noise_loss *= self.noise_weight
        else:
            noise_loss = None

        # return percep_loss, style_loss, noise_loss
        total_loss = 0

        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    percep_loss += torch.norm(x_features[k] - gt_features[k], p='fro') * self.layer_weights[k]
                else:
                    percep_loss += self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]
            total_loss += percep_loss * self.perceptual_weight

        if self.style_weight > 0:
            style_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    style_loss += torch.norm(
                        self._gram_mat(x_features[k]) - self._gram_mat(gt_features[k]), p='fro') * self.layer_weights[k]
                else:
                    style_loss += self.criterion(self._gram_mat(x_features[k]), self._gram_mat(
                        gt_features[k])) * self.layer_weights[k]
            total_loss += style_loss * self.style_weight

        if self.noise_weight > 0:
            noise = torch.randn_like(x)
            noise_features = self.vgg(noise)
            noise_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    noise_loss -= torch.norm(x_features[k] - noise_features[k], p='fro') * self.layer_weights[k]
                else:
                    noise_loss -= self.criterion(x_features[k], noise_features[k]) * self.layer_weights[k]
            total_loss += noise_loss * self.noise_weight

        return total_loss

    def _gram_mat(self, x):
        """Calculate Gram matrix.

        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Gram matrix.
        """
        n, c, h, w = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram

 

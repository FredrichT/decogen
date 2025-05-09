"""
Loss functions for CycleGAN style transfer.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class GANLoss(nn.Module):
    """
    Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """
    def __init__(self, target_real_label=1.0, target_fake_label=0.0, reduction='mean'):
        """
        Initialize the GANLoss class.
        
        Parameters:
            target_real_label (float) - - the label for a real image
            target_fake_label (float) - - the label for a fake image
            reduction (str) - - can be 'none', 'mean', 'sum'
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.loss = nn.MSELoss(reduction=reduction)
        
    def get_target_tensor(self, prediction, target_is_real):
        """
        Create label tensors with the same size as the input.
        
        Parameters:
            prediction (tensor) - - typically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
            
        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)
        
    def __call__(self, prediction, target_is_real):
        """
        Calculate loss given discriminator's output and ground truth labels.
        
        Parameters:
            prediction (tensor) - - typically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
            
        Returns:
            the calculated loss.
        """
        # Multi-scale discriminator returns a list of outputs
        if isinstance(prediction, list):
            loss = 0
            for pred_i in prediction:
                target_tensor = self.get_target_tensor(pred_i, target_is_real)
                loss += self.loss(pred_i, target_tensor)
            return loss / len(prediction)
        else:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
            return loss


class CycleLoss(nn.Module):
    """
    Cycle consistency loss.
    Ensures that the image translation cycle brings the image back to the original.
    """
    def __init__(self, lambda_cycle=10.0):
        super(CycleLoss, self).__init__()
        self.lambda_cycle = lambda_cycle
        self.criterion = nn.L1Loss()
        
    def forward(self, real_A, recovered_A, real_B, recovered_B):
        """
        Calculate cycle consistency loss.
        
        Parameters:
            real_A (tensor) - - original images from domain A
            recovered_A (tensor) - - recovered images (after A→B→A)
            real_B (tensor) - - original images from domain B
            recovered_B (tensor) - - recovered images (after B→A→B)
            
        Returns:
            cycle_loss_A: loss for domain A
            cycle_loss_B: loss for domain B
            total_cycle_loss: combined cycle loss
        """
        cycle_loss_A = self.criterion(recovered_A, real_A) * self.lambda_cycle
        cycle_loss_B = self.criterion(recovered_B, real_B) * self.lambda_cycle
        total_cycle_loss = cycle_loss_A + cycle_loss_B
        
        return {
            'cycle_loss_A': cycle_loss_A,
            'cycle_loss_B': cycle_loss_B,
            'total_cycle_loss': total_cycle_loss
        }


class IdentityLoss(nn.Module):
    """
    Identity loss.
    Encourages the generator to preserve color composition when translating images.
    """
    def __init__(self, lambda_identity=0.5):
        super(IdentityLoss, self).__init__()
        self.lambda_identity = lambda_identity
        self.criterion = nn.L1Loss()
        
    def forward(self, real_A, identity_A, real_B, identity_B):
        """
        Calculate identity loss.
        
        Parameters:
            real_A (tensor) - - original images from domain A
            identity_A (tensor) - - identity mapping of domain A (G_B(A))
            real_B (tensor) - - original images from domain B
            identity_B (tensor) - - identity mapping of domain B (G_A(B))
            
        Returns:
            id_loss_A: identity loss for domain A
            id_loss_B: identity loss for domain B
            total_id_loss: combined identity loss
        """
        id_loss_A = self.criterion(identity_A, real_A) * self.lambda_identity
        id_loss_B = self.criterion(identity_B, real_B) * self.lambda_identity
        total_id_loss = id_loss_A + id_loss_B
        
        return {
            'id_loss_A': id_loss_A,
            'id_loss_B': id_loss_B,
            'total_id_loss': total_id_loss
        }


class StructuralLoss(nn.Module):
    """
    Structural preservation loss using perceptual features.
    Helps preserve architectural elements during style transfer.
    """
    def __init__(self, perceptual_net, lambda_structure=10.0):
        super(StructuralLoss, self).__init__()
        self.perceptual_net = perceptual_net
        self.lambda_structure = lambda_structure
        self.criterion = nn.L1Loss()
        
    def forward(self, real_A, fake_B, real_B, fake_A):
        """
        Calculate structural preservation loss using perceptual features.
        
        Parameters:
            real_A (tensor) - - original images from domain A
            fake_B (tensor) - - generated images in domain B (G_A(A))
            real_B (tensor) - - original images from domain B
            fake_A (tensor) - - generated images in domain A (G_B(B))
            
        Returns:
            struct_loss_A: structural loss for domain A->B transfer
            struct_loss_B: structural loss for domain B->A transfer
            total_struct_loss: combined structural loss
        """
        # Extract perceptual features
        real_A_feat1, real_A_feat2, real_A_feat3 = self.perceptual_net(real_A)
        fake_B_feat1, fake_B_feat2, fake_B_feat3 = self.perceptual_net(fake_B)
        
        real_B_feat1, real_B_feat2, real_B_feat3 = self.perceptual_net(real_B)
        fake_A_feat1, fake_A_feat2, fake_A_feat3 = self.perceptual_net(fake_A)
        
        # Use deeper features for structural content (more abstract)
        # but with less weight on high-level features to allow style changes
        struct_loss_A = (
            self.criterion(fake_B_feat1, real_A_feat1) * 0.5 +
            self.criterion(fake_B_feat2, real_A_feat2) * 0.3 +
            self.criterion(fake_B_feat3, real_A_feat3) * 0.2
        ) * self.lambda_structure
        
        struct_loss_B = (
            self.criterion(fake_A_feat1, real_B_feat1) * 0.5 +
            self.criterion(fake_A_feat2, real_B_feat2) * 0.3 +
            self.criterion(fake_A_feat3, real_B_feat3) * 0.2
        ) * self.lambda_structure
        
        total_struct_loss = struct_loss_A + struct_loss_B
        
        return {
            'struct_loss_A': struct_loss_A,
            'struct_loss_B': struct_loss_B,
            'total_struct_loss': total_struct_loss
        }
"""
CycleGAN model implementation with enhanced architecture from Fu (2022).
"""
import os
import random
import torch
import torch.nn as nn
import itertools
from .networks import ResnetGenerator, MultiScaleDiscriminator, PerceptualNet, init_weights
from .losses import GANLoss, CycleLoss, IdentityLoss, StructuralLoss


class CycleGAN(nn.Module):
    """
    CycleGAN model for unpaired image-to-image translation.
    Implements the improved architecture from Fu (2022).
    """
    def __init__(
        self,
        device,
        lambda_cycle=10.0,
        lambda_identity=0.5,
        lambda_structure=10.0
    ):
        """
        Initialize the CycleGAN model.
        
        Parameters:
            device (torch.device) - - device to run the model on
            lambda_cycle (float) - - weight for cycle loss
            lambda_identity (float) - - weight for identity loss
            lambda_structure (float) - - weight for structural preservation loss
        """
        super(CycleGAN, self).__init__()
        self.device = device
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity
        self.lambda_structure = lambda_structure
        
        # Define generators (A->B and B->A)
        self.netG_A = ResnetGenerator().to(self.device)
        self.netG_B = ResnetGenerator().to(self.device)
        
        # Define discriminators (for A and B)
        self.netD_A = MultiScaleDiscriminator().to(self.device)
        self.netD_B = MultiScaleDiscriminator().to(self.device)
        
        # Initialize networks
        init_weights(self.netG_A)
        init_weights(self.netG_B)
        init_weights(self.netD_A)
        init_weights(self.netD_B)
        
        # Perceptual network for structural loss
        self.perceptual_net = PerceptualNet().to(self.device)
        
        # Define loss functions
        self.criterionGAN = GANLoss().to(self.device)
        self.criterionCycle = CycleLoss(lambda_cycle).to(self.device)
        self.criterionIdentity = IdentityLoss(lambda_identity).to(self.device)
        self.criterionStructure = StructuralLoss(self.perceptual_net, lambda_structure).to(self.device)
        
        # Optimizers
        self.optimizer_G = torch.optim.Adam(
            itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
            lr=0.0002,
            betas=(0.5, 0.999)
        )
        self.optimizer_D = torch.optim.Adam(
            itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
            lr=0.0002,
            betas=(0.5, 0.999)
        )
        
        # Learning rate schedulers
        self.scheduler_G = torch.optim.lr_scheduler.StepLR(
            self.optimizer_G, step_size=30, gamma=0.5
        )
        self.scheduler_D = torch.optim.lr_scheduler.StepLR(
            self.optimizer_D, step_size=30, gamma=0.5
        )
        
        # Image buffer to reduce model oscillation
        self.fake_A_buffer = ImageBuffer()
        self.fake_B_buffer = ImageBuffer()
        
    def set_requires_grad(self, nets, requires_grad=False):
        """
        Set requies_grad=False for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
    
    def forward(self, real_A, real_B):
        """
        Run forward pass.
        This is used for visualization, not for training.
        """
        # G_A(A) = fake_B
        fake_B = self.netG_A(real_A)
        # G_B(B) = fake_A
        fake_A = self.netG_B(real_B)
        # G_B(G_A(A)) = rec_A
        rec_A = self.netG_B(fake_B)
        # G_A(G_B(B)) = rec_B
        rec_B = self.netG_A(fake_A)
        # Identity mapping: G_A(B) = id_B, G_B(A) = id_A
        id_A = self.netG_B(real_A)
        id_B = self.netG_A(real_B)
        
        return {
            'real_A': real_A,
            'fake_B': fake_B,
            'rec_A': rec_A,
            'id_A': id_A,
            'real_B': real_B,
            'fake_A': fake_A,
            'rec_B': rec_B,
            'id_B': id_B
        }
    
    def optimize_parameters(self, real_A, real_B):
        """
        Update network parameters.
        
        Parameters:
            real_A (tensor) -- real images from domain A
            real_B (tensor) -- real images from domain B
            
        Returns:
            Dict of losses for monitoring
        """
        # Forward pass
        # G_A(A) = fake_B
        fake_B = self.netG_A(real_A)
        # G_B(B) = fake_A
        fake_A = self.netG_B(real_B)
        
        # G_A should generate realistic images in domain B
        disc_fake_B = self.netD_B(fake_B)
        # G_B should generate realistic images in domain A
        disc_fake_A = self.netD_A(fake_A)
        
        # Identity mapping
        id_A = self.netG_B(real_A)
        id_B = self.netG_A(real_B)
        
        # Cycle consistency
        rec_A = self.netG_B(fake_B)
        rec_B = self.netG_A(fake_A)
        
        # Part 1: Update G_A and G_B networks
        # Don't compute D gradients for G update
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        
        self.optimizer_G.zero_grad()
        
        # GAN loss for G_A and G_B
        gan_loss_A = self.criterionGAN(disc_fake_B, True)  # G_A should generate realistic B
        gan_loss_B = self.criterionGAN(disc_fake_A, True)  # G_B should generate realistic A
        gan_loss = gan_loss_A + gan_loss_B
        
        # Cycle consistency loss
        cycle_losses = self.criterionCycle(real_A, rec_A, real_B, rec_B)
        cycle_loss = cycle_losses['total_cycle_loss']
        
        # Identity loss
        identity_losses = self.criterionIdentity(real_A, id_A, real_B, id_B)
        identity_loss = identity_losses['total_id_loss']
        
        # Structural preservation loss
        struct_losses = self.criterionStructure(real_A, fake_B, real_B, fake_A)
        struct_loss = struct_losses['total_struct_loss']
        
        # Combined loss for generators
        g_loss = gan_loss + cycle_loss + identity_loss + struct_loss
        
        # Calculate gradients for G_A and G_B
        g_loss.backward()
        
        # Update G_A and G_B's weights
        self.optimizer_G.step()
        
        # Part 2: Update D_A and D_B networks
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()
        
        # Get previously generated fake samples from the buffer
        fake_A_buffer = self.fake_A_buffer.push_and_pop(fake_A)
        fake_B_buffer = self.fake_B_buffer.push_and_pop(fake_B)
        
        # D_A should classify real_A as real and fake_A as fake
        disc_real_A = self.netD_A(real_A)
        disc_fake_A = self.netD_A(fake_A_buffer.detach())
        d_loss_A_real = self.criterionGAN(disc_real_A, True)
        d_loss_A_fake = self.criterionGAN(disc_fake_A, False)
        d_loss_A = (d_loss_A_real + d_loss_A_fake) * 0.5
        
        # D_B should classify real_B as real and fake_B as fake
        disc_real_B = self.netD_B(real_B)
        disc_fake_B = self.netD_B(fake_B_buffer.detach())
        d_loss_B_real = self.criterionGAN(disc_real_B, True)
        d_loss_B_fake = self.criterionGAN(disc_fake_B, False)
        d_loss_B = (d_loss_B_real + d_loss_B_fake) * 0.5
        
        # Combined loss for discriminators
        d_loss = d_loss_A + d_loss_B
        
        # Calculate gradients for D_A and D_B
        d_loss.backward()
        
        # Update D_A and D_B's weights
        self.optimizer_D.step()
        
        # Return losses for monitoring
        return {
            # Generator losses
            'g_loss': g_loss.item(),
            'gan_loss_A': gan_loss_A.item(),
            'gan_loss_B': gan_loss_B.item(),
            'cycle_loss_A': cycle_losses['cycle_loss_A'].item(),
            'cycle_loss_B': cycle_losses['cycle_loss_B'].item(),
            'identity_loss_A': identity_losses['id_loss_A'].item(),
            'identity_loss_B': identity_losses['id_loss_B'].item(),
            'struct_loss_A': struct_losses['struct_loss_A'].item(),
            'struct_loss_B': struct_losses['struct_loss_B'].item(),
            
            # Discriminator losses
            'd_loss': d_loss.item(),
            'd_loss_A': d_loss_A.item(),
            'd_loss_B': d_loss_B.item(),
        }
    
    def update_learning_rate(self):
        """Update learning rates for all the networks"""
        self.scheduler_G.step()
        self.scheduler_D.step()
    
    def save_networks(self, epoch, save_dir):
        """
        Save models to the disk.
        
        Parameters:
            epoch (int) -- current epoch
            save_dir (str) -- directory to save the models
        """
        for name in ['G_A', 'G_B', 'D_A', 'D_B']:
            if name.startswith('G'):
                net = getattr(self, 'net' + name)
                save_filename = f'{name}_net_{epoch}.pth'
                save_path = os.path.join(save_dir, save_filename)
                torch.save(net.state_dict(), save_path)
    
    def load_networks(self, epoch, load_dir):
        """
        Load models from the disk.
        
        Parameters:
            epoch (int) -- saved model epoch
            load_dir (str) -- directory to load the models from
        """
        for name in ['G_A', 'G_B', 'D_A', 'D_B']:
            if name.startswith('G'):
                net = getattr(self, 'net' + name)
                load_filename = f'{name}_net_{epoch}.pth'
                load_path = os.path.join(load_dir, load_filename)
                
                if os.path.exists(load_path):
                    state_dict = torch.load(load_path, map_location=self.device)
                    net.load_state_dict(state_dict)
                    print(f'Successfully loaded {name} from {load_path}')
                else:
                    print(f'Failed to load {name} from {load_path}')


class ImageBuffer:
    """
    This class implements an image buffer that stores previously generated images.
    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    """
    def __init__(self, buffer_size=50):
        """
        Initialize the ImageBuffer class.
        
        Parameters:
            buffer_size (int) -- the size of the image buffer, if buffer_size=0, no buffer will be created
        """
        self.buffer_size = buffer_size
        self.num_imgs = 0
        self.buffer = []
    
    def push_and_pop(self, images):
        """
        Push images into the buffer and pop images out of the buffer.
        
        Parameters:
            images (tensor) -- the latest generated images from the generator
            
        Returns:
            images from the buffer
        """
        if self.buffer_size == 0:  # If buffer_size=0, return the input directly
            return images
        
        return_images = []
        for image in images:
            image = torch.unsqueeze(image, 0)  # Add batch dimension
            
            # If the buffer is not full, add current image to the buffer
            if self.num_imgs < self.buffer_size:
                self.buffer.append(image)
                self.num_imgs += 1
                return_images.append(image)
            else:
                # If buffer is full, 50% chance to return a buffer image
                if random.random() > 0.5:
                    idx = random.randint(0, self.buffer_size - 1)
                    temp = self.buffer[idx].clone()
                    self.buffer[idx] = image
                    return_images.append(temp)
                else:
                    return_images.append(image)
        
        return torch.cat(return_images, 0)  # Combine into a batch
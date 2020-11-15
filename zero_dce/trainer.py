import os
import torch
import wandb
import numpy as np
from PIL import Image
from .losses import *
from tqdm import tqdm
from torch.utils import data
from .model import DCENet, weights_init
from .dataloader import LowLightDataset


class Trainer:
    """Trainer for Lowlight Image Enhancement using Zero DCE"""

    def __init__(self):
        self.dataloader = None
        self.model = None
        self.color_loss = None
        self.exposure_loss = None
        self.illumination_smoothing_loss = None
        self.spatial_consistency_loss = None
        self.optimizer = None

    def build_dataloader(self, image_path, image_size=256, batch_size=8, num_workers=4):
        """Build Dataloader for training

        Args:
            image_path: list of image files
            image_size: size of image for resizing
            batch_size: batch size for training
            num_workers: number of workers for dataloader
        """
        dataset = LowLightDataset(image_files=image_path, image_size=image_size)
        self.dataloader = data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True
        )

    def build_model(self, pretrain_weights=None):
        """Build DCENet Model

        Args:
            pretrain_weights: Path to pre-trained weights
        """
        self.model = DCENet().cuda()
        self.model.apply(weights_init)
        if pretrain_weights is not None:
            self.load_weights(pretrain_weights)

    def compile(self, pretrain_weights=None, learning_rate=1e-4, weight_decay=1e-4):
        """Compile Trainer

        Builds Model, Losses and Optimizer

        Args:
            pretrain_weights: Path to pre-trained weights
            learning_rate: Learning Rate
            weight_decay: Weight Decay
        """
        self.build_model(pretrain_weights=pretrain_weights)
        self.color_loss = ColorConstancyLoss().cuda()
        self.spatial_consistency_loss = SpatialConsistancyLoss().cuda()
        self.exposure_loss = ExposureLoss(patch_size=16, mean_val=0.6).cuda()
        self.illumination_smoothing_loss = IlluminationSmoothnessLoss().cuda()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate,
            weight_decay=weight_decay
        )

    def _train_step(self, image_lowlight):
        image_lowlight = image_lowlight.cuda()
        enhanced_image_1, enhanced_image, A = self.model(image_lowlight)
        loss_tv = 200 * self.illumination_smoothing_loss(A)
        loss_spa = torch.mean(self.spatial_consistency_loss(enhanced_image, image_lowlight))
        loss_col = 5 * torch.mean(self.color_loss(enhanced_image))
        loss_exp = 10 * torch.mean(self.exposure_loss(enhanced_image))
        loss = loss_tv + loss_spa + loss_col + loss_exp
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.model.parameters(), 0.1)
        self.optimizer.step()
        return loss.item()

    def save_model(self, path):
        """Save Trained Model

        Args:
            path: path to save the model
        """
        torch.save(self.model.state_dict(), path)

    def _log_step(self, loss, epoch, iteration):
        wandb.log({'Loss': loss})
        if epoch % 20 == 0:
            self.save_model(
                os.path.join(
                    './checkpoints/',
                    'model_{}_{}.pth'.format(epoch, iteration)
                )
            )

    def train(self, epochs=200, log_frequency=100, notebook=True):
        """Train Model

        Args:
            epochs: number of epochs
            log_frequency: frequency of logging
            notebook: environment is notebook or not
        """
        wandb.watch(self.model)
        self.model.train()
        if notebook:
            from tqdm.notebook import tqdm as tqdm_notebook
            tqdm = tqdm_notebook
        for epoch in range(1, epochs + 1):
            print('Epoch {}/{}'.format(epoch, epochs))
            for iteration, image_lowlight in tqdm(enumerate(self.dataloader)):
                loss = self._train_step(image_lowlight)
                if iteration % log_frequency == 0:
                    self._log_step(loss, epoch, iteration)

    def load_weights(self, weights_path):
        """Load Weights from Checkpoint

        Args:
            weights_path: path to model checkpoint
        """
        self.model.load_state_dict(torch.load(weights_path))

    def infer_cpu(self, image_path, image_resize_factor=None):
        """Infer on CPU

        Args:
            image_path: path to image
            image_resize_factor: factor for resizing image for inference
        """
        with torch.no_grad():
            image_lowlight = Image.open(image_path)
            width, height = image_lowlight.size
            if image_resize_factor is not None:
                image = image_lowlight.resize(
                    (
                        width // image_resize_factor,
                        height // image_resize_factor
                    ),
                    Image.ANTIALIAS)
            lowlight = (np.asarray(image) / 255.0)
            lowlight = torch.from_numpy(lowlight).float()
            lowlight = lowlight.permute(2, 0, 1)
            lowlight = lowlight.unsqueeze(0)
            model = self.model.cpu()
            _, enhanced, _ = model(lowlight)
            enhanced = enhanced.squeeze().permute(1, 2, 0)
        return image_lowlight, enhanced.numpy()

    def infer_gpu(self, image_path, image_resize_factor=None):
        """Infer on GPU

        Args:
            image_path: path to image
            image_resize_factor: factor for resizing image for inference
        """
        with torch.no_grad():
            image_lowlight = Image.open(image_path)
            width, height = image_lowlight.size
            if image_resize_factor is not None:
                image = image_lowlight.resize(
                    (
                        width // image_resize_factor,
                        height // image_resize_factor
                    ),
                    Image.ANTIALIAS)
            lowlight = (np.asarray(image) / 255.0)
            lowlight = torch.from_numpy(lowlight).float()
            lowlight = lowlight.permute(2, 0, 1)
            lowlight = lowlight.cuda().unsqueeze(0)
            _, enhanced, _ = self.model(lowlight)
            enhanced = enhanced.squeeze().permute(1, 2, 0)
        return image_lowlight, enhanced.cpu().numpy()

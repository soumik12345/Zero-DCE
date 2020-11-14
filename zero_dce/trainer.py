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

    def __init__(self):
        self.dataloader = None
        self.model = None
        self.color_loss = None
        self.exposure_loss = None
        self.illumination_smoothing_loss = None
        self.spatial_consistency_loss = None
        self.optimizer = None

    def build_dataloader(self, image_path, image_size=256, batch_size=8, num_workers=4):
        dataset = LowLightDataset(image_files=image_path, image_size=image_size)
        self.dataloader = data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True
        )

    def build_model(self, pretrain_weights=None):
        self.model = DCENet().cuda()
        self.model.apply(weights_init)
        if pretrain_weights is not None:
            self.load_weights(pretrain_weights)

    def compile(self, pretrain_weights=None, learning_rate=1e-4, weight_decay=1e-4):
        self.build_model(pretrain_weights=pretrain_weights)
        self.color_loss = ColorConstancyLoss()
        self.spatial_consistency_loss = SpatialConsistancyLoss()
        self.exposure_loss = ExposureLoss(patch_size=16, mean_val=0.6)
        self.illumination_smoothing_loss = IlluminationSmoothnessLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate,
            weight_decay=weight_decay
        ).cuda()

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
        self.model.load_state_dict(torch.load(weights_path))

    def infer_cpu(self, image_path):
        with torch.no_grad():
            image_lowlight = Image.open(image_path)
            lowlight = (np.asarray(image_lowlight) / 255.0)
            lowlight = torch.from_numpy(lowlight).float()
            lowlight = lowlight.permute(2, 0, 1)
            lowlight = lowlight.unsqueeze(0)
            model = self.model.cpu()
            _, enhanced, _ = model(lowlight)
            enhanced = enhanced.squeeze().permute(1, 2, 0)
        return image_lowlight, enhanced.numpy()

    def infer_gpu(self, image_path):
        with torch.no_grad():
            image_lowlight = Image.open(image_path)
            lowlight = (np.asarray(image_lowlight) / 255.0)
            lowlight = torch.from_numpy(lowlight).float()
            lowlight = lowlight.permute(2, 0, 1)
            lowlight = lowlight.cuda().unsqueeze(0)
            model = self.model
            _, enhanced, _ = model(lowlight)
            enhanced = enhanced.squeeze().permute(1, 2, 0)
        return image_lowlight, enhanced.cpu().numpy()

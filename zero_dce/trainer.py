import os
import torch
import wandb
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

    def build_model(self, pretrain_weights=None, learning_rate=1e-4, weight_decay=1e-4):
        self.model = DCENet().cuda()
        self.model.apply(weights_init)
        if pretrain_weights is not None:
            self.model.load_state_dict(torch.load(pretrain_weights))
        self.color_loss = ColorConstancyLoss()
        self.spatial_consistency_loss = SpatialConsistancyLoss()
        self.exposure_loss = ExposureLoss(patch_size=16, mean_val=0.6)
        self.illumination_smoothing_loss = IlluminationSmoothnessLoss()
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

    def _log_step(self, loss, epoch, iteration):
        wandb.log({'Loss': loss})
        torch.save(
            self.model.state_dict(),
            os.path.join(
                wandb.run.dir,
                'model_{}_{}.pth'.format(epoch, iteration)
            )
        )

    def train(self, epochs=200, log_frequency=100):
        wandb.watch(self.model)
        self.model.train()
        for epoch in range(1, epochs + 1):
            print('Epoch {}/{}'.format(epoch, epochs))
            for iteration, image_lowlight in tqdm(enumerate(self.dataloader)):
                loss = self._train_step(image_lowlight)
                if iteration % log_frequency == 0:
                    self._log_step(loss, epoch)

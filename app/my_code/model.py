import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
# note that I am using pytorch_lightning version 0.7.6 (current version: 1.3.8)
# -> e.g. logging to tensorboard is different in newest version

import torch
import torch.nn.functional as F
import torch.nn as nn

from facenet_pytorch import InceptionResnetV1


class EmotionDetectionModel(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.num_classes = 3

        self.resnet = InceptionResnetV1(pretrained='vggface2', classify=True)

        # use all but the last 3 layers of InceptionResnetV1
        self.feature_extractor = nn.Sequential(*list(self.resnet.children())[:-3])

        # freeze the layers for transfer learning
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # 1x1 conv to reduce the dimensionality of feature_extractor output from (1,1,1792) -> (1,1,300)
        self.channel_reduction = nn.Conv2d(in_channels=1792, out_channels=300, kernel_size=1)

        # linear layers for classification
        self.classifier = nn.Sequential(
            nn.Linear(300, 150),
            nn.Linear(150, 30),
            nn.Linear(30, 3)
        )

    def forward(self, x):
        # if x is not a batch but a single image
        if len(x.shape) != 4:
            x = torch.unsqueeze(x, 0)
        x = x.to(self.device)

        x = self.feature_extractor(x)
        x = self.channel_reduction(x)
        # Flatten for the linear layers
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def general_step(self, batch, batch_idx, mode):
        images, targets = batch

        # forward pass
        out = self.forward(images)

        # loss
        loss = F.cross_entropy(out, targets)

        preds = out.argmax(axis=1)
        n_correct = (targets == preds).sum()

        if mode == "val" and batch_idx == 0:
            self.visualize_predictions(images.detach(), preds, targets)

        return loss, n_correct

    def training_step(self, batch, batch_idx):
        number_samples_in_batch = len(batch[1])
        loss, n_correct = self.general_step(batch, batch_idx, "train")
        tensorboard_logs = {'train_loss': loss}
        self.logger.experiment.add_scalars("losses_overlapped", {"train_loss": loss}, self.global_step)
        return {'loss': loss, 'train_n_correct': n_correct, 'num_samples': number_samples_in_batch, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        number_samples_in_batch = len(batch[1])
        loss, n_correct = self.general_step(batch, batch_idx, "val")
        self.logger.experiment.add_scalars("losses_overlapped", {"val_loss": loss}, self.global_step)
        return {'val_loss': loss, 'val_n_correct': n_correct, 'num_samples': number_samples_in_batch}
        
    def general_end(self, outputs, mode):
        # average over all batches aggregated during one epoch
        if mode == "val":
            avg_loss = torch.stack([x[mode + '_loss'] for x in outputs]).mean()
        else:
            avg_loss = None  # don't compute average train loss over epoch (better to see it for every step)
        total_correct = torch.stack([x[mode + '_n_correct'] for x in outputs]).sum().cpu().numpy()
        total_num = sum([x['num_samples'] for x in outputs])
        acc = total_correct / total_num
        return avg_loss, acc

    def training_epoch_end(self, outputs):
        # outputs is a list of outputs returned by validation_step for every mini-batch
        avg_loss, acc = self.general_end(outputs, "train")
        tensorboard_logs = {'train_acc': acc}
        return {'train_acc': acc, 'log': tensorboard_logs}

    def validation_epoch_end(self, outputs):
        # outputs is a list of outputs returned by validation_step for every mini-batch
        avg_loss, acc = self.general_end(outputs, "val")
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': acc}
        return {'val_epoch_loss': avg_loss, 'val_acc': acc, 'log': tensorboard_logs}

    def configure_optimizers(self):
        params = list(self.channel_reduction.parameters()) + list(self.classifier.parameters())
        optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.01)
        reduce_lr_on_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': reduce_lr_on_plateau,
                'monitor': 'val_loss',
            }
        }

    def visualize_predictions(self, images, preds, targets):
        """Helper function visualize the predictions of the model
        Only visualize 8 samples from batch"""

        class_names = ['angry', 'happy', 'sad']
        images = images[:8]
        preds = preds[:8]
        targets = targets[:8]

        # determine size of the grid based for the given batch size
        num_rows = int(torch.tensor(len(images)).float().sqrt().floor())

        fig = plt.figure(figsize=(7, 7))
        for i in range(len(images)):
            plt.subplot(num_rows, len(images) // num_rows + 1, i+1)
            img = images[i].permute(1, 2, 0).cpu().numpy()
            img = np.array([0.229, 0.224, 0.225]) * img + np.array([0.485, 0.456, 0.406])
            img = np.clip(img, 0, 1)
            plt.imshow(img)
            plt.title(f'pred: {class_names[preds[i]]}'
                      f'\ntruth: [{class_names[targets[i]]}]')
            plt.axis('off')

        self.logger.experiment.add_figure(
            'predictions', fig, global_step=self.global_step)

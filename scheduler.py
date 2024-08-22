import numpy as np
from torch import optim

class ExponentialDecayScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iter):
        self.warmup = warmup
        self.max_iter = max_iter
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch + 1)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        if epoch < self.warmup: # Linear warmup
            lr_factor = epoch / self.warmup
        else:   # Exponential decay
            decay = (epoch - self.warmup) / (self.max_iter - self.warmup)
            lr_factor = np.exp(-decay)
        return lr_factor

class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iter):
        self.warmup = warmup
        self.max_iter = max_iter
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_iter))
        if epoch <= self.warmup:
            lr_factor *= epoch / self.warmup
        return lr_factor

class MultiStepLRScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, milestones,
                 warmup = 0, gamma = 0.1, last_epoch= -1):
        self.milestones = sorted(milestones)
        self.gamma = gamma
        self.warmup = warmup
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch + 1)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        if epoch < self.warmup:  # Linear warmup
            lr_factor = epoch / self.warmup
        else:  # Multi-step decay
            lr_factor = 1.0
            for milestone in self.milestones:
                if epoch >= milestone:
                    lr_factor *= self.gamma
                else:
                    break
        return lr_factor
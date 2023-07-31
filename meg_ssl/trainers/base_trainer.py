import os
from meg_decoding.utils.loggers import Pickleogger
from torch.utils.data import DataLoader, Dataset
import torch
from torch.nn.parallel import DistributedDataParallel
import tqdm
import numpy as np
from logging import getLogger


class BaseSSLTrainer():
    global_rank=0
    def __init__(self,config, device_count:int, usewandb:bool):
        """
        config: OmegaConf {precision, max_epochs, val_check_interval, logdir, num_workers, batch_size, ckpt_dir}
        """
        self.logger = getLogger(__name__)
        self.config = config
        self.device_count = device_count
        self.device = 'cuda' if device_count > 0 else 'cpu'
        self.precision = config.precision
        self.max_epochs = config.max_epochs
        self.val_check_interval = config.val_check_interval
        self.logdir = config.logdir
        self.usewandb = usewandb
        if self.global_rank == 0:
            self.usewandb = False
        if self.usewandb:
            import wandb
            wandb.config.update(config)
        self.num_workers = config.num_workers
        self.logger.info("Preparing to fit...")

        self.setup_logger()

    def setup_logger(self):
        if self.global_rank == 0:
            self.pkl_logger = Pickleogger(self.logdir)

    def fit(self, model, train_dataset:Dataset, val_dataset:Dataset, ckpt_path:str=None):
        self.before_train(model, train_dataset, val_dataset, ckpt_path)
        self.train()
        self.after_train()
        pass

    def before_train(self, model, train_dataset, val_dataset, ckpt_path):
        self.model = model
        if ckpt_path is not None:
            self.model.load_state_dict(torch.load(ckpt_path))
            print('load from ', ckpt_path)
        self.model = self.model.to(self.device)
        if self.device_count > 1:
            self.original_model = model
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            self.model = DistributedDataParallel(model, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=False)
        self.train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=self.num_workers)
        self.val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=self.num_workers)
        self.best_loss = np.inf
        self.other_setup()


    def other_setup(self):
        # optimizer
        pass

    def train(self):
        for epoch in range(self.max_epochs):
            print('[{}/{}Epoch] start'.format(epoch, self.max_epochs))
            self.before_epoch()
            train_metric = self.train_one_epoch()
            train_metric.update({'epoch':epoch})
            self.after_epoch(train_metric)

    def after_train(self):
        if self.global_rank == 0:
            # save model
            self.save_model('last.pth')

    def before_epoch(self):
        pass

    def after_epoch(self, epoch:int, train_metrics:dict):
        if self.global_rank == 0:
            # lr
            for param_group in self.optimizer.param_groups:
                lr = param_group['lr']
                break
            train_metrics.update({'lr':lr})
            # logging
            self.pkl_logger.log(train_metrics, 'train')
            # validation
            if epoch % self.val_check_interval == 0:
                val_metrics = self.val_epoch()
                val_metrics.update({'epoch':epoch})
                # logging
                self.pkl_logger.log(val_metrics, 'val')
                # save model
                if val_metrics['val_loss'] < self.best_loss:
                    self.best_loss = val_metrics['val_loss']
                    print('!! best loss is updated to ', self.best_loss)
                    self.save_model('best.pth')

                train_metrics.update(val_metrics)

            if self.usewandb:
                wandb.log(train_metrics)


    def train_one_epoch(self, epoch:int)->dict:
        self.model.train()
        epoch_loss = []
        with tqdm.tqdm(self.train_loader) as pbar:
            for step, batch in enumerate(pbar):
                loss = self.train_one_step(batch, epoch)
                pbar.set_description("[Epoch {}] step={},loss={}".format(
                    epoch, step, np.mean(loss)))
                epoch_loss.append(loss)
        return {'train_loss': np.mean(epoch_loss)}

    def train_one_step(self, step, batch)->float:
        pass

    def validation(self, epoch:int)->dict:
        self.model.eval()
        val_loss = []
        with torch.no_grad():
            for step, batch in enumerate(self.val_loader):
                loss = self.val_one_step(step, batch)
                val_loss.append(loss)
        return {'val_loss': np.mean(val_loss)}

    def val_one_step(self, step, batch)->float:
        pass

    def save_model(self, filename:str):
        filepath =  os.path.join(self.config.ckpt_dir, filename)
        torch.save(self.model.state_dict(),filepath)
        print('model is saved as ', filepath)


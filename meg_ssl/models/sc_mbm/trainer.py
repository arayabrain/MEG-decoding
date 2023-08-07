import math, sys
import torch
from . import utils as ut
from torch import inf
import numpy as np
import time
from meg_ssl.trainers.base_trainer import BaseSSLTrainer
import timm.optim.optim_factory as optim_factory
import tqdm
import matplotlib.pyplot as plt
import os
import datetime
import random



class Trainer(BaseSSLTrainer):
    def other_setup(self):
        param_groups = optim_factory.param_groups_weight_decay(self.model, self.config.weight_decay) # optim_factory.add_weight_decay(self.model, self.config.weight_decay)
        self.optimizer = torch.optim.AdamW(param_groups, lr=self.config.lr, betas=(0.9, 0.95))
        print(self.optimizer)
        self.loss_scaler = NativeScalerWithGradNormCount()
        os.makedirs(self.config.reconst_fig_dir, exist_ok=True)


    def after_epoch(self, epoch:int, train_metrics:dict):
        super().after_epoch(epoch, train_metrics)
        if epoch % self.val_check_interval==0:
            self.plot_recon_figures2(epoch, split='train')
            self.plot_recon_figures2(epoch, split='val')
            

    def train_one_epoch(self, epoch:int)->dict:
        self.model.train()
        epoch_loss = []
        epoch_corr = []
        with tqdm.tqdm(self.train_loader) as pbar:
            for step, batch in enumerate(pbar):
                loss, corr = self.train_one_step(epoch, step, batch)
                pbar.set_description("[Epoch {}] step={},loss={}, corr={}".format(
                    epoch, step, np.mean(loss), np.mean(corr)))
                epoch_loss.append(loss)
                epoch_corr.append(corr)
        return {'train_loss': np.mean(epoch_loss), 'train_corr':np.mean(epoch_corr)}

    def train_one_step(self, epoch, step, batch)->float:
        batch_eeg = batch

        if step % self.config.accum_iter == 0:
            ut.adjust_learning_rate(self.optimizer, step / len(self.train_loader) + epoch, self.config)
        samples = batch_eeg # data_dcit['eeg']
        img_features = None
        valid_idx = None
        samples = samples.to(self.device)
        # img_features = img_features.to(device)

        self.optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=True):
            loss, pred, _ = self.model(samples, img_features, valid_idx=valid_idx, mask_ratio=self.config.mask_ratio)
        # loss.backward()
        # norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
        # optimizer.step()

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training at step {step} epoch {epoch}")
            sys.exit(1)

        # loss /= accum_iter
        self.loss_scaler(loss, self.optimizer, parameters=self.model.parameters(), clip_grad=self.config.clip_grad)

        pred = pred.to('cpu').detach()
        samples = samples.to('cpu').detach()
        if self.device_count > 1:
            pred = self.original_model.unpatchify(pred)
        pred = self.model.unpatchify(pred)
        cor = torch.mean(torch.tensor([torch.corrcoef(torch.cat([p[0].unsqueeze(0), s[0].unsqueeze(0)],axis=0))[0,1] for p, s in zip(pred, samples)])).item()
        self.optimizer.zero_grad()
        return loss_value, cor

    def validation(self, epoch:int)->dict:
        self.model.eval()
        val_loss = []
        val_corr = []
        with torch.no_grad():
            for step, batch in enumerate(self.val_loader):
                loss, corr = self.val_one_step(step, batch)
                val_loss.append(loss)
                val_corr.append(corr)
        print('val loss: {}, val_corr: {}'.format(np.mean(val_loss), np.mean(val_corr)))
        return {'val_loss': np.mean(val_loss), 'val_corr':np.mean(val_corr)}

    def val_one_step(self, step, batch)->float:
        batch_eeg = batch
        samples = batch_eeg # data_dcit['eeg']

        img_features = None
        valid_idx = None
        samples = samples.to(self.device)
        # img_features = img_features.to(device)
        with torch.cuda.amp.autocast(enabled=True):
            loss, pred, _ = self.model(samples, img_features, valid_idx=valid_idx, mask_ratio=self.config.mask_ratio)

        loss_value = loss.item()

        pred = pred.to('cpu').detach()
        samples = samples.to('cpu').detach()
        if self.device_count > 1:
            pred = self.original_model.unpatchify(pred)
        pred = self.model.unpatchify(pred)
        cor = torch.mean(torch.tensor([torch.corrcoef(torch.cat([p[0].unsqueeze(0), s[0].unsqueeze(0)],axis=0))[0,1] for p, s in zip(pred, samples)])).item()
        
        return loss_value, cor

    @torch.no_grad()
    def plot_recon_figures2(self, epoch:int=None, split:str='val'):
        patch_size = self.model.patch_size
        n_axis=5
        self.model.eval()
        fig, axs = plt.subplots(n_axis, 2, figsize=(20,15))
        fig.tight_layout()
        axs[0,0].set_title('Ground-truth')
        # axs[0,1].set_title('Masked Ground-truth')
        dataloader = self.val_loader if split=='val' else self.train_loader
        axs[0,1].set_title('Reconstruction')
        electrode_ids = np.arange(0,20, int(20/n_axis))
        reconstruct_sample_ids = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45] if dataloader.batch_size > 45 else np.arange(dataloader.batch_size)
        for i, ax in enumerate(axs):
            ei = electrode_ids[i]
            sample = next(iter(dataloader))
            sample_id = reconstruct_sample_ids[i]
            sample = sample.to(self.device)
            with torch.cuda.amp.autocast(enabled=True):
                _, pred, mask = self.model(sample, None, valid_idx=None, mask_ratio=self.config.mask_ratio)
            # sample_with_mask = model_without_ddp.patchify(sample.transpose(1,2))[0].to('cpu').numpy().reshape(-1, model_without_ddp.patch_size)
            sample_with_mask = sample.to('cpu').squeeze(0)[sample_id].numpy().reshape(-1, self.model.patch_size)
            # pred = model_without_ddp.unpatchify(pred.transpose(1,2)).to('cpu').squeeze(0)[0].unsqueeze(0).numpy()
            # sample = sample.to('cpu').squeeze(0)[0].unsqueeze(0).numpy()
            
            if self.device_count > 1:
                pred = self.original_model.unpatchify(pred).to('cpu').squeeze(0)[sample_id].numpy()
            else:
                pred = self.model.unpatchify(pred).to('cpu').squeeze(0)[sample_id].numpy()
            sample = sample.to('cpu')
            # pred = model_without_ddp.unpatchify(model_without_ddp.patchify(sample.transpose(1,2))).to('cpu').squeeze(0)[0].numpy()
            sample = sample.to('cpu').squeeze(0)[sample_id].numpy()
            mask = mask.to('cpu').squeeze(0)[sample_id].numpy()
            mask_start = np.where(mask)[0] * patch_size
            mask_end = np.where(mask)[0] * patch_size + patch_size
            unpatched_mask = np.zeros([1,len(mask)*patch_size]).squeeze()
            # import pdb; pdb.set_trace()
            for s,e in zip(mask_start, mask_end):
                unpatched_mask[s:e] = 1
            
            cor = np.corrcoef([pred[ei], sample[ei]])[0,1]

            x_axis = np.arange(0, sample.shape[-1])
            # groundtruth
            # import pdb; pdb.set_trace()
            # ax[0].plot(x_axis[np.where(unpatched_mask==0)], sample[ei, np.where(unpatched_mask==0)[0]],'m-')
            ax[0].plot(x_axis, sample[ei], 'm-')
            for s, e in zip(mask_start, mask_end):
                ax[0].plot(x_axis[s:e], sample[ei, s:e], 'k-')
            # ax[0].plot(x_axis, sample_with_mask.reshape(*sample.shape)[0], color='r')

            ax[1].plot(x_axis, pred[ei])
            for s, e in zip(mask_start, mask_end):
                ax[1].plot(x_axis[s:e], pred[ei, s:e], 'k-')
            ax[1].set_ylabel('cor: %.4f'%cor, weight = 'bold')
            ax[1].yaxis.set_label_position("right")
        fig_name = 'reconst-%s'%(datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
        if epoch is not None:
            fig_name = '{}epoch-{}-{}'.format(epoch, split, fig_name)
        reconstruct_image_path =os.path.join(self.config.reconst_fig_dir, f'{fig_name}.png')
        fig.savefig(reconstruct_image_path)
        print(f'{epoch} Epoch: save reconstruct image to ', reconstruct_image_path)
        plt.close(fig)


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


def train_one_epoch(model, data_loader, optimizer, device, epoch,
                        loss_scaler, log_writer=None, config=None, start_time=None, model_without_ddp=None,
                        img_feature_extractor=None, preprocess=None):
    model.train(True)
    optimizer.zero_grad()
    total_loss = []
    total_cor = []
    accum_iter = config.accum_iter
    for data_iter_step, batch_data in enumerate(data_loader):
        batch_eeg, batch_image = batch_data
        # we use a per iteration (instead of per epoch) lr scheduler
        # print(data_iter_step)
        # print(len(data_loader))

        if data_iter_step % accum_iter == 0:
            ut.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, config)
        samples = batch_eeg # data_dcit['eeg']

        img_features = None
        valid_idx = None
        if img_feature_extractor is not None:
            images = batch_image # data_dcit['image']
            valid_idx = torch.nonzero(images.sum(dim=(1,2,3)) != 0).squeeze(1)
            img_feature_extractor.eval()
            with torch.no_grad():
                img_features = img_feature_extractor(preprocess(images[valid_idx]).to(device))['layer2']
        samples = samples.to(device)
        # img_features = img_features.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=True):
            loss, pred, _ = model(samples, img_features, valid_idx=valid_idx, mask_ratio=config.mask_ratio)
        # loss.backward()
        # norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
        # optimizer.step()

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training at step {data_iter_step} epoch {epoch}")
            sys.exit(1)

        # loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(), clip_grad=config.clip_grad)

        # if (data_iter_step + 1) % accum_iter == 0:
        # cal the cor
        pred = pred.to('cpu').detach()
        samples = samples.to('cpu').detach()
        # pred = pred.transpose(1,2) #model_without_ddp.unpatchify(pred)
        pred = model_without_ddp.unpatchify(pred)
        # print(pred.shape)
        # print(samples.shape)
        # for p, s in zip(pred, samples):
        #     print(p[0], s[0])
        #     print(torch.cat([p[0].unsqueeze(0), s[0].unsqueeze(0)],axis=0))
        #     print(torch.corrcoef(torch.cat([p[0].unsqueeze(0), s[0].unsqueeze(0)],axis=0)))
        #     print(torch.corrcoef(torch.cat([p[0].unsqueeze(0), s[0].unsqueeze(0)],axis=0))[0,1])

        cor = torch.mean(torch.tensor([torch.corrcoef(torch.cat([p[0].unsqueeze(0), s[0].unsqueeze(0)],axis=0))[0,1] for p, s in zip(pred, samples)])).item()
        optimizer.zero_grad()

        total_loss.append(loss_value)
        total_cor.append(cor)
        if device == torch.device('cuda:0'):
            lr = optimizer.param_groups[0]["lr"]
            print('train_loss_step:', np.mean(total_loss), 'lr:', lr, 'cor', np.mean(total_cor))

    if log_writer is not None:
        lr = optimizer.param_groups[0]["lr"]
        log_writer.log('train_loss_step', np.mean(total_loss), step=epoch)
        log_writer.log('lr', lr, step=epoch)
        log_writer.log('cor', np.mean(total_cor), step=epoch)
        if start_time is not None:
            log_writer.log('time (min)', (time.time() - start_time)/60.0, step=epoch)
    if config.local_rank == 0:
        print(f'[Epoch {epoch}] loss: {np.mean(total_loss)}')

    return np.mean(total_cor)
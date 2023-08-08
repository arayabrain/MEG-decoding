from meg_ssl.trainers.base_trainer import BaseSSLTrainer
import torch
import os
import numpy as np
import math, sys
import torch
from meg_ssl.models.sc_mbm import utils as ut
from torch import inf
import numpy as np
from meg_ssl.trainers.base_trainer import BaseSSLTrainer
import tqdm
import os
from torch.utils.data import Dataset, DataLoader
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from hydra import compose, initialize

def clip_loss(target_emb, image_embeds):
    loss = 1 - torch.cosine_similarity(target_emb, image_embeds, dim=-1).mean()
    return loss


class AlignTrainer(BaseSSLTrainer):
    # @staticmethod
    # def get_lora_config(lora_config_name='lora.yaml'):
    #     with initialize(config_path='/Users/inoue/Desktop/RD/moonshot/MEG/codes/MEG-decoding/meg_ssl/task_configs/fine_tuning'):
    #         cfg = compose(lora_config_name)
    #     return cfg

    def other_setup(self):
        # scaler
        self.scaler = torch.cuda.amp.GradScaler()
        # decoder
        self.optimizer = torch.optim.AdamW(self.decoder.parameters(), lr=self.config.lr, betas=(0.9, 0.95))
        # meg_encoder
        self.train_meg_encoder = False if self.config.meg_encoder_finetune == 'none' else True
        if self.config.meg_encoder_finetune == 'lora': # use peft
            # get peft model
            lora_config = self.config.lora_config# self.get_lora_config('lora.yaml')
            self.meg_encoder = get_peft_model(self.meg_encoder, peft_config=LoraConfig(**lora_config))
            print('========== meg_encoder is transformed by peft ==========')
            self.meg_encoder.print_trainable_parameters()
            print('========================================================')
            # add params of meg_encoder to optimizer
            self.optimizer.add_param_group({'params': self.meg_encoder.parameters(), 'lr':5e-5})
        elif self.config.meg_encoder_finetune == 'full': # full fine-tuning
            # add params of meg_encoder to optimizer
            self.optimizer.add_param_group({'params': self.meg_encoder.parameters(), 'lr':5e-5})
            print('========== meg_encoder is trainable (full fine-tune) ==========')
        elif self.config.meg_encoder_finetune == 'none': # no fine-tuning
            print('========== meg_encoder is frozen ==========')
            pass
        else:
            raise ValueError('meg_encoder_finetune should be one of [rola, full, none]')

        # image_encoder
        self.train_image_encoder = False if self.config.image_encoder_finetune == 'none' else True
        if self.config.image_encoder_finetune == 'lora': # use peft
             # get peft model
            lora_config = self.config.lora_config# self.get_lora_config('lora.yaml')
            self.image_encoder = get_peft_model(self.image_encoder, peft_config=LoraConfig(**lora_config))
            print('========== image_encoder is transformed by peft ==========')
            self.image_encoder.print_trainable_parameters()
            print('========================================================')
            # add params of image_encoder to optimizer
            self.optimizer.add_param_group({'params': self.image_encoder.parameters(), 'lr':5e-5})
        elif self.config.image_encoder_finetune == 'full': # full fine-tuning
            # add params of image_encoder to optimizer
            self.optimizer.add_param_group({'params': self.image_encoder.parameters(), 'lr':5e-5})
            print('========== image_encoder is trainable (full fine-tune) ==========')
        elif self.config.image_encoder_finetune == 'none': # no fine-tuning
            print('========== image_encoder is frozen ==========')
            pass
        else:
            raise ValueError('image_encoder_finetune should be one of [rola, full, none]')





    def fit(self, meg_encoder, image_encoder, decoder, image_preprocessor:list, train_dataset:Dataset, val_dataset:Dataset,
            meg_encoder_ckpt_path:str=None, image_encoder_ckpt_path:str=None, decoder_ckpt_path:str=None):
        """_summary_

        Args:
            meg_encoder (_type_): meg_encoder to meg feature (basically pretrained)
            image_encoder (_type_): image encoder to image feature (basically pretrained)
            decoder (_type_): meg decoder to image feature
            train_dataset (data.Dataset): _description_
            val_dataset (data.Dataset): _description_
            ckpt_path (str, optional): resume_path of decoder.
        """
        self.before_train(meg_encoder, image_encoder, decoder, image_preprocessor, train_dataset, val_dataset,
                          meg_encoder_ckpt_path, image_encoder_ckpt_path, decoder_ckpt_path)
        self.train()
        self.after_train()
        pass

    def before_train(self, meg_encoder, image_encoder, decoder, image_preprocessor,
                     train_dataset:Dataset, val_dataset:Dataset,
                     meg_encoder_ckpt_path:str=None,
                     image_encoder_ckpt_path:str=None,
                     decoder_ckpt_path:str=None):
        self.meg_encoder = meg_encoder
        self.image_encoder = image_encoder
        self.decoder = decoder
        self.image_preprocessor = image_preprocessor if image_preprocessor is not None else []
        assert isinstance(self.image_preprocessor, list)

        os.makedirs(self.ckpt_dir, exist_ok=True)
        if meg_encoder_ckpt_path is not None:
            self.meg_encoder.load_state_dict(torch.load(meg_encoder_ckpt_path))
            print('meg_encoder load from ', meg_encoder_ckpt_path)
        if image_encoder_ckpt_path is not None:
            self.meg_encoder.load_state_dict(torch.load(image_encoder_ckpt_path))
            print('meg_encoder load from ', image_encoder_ckpt_path)
        if decoder_ckpt_path is not None:
            self.meg_encoder.load_state_dict(torch.load(decoder_ckpt_path))
            print('meg_encoder load from ', decoder_ckpt_path)
        self.meg_encoder = self.meg_encoder.to(self.device).eval()
        self.image_encoder = self.image_encoder.to(self.device).eval()
        self.decoder = self.decoder.to(self.device).eval()

        self.train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=self.num_workers)
        self.val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=self.num_workers)
        self.best_loss = np.inf
        self.other_setup()

    def train_one_epoch(self, epoch:int)->dict:
        self.decoder.train()
        if self.train_meg_encoder:
            self.meg_encoder.train()
        else:
            self.meg_encoder.eval()
        if self.train_image_encoder:
            self.image_encoder.train()
        else:
            self.image_encoder.eval()

        epoch_loss = []
        with tqdm.tqdm(self.train_loader) as pbar:
            for step, batch in enumerate(pbar):
                loss = self.train_one_step(epoch, step, batch)
                pbar.set_description("[Epoch {}] step={},loss={}".format(
                    epoch, step, np.mean(loss)))
                epoch_loss.append(loss)
        return {'train_loss': np.mean(epoch_loss)}

    def train_one_step(self, epoch, step, batch)->float:
        self.optimizer.zero_grad()

        batch_eeg, batch_image = batch
        if step % self.config.accum_iter == 0:
            ut.adjust_learning_rate(self.optimizer, step / len(self.train_loader) + epoch, self.config)
        samples = batch_eeg # data_dcit['eeg']
        samples = samples.to(self.device)
        batch_image = batch_image.to(self.device)
        # huggingface like processor
        for preprocessor in self.image_preprocessor:
            batch_image = preprocessor(batch_image)

        with torch.cuda.amp.autocast(enabled=True):
            latent, mask, ids_restore = self.meg_encoder.forward_encoder(samples, -1)
            assert mask is None
            latent_wo_cls = latent[:, 1:, :]
            pred = self.decoder(latent_wo_cls)
            # target
            target = self.image_encoder(**batch_image)
            loss = clip_loss(pred, target)

        self.scaler.scale(loss).backward(create_graph=True)
        self.scaler.unscale_(self.optimizer)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training at step {step} epoch {epoch}")
            sys.exit(1)

        self.optimizer.zero_grad()
        return loss_value

    def validation(self, epoch:int)->dict:
        self.decoder.eval()
        self.meg_encoder.eval()
        self.image_encoder.eval()
        val_loss = []
        with torch.no_grad():
            for step, batch in enumerate(self.val_loader):
                loss = self.val_one_step(step, batch)
                val_loss.append(loss)
        print('val loss: {}'.format(np.mean(val_loss)))
        return {'val_loss': np.mean(val_loss)}

    @torch.no_grad()
    def val_one_step(self, step, batch)->float:
        batch_eeg, batch_image = batch
        samples = batch_eeg # data_dcit['eeg']
        samples = samples.to(self.device)
        batch_image = batch_image.to(self.device)

        with torch.cuda.amp.autocast(enabled=True):
            latent, mask, ids_restore = self.meg_encoder.forward_encoder(samples, -1)
            assert mask is None
            latent_wo_cls = latent[:, 1:, :]
            pred = self.decoder(latent_wo_cls)
            # target
            target = self.image_encoder(**batch_image)
            loss = clip_loss(pred, target)

        loss_value = loss.item()

        pred = pred.to('cpu').detach()
        samples = samples.to('cpu').detach()
        if self.device_count > 1:
            pred = self.original_model.unpatchify(pred)
        pred = self.model.unpatchify(pred)
        cor = torch.mean(torch.tensor([torch.corrcoef(torch.cat([p[0].unsqueeze(0), s[0].unsqueeze(0)],axis=0))[0,1] for p, s in zip(pred, samples)])).item()

        return loss_value, cor


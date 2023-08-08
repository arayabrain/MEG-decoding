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
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, PeftModel
from hydra import compose, initialize
import torch.nn.functional as F
from meg_decoding.utils.loss import CLIPLoss

# def clip_loss(target_emb, image_embeds):
#     loss = 1 - torch.cosine_similarity(target_emb, image_embeds, dim=-1).mean()
#     return loss

# def contrastive_loss(logits, dim):
#     neg_ce = torch.diag(F.log_softmax(logits, dim=dim))
#     return -neg_ce.mean()
    
# def clip_loss(target_emb, image_embeds) -> torch.Tensor:
#     similarity = torch.nn.CosineSimilarity(dim=2)(target_emb.unsqueeze(0), image_embeds.unsqueeze(0))# torch.cosine_similarity(target_emb, image_embeds, dim=-1)
#     import pdb; pdb.set_trace()
#     caption_loss = contrastive_loss(similarity, dim=0)
#     image_loss = contrastive_loss(similarity, dim=1)
#     return (caption_loss + image_loss) / 2.0

# def clip_loss(target_embeds, image_embeds):
#     similarity = torch.empty(batch_size, batch_size).to(device)
#     for i in range(batch_size):
#         for j in range(batch_size):
#             similarity[i, j] = (x[i] @ y[j]) / max((x[i].norm() * y[j].norm()), 1e-8)


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
            collate_fn=None, meg_encoder_ckpt_path:str=None, image_encoder_ckpt_path:str=None, decoder_ckpt_path:str=None):
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
                          collate_fn, meg_encoder_ckpt_path, image_encoder_ckpt_path, decoder_ckpt_path)
        self.train()
        self.after_train()
        pass

    def before_train(self, meg_encoder, image_encoder, decoder, image_preprocessor,
                     train_dataset:Dataset, val_dataset:Dataset, collate_fn=None,
                     meg_encoder_ckpt_path:str=None,
                     image_encoder_ckpt_path:str=None,
                     decoder_ckpt_path:str=None):
        self.meg_encoder = meg_encoder
        self.image_encoder = image_encoder
        self.decoder = decoder
        self.image_preprocessor = image_preprocessor if image_preprocessor is not None else []
        assert isinstance(self.image_preprocessor, list)

        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=collate_fn)
        self.val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=collate_fn)
        self.best_loss = np.inf
        if self.config.criterion == 'clip':
            self.criterion = CLIPLoss(self.config.clip).train().to(self.device)
        elif self.config.criterion == 'MSE':
            self.criterion = torch.nn.MSELoss(reduction="mean")
        else:
            raise ValueError('criterion should be clip or MSE')
        
        self.other_setup()
        
        if meg_encoder_ckpt_path is not None:
            if 'peft' in meg_encoder_ckpt_path:
                self.meg_encoder = PeftModel.from_pretrained(
                                            self.meg_encoder,
                                            meg_encoder_ckpt_path,
                                            torch_dtype=torch.float32,   
                                        )
            else:
                self.meg_encoder.load_state_dict(torch.load(meg_encoder_ckpt_path))
            print('meg_encoder load from ', meg_encoder_ckpt_path)
        if image_encoder_ckpt_path is not None:
            if 'peft' in image_encoder_ckpt_path:
                self.image_encoder = PeftModel.from_pretrained(
                                            self.image_encoder,
                                            image_encoder_ckpt_path,
                                            torch_dtype=torch.float32,   
                                        )
            else:
                self.image_encoder.load_state_dict(torch.load(image_encoder_ckpt_path))
            print('image_encoder load from ', image_encoder_ckpt_path)
        if decoder_ckpt_path is not None:
            self.decoder.load_state_dict(torch.load(decoder_ckpt_path))
            print('decoder load from ', decoder_ckpt_path)
        self.meg_encoder = self.meg_encoder.to(self.device).eval()
        self.image_encoder = self.image_encoder.to(self.device).eval()
        self.decoder = self.decoder.to(self.device).eval()

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
        if self.config.criterion == 'clip':
            self.criterion.train()

        epoch_loss = []
        epoch_acc = []
        with tqdm.tqdm(self.train_loader) as pbar:
            for step, batch in enumerate(pbar):
                loss, acc = self.train_one_step(epoch, step, batch)
                pbar.set_description("[Epoch {}] step={},loss={}".format(
                    epoch, step, np.mean(loss)))
                epoch_loss.append(loss)
                epoch_acc.append(acc)
        return {'train_loss': np.mean(epoch_loss), 'train_acc': np.mean(epoch_acc)}

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
            latent_wo_cls = latent[:, 1:, :] # torch.Size([100, 52, 1024])  208(time_len) / 4 (patch_size) = 52 (num_patch)
            pred = self.decoder(latent_wo_cls)
            # target
            # target = self.image_encoder(**batch_image)# .image_embeds
            if self.train_image_encoder:
                target = self.image_encoder.encode_image(batch_image)
            else:
                with torch.no_grad():
                    target = self.image_encoder.encode_image(batch_image)
            loss = self.criterion(pred, target)
            acc = self.calc_acc(pred, target, self.device)

        # self.scaler.scale(loss).backward(create_graph=True)
        # self.scaler.unscale_(self.optimizer)
        # self.scaler.step(self.optimizer)
        # self.scaler.update()
        loss.backward()
        self.optimizer.step()

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training at step {step} epoch {epoch}")
            sys.exit(1)

        self.optimizer.zero_grad()
        return loss_value, acc

    def validation(self, epoch:int)->dict:
        self.decoder.eval()
        self.meg_encoder.eval()
        self.image_encoder.eval()
        if self.config.criterion == 'clip':
            self.criterion.eval()
        val_loss = []
        val_acc = []
        with torch.no_grad():
            for step, batch in enumerate(self.val_loader):
                loss, acc = self.val_one_step(step, batch)
                val_loss.append(loss)
                val_acc.append(acc)
        print('val loss: {} acc: {}'.format(np.mean(val_loss), np.mean(val_acc)))
        return {'val_loss': np.mean(val_loss), 'val_acc': np.mean(val_acc)}

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
            # target = self.image_encoder(**batch_image)# .image_embeds
            target = self.image_encoder.encode_image(batch_image)
            loss = self.criterion(pred, target)
            acc = self.calc_acc(pred, target, self.device)

        loss_value = loss.item()
        return loss_value, acc
    @staticmethod
    def calc_acc(target_embs, meg_embs, device):
        batch_size = meg_embs.size(0)
        diags = torch.arange(batch_size).to(device)
        similarity = torch.empty(batch_size, batch_size).to(device)
        for i in range(batch_size):
            for j in range(batch_size):
                similarity[i, j] = (meg_embs[i] @ target_embs[j]) / max((meg_embs[i].norm() * target_embs[j].norm()), 1e-8)

        similarity = similarity.T
        top1accuracy = (similarity.argmax(axis=1) == diags).to(torch.float).mean().item()
        return top1accuracy

    def save_model(self, filename:str):
        filepath =  os.path.join(self.ckpt_dir, filename)
        torch.save(self.meg_encoder.state_dict(),filepath)
        print('meg_encoder is saved as ', filepath)
        if self.train_image_encoder:
            if self.config.image_encoder_finetune=='lora':
                filepath = os.path.join(self.ckpt_dir, 'peft-image_encoder_' + filename)
                self.image_encoder.save_pretrained(filepath)
            else:
                filepath = os.path.join(self.ckpt_dir, 'image_encoder_' + filename)
                torch.save(self.image_encoder.state_dict(), filepath)
            print('image_encoder is saved as ', filepath)
        else:
            print('image encoder is frozen. then, not saved')
        if self.train_meg_encoder:
            if self.config.meg_encoder_finetune=='lora':
                filepath = os.path.join(self.ckpt_dir, 'peft-meg_encoder_' + filename)
                self.meg_encoder.save_pretrained(filepath)
            else:
                filepath = os.path.join(self.ckpt_dir, 'meg_encoder_' + filename)
                torch.save(self.meg_encoder.state_dict(), filepath)
            print('meg_encoder is saved as ', filepath)
        else:
            print('meg encoder is frozen. then, not saved')

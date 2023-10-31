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
from PIL import Image
import copy
import wandb
import peft

def cosin_sim(target_emb, image_embeds):
    loss = 1 - torch.cosine_similarity(target_emb, image_embeds, dim=-1).mean()
    return loss

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


class DiffusionTrainer(BaseSSLTrainer):
    # @staticmethod
    # def get_lora_config(lora_config_name='lora.yaml'):
    #     with initialize(config_path='/Users/inoue/Desktop/RD/moonshot/MEG/codes/MEG-decoding/meg_ssl/task_configs/fine_tuning'):
    #         cfg = compose(lora_config_name)
    #     return cfg

    def other_setup(self):
        if self.generator.train_cond_stage_only:
            print(f"{self.generator.__class__.__name__}: Only optimizing conditioner params!")
            cond_parms = [p for n, p in self.generator.named_parameters()
                    if 'attn2' in n or 'time_embed_condtion' in n or 'norm2' in n]
            params = list(self.generator.cond_stage_model.parameters()) + cond_parms

            for p in params:
                p.requires_grad = True
        else:
            params = list(self.generator.model.parameters())
            if self.generator.cond_stage_trainable:
                print(f"{self.generator.__class__.__name__}: Also optimizing conditioner params!")
                params = params + list(self.generator.cond_stage_model.parameters())
            if self.generator.learn_logvar:
                print('Diffusion model optimizing logvar')
                params.append(self.generator.logvar)

        # decoder
        ## generator parametersは不足してる？conditionとfirst stageが入っているか怪しい
        self.optimizer = torch.optim.AdamW(params, lr=self.config.lr, betas=(0.9, 0.95))


    def fit(self, generator, image_preprocessor:list, train_dataset:Dataset, val_dataset:Dataset,
            collate_fn=None):
        """_summary_

        Args:
            meg_encoder (_type_): meg_encoder to meg feature (basically pretrained)
            image_encoder (_type_): image encoder to image feature (basically pretrained)
            decoder (_type_): meg decoder to image feature
            train_dataset (data.Dataset): _description_
            val_dataset (data.Dataset): _description_
            ckpt_path (str, optional): resume_path of decoder.
        """
        self.before_train(generator, image_preprocessor, train_dataset, val_dataset,
                          collate_fn)
        self.train()
        self.after_train()
        pass

    def before_train(self, generator, image_preprocessor,
                     train_dataset:Dataset, val_dataset:Dataset, collate_fn=None,
                     ):
        self.generator = generator.to(self.device).eval()
        if isinstance(self.generator, peft.peft_model.PeftModel):
            self.peft_flag = True
        else:
            self.peft_flag = False
        self.image_preprocessor = image_preprocessor if image_preprocessor is not None else []
        assert isinstance(self.image_preprocessor, list)

        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=collate_fn)
        self.val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=collate_fn)
        self.best_acc = 0# np.inf
        self.reconst_dir = os.path.join(self.config.reconst_dir, 'val')

        assert self.config.trainable_cond_model or self.config.trainable_generator_model

        if self.peft_flag:
            pass
        else:
            if self.config.trainable_generator_model:
                self.generator.train_cond_stage_only = False
                self.generator.unfreeze_whole_model()
                self.generator.freeze_first_stage()
                print('generator is trainable but first stage model')
            else:
                # self.generator.freeze_whole_model()
                print('generator is frozen')

            if self.config.trainable_cond_model:
                self.generator.cond_stage_trainable = True
                self.generator.unfreeze_cond_stage()
                print('meg encoder is trainable')
            else:
                self.generator.cond_stage_trainable = False
                self.generator.cond_stage_model.eval()
                self.generator.cond_stage_model.train = self.disabled_train
                for param in self.generator.cond_stage_model.parameters():
                    param.requires_grad = False
                # self.generator.freeze_cond_stage()
                print('meg encoder is frozen')


        os.makedirs(self.reconst_dir, exist_ok=True)
        self.other_setup()

    @staticmethod
    def disabled_train(self, mode=True):
        """Overwrite model.train with this function to make sure train/eval mode
        does not change anymore."""
        return self

    def update_loss_logs(self, loss_dict:dict, loss_logs:dict, prefix:str=None):
        for key, value in loss_dict.items():
            if prefix is not None:
                key = 'train_' + key
            if key not in loss_logs:
                loss_logs[key] = []
            loss_logs[key].append(value)
        return loss_logs

    def train_one_epoch(self, epoch:int)->dict:
        self.generator.train()
        self.generator.first_stage_model.eval() # image encoderは、trainingの対象外
        if self.config.trainable_cond_model:
            self.generator.cond_stage_model.train()
        else:
            self.generator.cond_stage_model.eval()

        loss_logs = {}

        with tqdm.tqdm(self.train_loader) as pbar:
            for step, batch in enumerate(pbar):
                # import pdb; pdb.set_trace()
                # DEBUG START
                # befora_state = copy.deepcopy(self.generator.state_dict())
                # DEBUG END
                # barch:: eeg: 5,22,208, image: 5,512,512,3,  label: 5, image_raw:{'pixel_values': 5,3,224,224}
                # import pdb; pdb.set_trace()
                total_loss, loss_dict, c = self.train_one_step(epoch, step, batch)
                # DEBUG START
                # after_state = self.generator.state_dict()
                # self.compare_weight(before_state,after_state, target_flag=False)
                # self.compare_weight(before_state, after_state, target_flag=True)
                # module_list = self.get_largest_module_names(before_state)
                # self.compare_weight(before_state, after_state, target_flag=True, module_names=module_list)
                # self.get_grad(self.generator, module_names=module_list)
                # self.get_weight_sum_avg(after_state, module_name='model')
                # import pdb; pdb.set_trace()
                # DEBUG END
                pbar.set_description("[Epoch {}] step={},loss={}".format(
                    epoch, step, total_loss))
                loss_dict.update({'train/total_loss': total_loss})
                self.update_loss_logs(loss_dict, loss_logs)
        ret = {key: np.mean(value) for key, value in loss_logs.items()}
        return ret

    def train_one_step(self, epoch, step, batch)->float:
        self.generator.on_train_batch_start(batch, step)
        self.optimizer.zero_grad()

        # shared step
        self.generator.freeze_first_stage()
        # x: MEG latent = batch[self.generator.first_stage_key]をfirst_stage encoderに入れたもの
        # c: batch[self.cond_stage_key]をcond_stage encoderに入れたもの
        # label: batch['label']
        # image_raw: batch['image_raw']
        
        x, c, label, image_raw = self.generator.get_input(batch, self.generator.first_stage_key)
        x = x.to(self.device)
        c = c.to(self.device)
        label = label.to(self.device)
        image_raw = image_raw.to(self.device)

        if self.generator.return_cond:
            loss, cc = self.generator(x, c, label, image_raw)
            total_loss = loss[0] # * 1e5 # DEBUG
            loss_dict = loss[1]
            total_loss.backward()
            self.optimizer.step()
            return total_loss.item(), {k:v.item() for k, v in loss_dict.items()}, cc
        else:
            loss = self.generator(x, c, label, image_raw)
            total_loss = loss[0] # * 1e5 # DEBUG
            loss_dict = loss[1]
            total_loss.backward()
            self.optimizer.step()
            return total_loss.item(), {k:v.item() for k, v in loss_dict.items()}, None

    @staticmethod
    def compare_weight(state1, state2, target_flag=True, module_names:list=None):
        print(f'======================================= TARGET FLAG = {target_flag}=====================================================')
        if module_names is not None:
            flag_per_module = {k : [] for k in module_names}
        for key in state1.keys():
            val1 = state1[key]
            val2 = state2[key]
            flag = (val1==val2).cpu().numpy().all()
            if module_names is not None:
                module_name = key.split('.')[0]
                flag_per_module[module_name].append(flag)
            else:
                if flag == target_flag:
                    print(key, '::\t', flag)

        print('=============weight update judge for MODULE=============')
        if module_names is not None:
            for key, value in flag_per_module.items():
                print(key, np.all(value))

    @staticmethod
    def get_weight_sum_avg(state, module_name='all'):
        sum_list = []
        for key, value in state.items():
            if module_name == 'all':
                pass
            elif module_name in key:
                pass
            else:
                continue
            sum_list.append(value.sum().item())

        avg_sum = np.mean(sum_list)
        print('=============average of weight sum=============')
        print(module_name, avg_sum)
        return avg_sum

    @staticmethod
    def get_grad(model, module_names:list=None):
        if module_names is not None:
            grad_per_module = {k : [] for k in module_names}
            require_grad_per_module = {k : [] for k in module_names}


        for name, parameter in model.named_parameters():
            grads = parameter.grad
            grad_sum = grads.sum().item() if grads is not None else 0
            if module_names is not None:
                module_name = name.split('.')[0]
                grad_per_module[module_name].append(grad_sum)
                require_grad_per_module[module_name].append(parameter.requires_grad)
            else:
                print(name, grad_sum)

        print('============grad and requires grad for MODULE================')
        if module_names is not None:
            for key, value in grad_per_module.items():
                print(key, np.sum(value), np.all(require_grad_per_module[key]))



    @staticmethod
    def get_largest_module_names(state):
        largest_module_name_list = []
        for key in state:
            largest_module_name = key.split('.')[0]
            if largest_module_name in largest_module_name_list:
                continue
            else:
                largest_module_name_list.append(largest_module_name)
        print(largest_module_name_list)
        return largest_module_name_list


    def validation(self, epoch:int)->dict:
        self.generator.eval()
        self.generator.first_stage_model.eval()
        self.generator.cond_stage_model.eval()
        metrics_logs = {}
        limit=10
        with torch.no_grad():
            for step, batch in enumerate(self.val_loader):
                metrics, grid_imgs = self.val_one_step(step, batch, limit=limit)
                self.update_loss_logs(metrics, metrics_logs, prefix=None)
                if step < 2:
                    savefile = os.path.join(self.reconst_dir, f'test-{epoch}-{step}.png')
                    grid_imgs.save(savefile)
                    print('generated images is saved as ', savefile)

                    if self.config.allow_full_validation and (metrics['val/top-1-class (max)'] > self.generator.run_full_validation_threshold):
                        print('full validation start')
                        limit = None
                        pass
                    else:
                        pass # take so much time
                else:
                    break
        # import pdb; pdb.set_trace()
        metrics_logs = {key: np.mean(value) for key, value in metrics_logs.items()}
        print(metrics_logs)
        return metrics_logs

    @torch.no_grad()
    def val_one_step(self, step, batch, limit=5)->float:
        # generate
        # import pdb; pdb.set_trace()
        grid, all_samples, state = self.generator.generate(batch,
                                                           ddim_steps=self.generator.ddim_steps,
                                                           num_samples=1,
                                                           limit=limit)
        #
        metric, metric_list = self.generator.get_eval_metric(all_samples, avg=self.config.eval_avg)
        grid_imgs = Image.fromarray(grid.astype(np.uint8))
        # import pdb; pdb.set_trace()
        metric_dict = {f'val/{k}':v for k, v in zip(metric_list, metric)}

        return metric_dict, grid_imgs

    def after_epoch(self, epoch:int, train_metrics:dict):
        print('train: \n', train_metrics)
        if self.global_rank == 0:
            # lr
            for param_group in self.optimizer.param_groups:
                lr = param_group['lr']
                break
            train_metrics.update({'lr':lr})
            print('lr: ', lr)
            # logging
            self.pkl_logger.log(train_metrics, 'train')
            # validation
            if epoch % self.val_check_interval == 0:
                val_metrics = self.validation(epoch)
                val_metrics.update({'epoch':epoch})
                # logging
                self.pkl_logger.log(val_metrics, 'val')
                # save model
                if val_metrics['val/top-1-class (max)'] > self.best_acc:
                    self.best_acc = val_metrics['val/top-1-class (max)']
                    print('!! best top-1-class (max) is updated to ', self.best_acc)
                    self.save_model('best.pth')
                self.save_model('current.pth')
                train_metrics.update(val_metrics)
            if self.usewandb:
                wandb.log(train_metrics)


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
        if self.generator.train_cond_stage_only:
            filepath =  os.path.join(self.ckpt_dir, filename.replace('.pth', '-meg_enc.pth'))
            torch.save(self.generator.cond_stage_model.state_dict(), filepath)
            print('cond_stage_model is saved as ', filepath)
            # attnの分
            filepath =  os.path.join(self.ckpt_dir, filename.replace('.pth', '-generator.pth'))
            torch.save(self.generator.state_dict(), filepath)
            print('generator is saved as ', filepath)
        else:
            filepath =  os.path.join(self.ckpt_dir, filename.replace('.pth', '-generator.pth'))
            torch.save(self.generator.state_dict(), filepath)
            print('generator is saved as ', filepath)

            if self.generator.cond_stage_trainable:
                filepath =  os.path.join(self.ckpt_dir, filename.replace('.pth', '-meg_enc.pth'))
                torch.save(self.generator.cond_stage_model.state_dict(), filepath)
                print('cond_stage_model is saved as ', filepath)

            if self.generator.learn_logvar:
                filepath =  os.path.join(self.ckpt_dir, filename.replace('.pth', '-learn_logvar.pth'))
                torch.save(self.generator.learn_logvar.state_dict(), filepath)
                print('learn_logvar is saved as ', filepath)
# -*- coding: utf-8 -*-
# @Time        : 2023/3/16 23:59
# @Author      : ssxy00, lz
# @File        : lgcm_trainer.py
# @Description :


import os
import math
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import AdamW

from torch.utils.tensorboard import SummaryWriter

from utils import create_pairwise_dist, create_utterance_mask


class LGCMTrainer:
    def __init__(self, model, args, train_dataset, valid_dataset, device, vocab):
        self.args = args
        self.device = device
        self.model = model.to(device)
        self.vocab = vocab
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset

        # train config
        self.n_epochs = args.n_epochs
        self.batch_size = args.batch_size
        self.gradient_accumulate_steps = args.gradient_accumulate_steps
        self.clip_grad = args.clip_grad

        # checkpoint
        self.save_model_dir = args.save_model_dir
        self.save_interval = args.save_interval

        # log
        self.writer = SummaryWriter(args.log_dir)

        # optimizer
        lr = args.lr
        base_optimizer = AdamW(model.parameters(), lr=lr, correct_bias=True)
        self.optimizer = base_optimizer

    def state_dict(self):
        return {'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict()}

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict['model'], strict=False)
        self.optimizer.load_state_dict(state_dict['optimizer'])

    def _eval_train(self, epoch):
        self.model.train()
        ave_lm_loss = 0
        train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                                      collate_fn=self.train_dataset.collate_func, num_workers=4)

        tqdm_data = tqdm(train_dataloader, desc='Train (epoch #{})'.format(epoch))

        for i, data in enumerate(tqdm_data):
            data = {key: data[key].to(self.device) for key in data}
            pairwise_dist = create_pairwise_dist(data["turn_position_ids"])
            utterance_mask = create_utterance_mask(data["turn_position_ids"])

            model_out = self.model(src=data['context'],
                                   tgt=data['tgt'],
                                   src_type_ids=data['context_type_ids'],
                                   tgt_type_ids=data['tgt_type_ids'],
                                   src_position_ids=data['context_position_ids'],
                                   tgt_position_ids=data['tgt_position_ids'],
                                   turn_position_ids=data['turn_position_ids'],
                                   lm_labels=data["tgt"],
                                   pairwise_dist=pairwise_dist,
                                   utterance_mask=utterance_mask
                                   )

            lm_loss = model_out["lm_loss"]
            loss = lm_loss / self.gradient_accumulate_steps
            loss.backward()
            if self.clip_grad is not None:
                for group in self.optimizer.param_groups:
                    nn.utils.clip_grad_norm_(group['params'], self.clip_grad)

            if (i + 1) % self.gradient_accumulate_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            ave_lm_loss = (ave_lm_loss * i + lm_loss.item()) / (i + 1)

            tqdm_data.set_postfix({'lm_loss': lm_loss.item(),
                                   'ave_lm_loss': ave_lm_loss,
                                   'lr': self.optimizer.param_groups[0]['lr']})
            self.writer.add_scalar('Train/loss', ave_lm_loss, (epoch - 1) * len(tqdm_data) + i + 1)
            self.writer.add_scalar('Train/lr', self.optimizer.param_groups[0]['lr'],
                                   (epoch - 1) * len(tqdm_data) + i + 1)
        print(f"train {epoch}:")
        print({'ave_lm_loss': ave_lm_loss, 'lr': self.optimizer.param_groups[0]['lr']})

    def _eval_valid(self, epoch):
        self.model.eval()
        valid_dataloader = DataLoader(self.valid_dataset, batch_size=self.batch_size,
                                      collate_fn=self.valid_dataset.collate_func, num_workers=4)

        tqdm_data = tqdm(valid_dataloader, desc='Valid (epoch #{})'.format(epoch))
        ave_lm_loss = 0
        with torch.no_grad():
            for i, data in enumerate(tqdm_data):
                data = {key: data[key].to(self.device) for key in data}
                pairwise_dist = create_pairwise_dist(data["turn_position_ids"])
                utterance_mask = create_utterance_mask(data["turn_position_ids"])
                model_out = self.model(src=data['context'],
                                       tgt=data['tgt'],
                                       src_type_ids=data['context_type_ids'],
                                       tgt_type_ids=data['tgt_type_ids'],
                                       src_position_ids=data['context_position_ids'],
                                       tgt_position_ids=data['tgt_position_ids'],
                                       turn_position_ids=data['turn_position_ids'],
                                       lm_labels=data["tgt"],
                                       pairwise_dist=pairwise_dist,
                                       utterance_mask=utterance_mask
                                       )

                lm_loss = model_out["lm_loss"]
                ave_lm_loss = (ave_lm_loss * i + lm_loss.item()) / (i + 1)

                tqdm_data.set_postfix({'lm_loss': lm_loss.item(),
                                       'ave_lm_loss': ave_lm_loss,
                                       'ppl': math.exp(ave_lm_loss)})
        print(f"valid {epoch}:")
        print({'ave_lm_loss': ave_lm_loss,
               'ppl': math.exp(ave_lm_loss)})
        self.writer.add_scalar('Valid/loss', ave_lm_loss, epoch)
        self.writer.add_scalar('Valid/ppl', math.exp(ave_lm_loss), epoch)

    def train(self, last_epoch=0):
        print('begin to train')
        for epoch_idx in range(last_epoch + 1, self.n_epochs + 1):
            self._eval_train(epoch_idx)
            self._eval_valid(epoch_idx)
            if epoch_idx % self.save_interval == 0:
                save_dir = os.path.join(self.save_model_dir, f"checkpoint{epoch_idx}.pt")
                torch.save(self.state_dict(), save_dir)

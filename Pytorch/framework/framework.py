import logging
import torch
from tqdm import tqdm
import time
import torch.nn as nn
import sys
import numpy as np
import os

class FrameWork(nn.Module):
    def __init__(self,
                model,
                tokenizer,
                train_loader,
                dev_loader,
                test_loader,
                save_ckpt,
                batch_size,
                max_epoch,
                lr,
                PAD_ID,
                warmup_steps,
                # optimizer,
                use_prefix,
                opt = 'adamw',
                weight_decay = 1e-5):
        super().__init__()
        # model
        self.model = model
        self.tokenizer=tokenizer
        self.parallel_model = nn.DataParallel(self.model)

        # data
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.PAD_ID = PAD_ID

        # training config
        self.save_ckpt = save_ckpt
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.lr = lr
        self.warmup_steps = warmup_steps

        # optimizer
        if opt == 'adamw':
            from transformers import AdamW
            params = list(self.model.named_parameters())
            # print(params)
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            grouped_params = [
                {
                    'params': [p for n, p in params if not any(nd in n for nd in no_decay) and p.requires_grad],
                    'weight_decay': 0.0,
                    'lr': lr,
                    'ori_lr': lr
                },
                {
                    'params': [p for n, p in params if any(nd in n for nd in no_decay) and p.requires_grad], 
                    'weight_decay': 0.0,
                    'lr': lr,
                    'ori_lr': 0
                }
            ]
            print(grouped_params)
            self.optimizer = AdamW(grouped_params, correct_bias=False,eps = 1e-7)
        else:
            raise 

        if warmup_steps > 0 and self.train_loader is not None:
            from transformers import get_linear_schedule_with_warmup
            training_steps = self.train_loader.dataset.__len__() // batch_size * self.max_epoch
            self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup_steps,num_training_steps=training_steps)
        else:
            self.scheduler = None

        # cuda
        if torch.cuda.is_available():
            self.cuda()

    def train_model(self, save_to, max_grad_norm=1.0):
        best_val_loss = 1e6
        best_epoch = -1
        global_step = 1
        
        self.fast_evaluate(self.dev_loader)

        for epoch in range(self.max_epoch):
            start_time = time.time()
            losses = 0
            self.model.train()
            
            logging.info("=== Epoch %d train ==="%epoch)
            # self.generate(save=True, dataloader=self.test_loader, output_dir='./temp.txt', decode_strategy='top-k', temperature=0.9, top_p=1, top_k=40, maxlen=256, verbose=True, limit=3)
            t = tqdm(self.train_loader)
            for _iter, batch in enumerate(t):
                if torch.cuda.is_available():
                    for i in range(len(batch)):
                        try:
                            batch[i] = batch[i].cuda
                        except:
                            pass
                
                
                self.optimizer.zero_grad()
                
                loss = self.parallel_model(batch=batch)
                loss.backward()

                global_step += 1
                losses += loss.item()
                t.set_postfix({'loss': loss.item()})
                t.set_postfix({'avg_loss': losses / global_step})
                
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

            train_loss = losses / global_step
            epoch_time = time.time() - start_time

            print("=== Epoch %d val ===" % epoch)
            val_loss, val_ppl = self.fast_evaluate(self.dev_loader)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_ppl = val_ppl
                best_epoch = epoch + 1

                with open(self.save_ckpt, 'wb') as fout:
                    torch.save(self.model.state_dict(), fout)
                
                print("Best ckpt saved.")
                self.generate(limit=5, save=True,dataloader=self.test_loader, output_dir='./temp.txt', decode_strategy='top-p', temperature=1.0, top_p=0.9, top_k=40, maxlen=256, verbose=True)
            
            print("Epoch " + str(epoch+1) + " of " + str(self.max_epoch) + " took " + str(epoch_time) + "s")
            print("  training loss:                 " + str(train_loss))
            print("  validation loss:               " + str(val_loss))
            print("  validation perplexity:         " + str(val_ppl))
            print("  best epoch:                    " + str(best_epoch))
            print("  best validation perplexity:    " + str(best_val_ppl))

        # torch.save(self.model, os.path.join("./result", save_to + ".pth"))

    def fast_evaluate(self, dataloader):
        PAD_ID = self.PAD_ID
        device = torch.device('cuda')
        self.model.eval()
        all_loss = 0
        steps = 0
        for _iter, batch in enumerate(dataloader):
            with torch.no_grad():
                input_ids = batch['input_ids'].to(device)
                batch_size = input_ids.shape[0]
            
                loss = self.parallel_model(batch=batch)
                all_loss += loss.cpu()
                steps += 1
        loss = all_loss / steps
        ppl = np.exp(loss)
        self.model.train()
        return loss, ppl

    def generate(self, dataloader, output_dir, decode_strategy, temperature, top_p, top_k, maxlen=128, save=False, verbose=False,limit=-1, use_prefix=False):
        generated_sentences = []
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        t = tqdm(dataloader)
        count = -1
        for _iter, batch in enumerate(t):
            count += 1
            if (limit > 0 and count == limit):
                break
            output = self.model.generate(batch=batch, tokenizer=self.tokenizer, device=device, PAD_ID=self.PAD_ID, 
                maxlen=maxlen, decode_strategy=decode_strategy, temperature=temperature, top_p=top_p, top_k=top_k)
            generated_sentences.append(output[0])
            if(verbose):
                print(f'【generated sentence】: {output[0]}')
                print('【target sentence】: {}'.format(batch['tgt_text'][0]))

        if(save):
            with open(output_dir,'w') as f:
                for sent in generated_sentences:
                    f.write(sent+'\n')
        

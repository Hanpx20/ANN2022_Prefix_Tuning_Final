import os
import argparse
from omegaconf import OmegaConf
import sys
sys.path.append('.')
from utils.dataloader import *
from utils.pretrained import *
from utils.pipeline import *
from gpt2.model import TfmrLMHeadModel, TransposeLinear, GenerationModel
from model import PrefixGPTModel
from gpt2.config import ModelConfig
from framework.framework import FrameWork
import jittor
import json
from jittor import optim
import collections
from transformers import AutoTokenizer, GPT2Tokenizer

import os


parser = argparse.ArgumentParser()
parser.add_argument('--name',type=str, default="run")
parser.add_argument('--task',type=str, default="webnlg")
parser.add_argument('--output',type=str, default="output")
parser.add_argument('--config',type=str, default="baseline.yaml")
parser.add_argument('--use_prefix', type=int, default=1)
parser.add_argument('--prefix_len', type=int, default=10)
parser.add_argument('--use_prefixMLP', type=int, default=1)
parser.add_argument('--embedding_only', type=int, default=0)
args = parser.parse_args()

def set_seed(seed):
    jittor.set_global_seed(seed)

def run():
    set_seed(42)
    cfg = OmegaConf.load(args.config)

    # dataset
    if args.task == "webnlg":
        train_dataset = get_train_dataset(cfg.dataset)
        dev_dataset = get_dev_dataset(cfg.dataset)
        test_dataset = get_test_dataset(cfg.dataset)
    
    state_dict = jittor.load('gpt.pth')

    tokenizer = AutoTokenizer.from_pretrained('gpt2-medium', use_fast=False)
    # tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.add_special_tokens({'pad_token': '<pad>'})
    if args.task == "webnlg":
        train_dataloader = get_webnlg_train_dataloader(dataset=train_dataset,template=None, batch_size=cfg.batch_size, max_length=cfg.max_length, tokenizer=tokenizer, PAD_ID='<pad>', EOS_ID='<|endoftext|>', shuffle=True)
        dev_dataloader = get_webnlg_train_dataloader(dataset=dev_dataset,template=None, batch_size=cfg.batch_size, max_length=cfg.max_length, tokenizer=tokenizer, PAD_ID='<pad>', EOS_ID='<|endoftext|>', shuffle=False)
        test_dataloader = get_webnlg_train_dataloader(dataset=test_dataset,template=None, batch_size=1, max_length=cfg.max_length, tokenizer=tokenizer, PAD_ID='<pad>', EOS_ID='<|endoftext|>', shuffle=False)
    elif args.task == "e2e":
        train_dataloader = get_e2e_train_dataloader(path="./data/e2e_data/src1_train.txt",template=None, batch_size=cfg.batch_size, max_length=cfg.max_length, tokenizer=tokenizer, PAD_ID='<pad>', EOS_ID='<|endoftext|>', shuffle=True)
        dev_dataloader = get_e2e_train_dataloader(path="./data/e2e_data/src1_valid.txt",template=None, batch_size=cfg.batch_size, max_length=cfg.max_length, tokenizer=tokenizer, PAD_ID='<pad>', EOS_ID='<|endoftext|>', shuffle=False)
        test_dataloader = get_e2e_train_dataloader(path="./data/e2e_data/src1_test.txt",template=None, batch_size=1, max_length=cfg.max_length, tokenizer=tokenizer, PAD_ID='<pad>', EOS_ID='<|endoftext|>', shuffle=False)
    elif args.task == "animal":
        train_dataloader = get_AP_train_dataloader(path="./data/animal_person/animal_train.json", template=None,
                                                    batch_size=cfg.batch_size, max_length=cfg.max_length,
                                                    tokenizer=tokenizer, PAD_ID='<pad>', EOS_ID='<|endoftext|>',
                                                    shuffle=True)
        dev_dataloader = get_AP_train_dataloader(path="./data/animal_person/animal_valid.json", template=None,
                                                  batch_size=cfg.batch_size, max_length=cfg.max_length,
                                                  tokenizer=tokenizer, PAD_ID='<pad>', EOS_ID='<|endoftext|>',
                                                  shuffle=False)
        test_dataloader = get_AP_train_dataloader(path="./data/animal_person/animal_test.json", template=None,
                                                   batch_size=1, max_length=cfg.max_length,
                                                   tokenizer=tokenizer, PAD_ID='<pad>', EOS_ID='<|endoftext|>',
                                                   shuffle=False)
    elif args.task == "person":
        train_dataloader = get_AP_train_dataloader(path="./data/animal_person/person_train.json", template=None,
                                                    batch_size=cfg.batch_size, max_length=cfg.max_length,
                                                    tokenizer=tokenizer, PAD_ID='<pad>', EOS_ID='<|endoftext|>',
                                                    shuffle=True)
        dev_dataloader = get_AP_train_dataloader(path="./data/animal_person/person_valid.json", template=None,
                                                  batch_size=cfg.batch_size, max_length=cfg.max_length,
                                                  tokenizer=tokenizer, PAD_ID='<pad>', EOS_ID='<|endoftext|>',
                                                  shuffle=False)
        test_dataloader = get_AP_train_dataloader(path="./data/animal_person/person_test.json", template=None,
                                                   batch_size=1, max_length=cfg.max_length,
                                                   tokenizer=tokenizer, PAD_ID='<pad>', EOS_ID='<|endoftext|>',
                                                   shuffle=False)
    else:
        print("Task not supported.")
        exit(0)
    
    jittor.flags.use_cuda = 1 # 0->cpu, 1->try to use gpu, 2->force using gpu

    with open('config.json','r') as fin:    
        model_config = json.load(fin)
        config = ModelConfig(**model_config)
    
    if args.use_prefix:
        pretrain_model = PrefixGPTModel(config=config)
        pretrain_model.base_gpt_model.resize_token_embeddings(len(tokenizer))
        pretrain_model.base_gpt_model.load_state_dict(state_dict)
        pretrain_model.set_gradient()
    else:
        pretrain_model = TfmrLMHeadModel(config)
        pretrain_model.resize_token_embeddings(len(tokenizer))
        pretrain_model.load_state_dict(state_dict)

    # for name, param in pretrain_model.named_parameters():
    #    print(name, param.requires_grad)

    generation_model = GenerationModel(pretrain_model, False, tokenizer, tokenizer.encode('<pad>')[0])

    optimizer = optim.AdamW(pretrain_model.parameters(), lr=cfg.lr, weight_decay=0)

    framework = FrameWork(
        model=generation_model,
        tokenizer=tokenizer,
        train_loader=train_dataloader,
        dev_loader=dev_dataloader,
        test_loader=test_dataloader,
        save_ckpt=cfg.save_ckpt,
        batch_size=cfg.batch_size,
        max_epoch=cfg.max_epoch,
        lr=cfg.lr,
        PAD_ID=tokenizer.encode('<pad>')[0],
        warmup_steps=cfg.warmup_steps,
        # optimizer=optimizer,
        use_prefix=args.use_prefix,
        opt="adamw"
    )
    
    framework.train_model(args.name)
    framework.model.load_state_dict(jittor.load(cfg.save_ckpt))
    framework.generate(save=True,dataloader=framework.test_loader, output_dir='./{}.txt'.format(args.output), decode_strategy='top-p', temperature=1.0, top_p=0.9, top_k=40, maxlen=256, verbose=False)

if __name__ == '__main__':
    run()

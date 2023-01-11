import argparse
from omegaconf import OmegaConf
import sys
sys.path.append('.')
from utils.dataloader import *
from utils.pretrained import *
from utils.pipeline import *
from transformers import AutoTokenizer
from gpt2.config import ModelConfig
from gpt2.model import TfmrLMHeadModel
from model import PrefixGPTModel
from framework.framework import FrameWork
import json

parser = argparse.ArgumentParser()
parser.add_argument('--name',type=str, default="run")
parser.add_argument('--task',type=str, default="webnlg")
parser.add_argument('--output',type=str, default="output")
parser.add_argument('--config',type=str, default="baseline.yaml")
parser.add_argument('--model_dict', type=str, default="test.pth")
parser.add_argument('--use_prefix', type=int, default=1)
args = parser.parse_args()

def run():
    cfg = OmegaConf.load(args.config)

    if args.task == "webnlg":
        test_dataset = get_test_dataset(cfg.dataset)

    model_dict = jittor.load(args.model_dict)

    tokenizer = AutoTokenizer.from_pretrained('gpt2-large')
    tokenizer.add_special_tokens({'pad_token': '<pad>'})

    if args.task == "webnlg":
        test_dataloader = get_webnlg_eval_dataloader(dataset=test_dataset,template=None, batch_size=cfg.batch_size, max_length=cfg.max_length, tokenizer=tokenizer, PAD_ID='<pad>', EOS_ID='<|endoftext|>')
    elif args.task == "e2e":
        test_dataloader = get_e2e_eval_dataloader(path="./data/e2e_data/src1_test.txt",template=None, batch_size=cfg.batch_size, max_length=cfg.max_length, tokenizer=tokenizer, PAD_ID='<|endoftext|>', EOS_ID='<|endoftext|>')
    else:
        print("Task not supported.")
        exit(0)

    jittor.flags.use_cuda = 1 # 0->cpu, 1->try to use gpu, 2->force using gpu

    with open('config.json','r') as fin:    
        model_config = json.load(fin)
        config = ModelConfig(**model_config)

    if args.use_prefix:
        pretrain_model = PrefixGPTModel(config=config)
    else:
        pretrain_model = TfmrLMHeadModel(config)

    pretrain_model.resize_token_embeddings(len(tokenizer))
    pretrain_model.load_state_dict(model_dict)

    framework = FrameWork(
        model=pretrain_model,
        tokenizer=tokenizer,
        train_loader=None,
        dev_loader=None,
        test_loader=test_dataloader,
        save_ckpt=cfg.save_ckpt,
        batch_size=cfg.batch_size,
        max_epoch=cfg.max_epoch,
        lr=cfg.lr,
        PAD_ID=tokenizer.encode('<pad>')[0],
        warmup_steps=cfg.warmup_steps,
        optimizer=None,
        opt="use_own" if args.use_prefix else "adamw"
    )

    framework.generate(dataloader=test_dataloader, output_dir=args.output+".txt", batch_size=cfg.batch_size, decode_strategy=cfg.decode_strategy, temperature=cfg.temperature, top_p=cfg.top_p, top_k=cfg.top_k, maxlen=cfg.max_length, verbose=False, use_prefix=args.use_prefix)


if __name__ == '__main__':
    run()

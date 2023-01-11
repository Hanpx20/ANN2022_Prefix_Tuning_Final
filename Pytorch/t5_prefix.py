import argparse
import torch
import numpy as np
from openprompt.data_utils.conditional_generation_dataset import WebNLGProcessor
from openprompt.utils.metrics import generation_metric
from openprompt.prompts.prefix_tuning_template import PrefixTuningTemplate
from openprompt.plms import load_plm
from openprompt import PromptDataLoader, PromptForGeneration
from transformers import AdamW
from transformers.optimization import get_linear_schedule_with_warmup

parser = argparse.ArgumentParser("")
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--plm_eval_mode", action="store_true")
parser.add_argument("--model", type=str, default='t5')  # tested model are gpt2/t5
parser.add_argument("--model_name_or_path", default='t5-large')
args = parser.parse_args()
print(args)

dataset = {}
dataset['train'] = WebNLGProcessor().get_train_examples("../../ANN2022_Prefix_Tuning/prefix/data/webnlg_2017/")
dataset['validation'] = WebNLGProcessor().get_dev_examples("../../ANN2022_Prefix_Tuning/prefix/data/webnlg_2017/")
dataset['test'] = WebNLGProcessor().get_test_examples("../../ANN2022_Prefix_Tuning/prefix/data/webnlg_2017/")

def set_seed(seed):
    torch.manual_seed(seed)  
    if torch.cuda.is_available():  # 固定随机种子（GPU)
        torch.cuda.manual_seed(seed)  # 为当前GPU设置
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置
    np.random.seed(seed) 

set_seed(42)


plm, tokenizer, model_config, WrapperClass = load_plm('gpt2', 'gpt2-medium')

mytemplate = PrefixTuningTemplate(model=plm, tokenizer=tokenizer, text=' {"placeholder":"text_a"} {"special": "<eos>"} {"mask"} ')


train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=256,
    batch_size=5,shuffle=False, teacher_forcing=True, predict_eos_token=True, # be sure to pass predict_eos_token=True if your template doesn't contain one, or you model may fail to stop generation.
    truncate_method="head")

validation_dataloader = PromptDataLoader(dataset=dataset["validation"], template=mytemplate, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=256,
    batch_size=5,shuffle=False, teacher_forcing=False, predict_eos_token=True,
    truncate_method="head")

test_dataloader = PromptDataLoader(dataset=dataset["test"], template=mytemplate, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=256,
    batch_size=5,shuffle=False, teacher_forcing=False, predict_eos_token=True,
    truncate_method="head")


use_cuda = True
prompt_model = PromptForGeneration(plm=plm,template=mytemplate, freeze_plm=True,tokenizer=tokenizer, plm_eval_mode=args.plm_eval_mode)
if use_cuda:
    prompt_model=  prompt_model.cuda()


no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
{
    "params": [p for n, p in prompt_model.named_parameters() if (not any(nd in n for nd in no_decay))],
    "weight_decay": 0.0,
},
{
    "params": [p for n, p in prompt_model.named_parameters() if any(nd in n for nd in no_decay)],
    "weight_decay": 0.0,
},
]

optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=1e-8)


tot_step  = len(train_dataloader)*5
scheduler = get_linear_schedule_with_warmup(optimizer, 0, tot_step)


def evaluate(prompt_model, dataloader):
    generated_sentence = []
    groundtruth_sentence = []
    prompt_model.eval()
    from tqdm import tqdm
    t = tqdm(dataloader)
    for step, inputs in enumerate(t):
        if use_cuda:
            inputs = inputs.cuda()
        _, output_sentence = prompt_model.generate(inputs, **generation_arguments)
        generated_sentence.extend(output_sentence)
        groundtruth_sentence.extend(inputs['tgt_text'])
    return generated_sentence

generation_arguments = {
    "max_length": 512,
    "max_new_tokens": None,
    "min_length": 5,
    "temperature": 1.0,
    "do_sample": False,
    "top_k": 0,
    "top_p": 0.9,
    "repetition_penalty": 1.0,
    "num_beams": 5,
    "bad_words_ids": [[628], [198]]
}

global_step = 0
tot_loss = 0
log_loss = 0


for epoch in range(5):
    prompt_model.train()
    from tqdm import tqdm
    t = tqdm(train_dataloader)
    for step, inputs in enumerate(t):
        global_step +=1
        if use_cuda:
            inputs = inputs.cuda()
        loss = prompt_model(inputs)
        loss.backward()
        tot_loss += loss.item()
        t.set_postfix({'loss':tot_loss/global_step})
        torch.nn.utils.clip_grad_norm_(mytemplate.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        if global_step %500 ==0:
            print("Epoch {}, global_step {} average loss: {} lr: {}".format(epoch, global_step, (tot_loss-log_loss)/500, scheduler.get_last_lr()[0]), flush=True)
            log_loss = tot_loss

generated_sentence = evaluate(prompt_model, test_dataloader)

with open(f"./outputs/t5.txt",'w') as f:
    for i in generated_sentence:
        f.write(i+"\n")


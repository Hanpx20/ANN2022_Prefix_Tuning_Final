import copy
import json

import jittor
import numpy as np
from openprompt.data_utils.conditional_generation_dataset import WebNLGProcessor 

def get_train_dataset(dataset_name):
    if(dataset_name == 'webnlg'):
        return WebNLGProcessor().get_train_examples('./data/webnlg_2017/')
    
def get_dev_dataset(dataset_name):
    if(dataset_name == 'webnlg'):
        return WebNLGProcessor().get_dev_examples("./data/webnlg_2017/")

def get_test_dataset(dataset_name):
    if(dataset_name == 'webnlg'):
        return WebNLGProcessor().get_test_examples("./data/webnlg_2017/")

# def _webnlg_table_trans(text):
    
def pad(text, PAD_ID, input_length, max_length):
    if input_length < max_length:
        pad_len = max_length - input_length
        for i in range(pad_len):
            text = text + PAD_ID
    return text

def get_attention_mask(pad_input_ids, PAD_TOKEN):
    shape = [len(pad_input_ids)]
    pad_seq = np.ones(shape) * PAD_TOKEN[0]
    return 1 - np.equal(pad_input_ids, pad_seq).astype(np.int)

def gen_loss_ids(text_a_len, text_b_len, max_length):
    text_a_pad = jittor.zeros([text_a_len]).int32()
    if(text_a_len + text_b_len < max_length):
        text_b_pad = jittor.ones([text_b_len]).int32()
        pad = jittor.zeros([max_length - text_a_len - text_b_len]).int32()
        return jittor.concat([text_a_pad, text_b_pad, pad], dim=0)
    else:
        text_b_pad = jittor.ones([max_length - text_a_len]).int32()
        return jittor.concat([text_a_pad, text_b_pad], dim=0)

class DataLoader(jittor.dataset.Dataset):
    def __init__(self, trans_dataset):
        super().__init__()
        self.set_attrs(total_len=len(trans_dataset))
        self.trans_dataset = trans_dataset

    def __getitem__(self, k):
        return {
            'input_ids':self.trans_dataset[k]['input_ids'],
            # 'gen_input_ids': jittor.Var(eval_input_ids),
            'tgt_text':self.trans_dataset[k]['tgt_text'],
            'loss_ids':self.trans_dataset[k]['loss_ids'],
            'attention_mask':self.trans_dataset[k]['attention_mask'],
            'input_ids_len':self.trans_dataset[k]['input_ids_len'],
        }

class EvalDataLoader(jittor.dataset.Dataset):
    def __init__(self, trans_dataset):
        super().__init__()
        self.set_attrs(total_len=len(trans_dataset))
        self.trans_dataset = trans_dataset

    def __getitem__(self, k):
        return {
            'input_ids':self.trans_dataset[k]['input_ids'],
            'tgt_text':self.trans_dataset[k]['tgt_text'],
            'loss_ids':self.trans_dataset[k]['loss_ids'],
            'attention_mask':self.trans_dataset[k]['attention_mask'],
            'input_ids_len':self.trans_dataset[k]['input_ids_len'],
        }


def get_webnlg_train_dataloader(dataset, template, batch_size, max_length, tokenizer, PAD_ID, EOS_ID, shuffle, input_limit = 158):
    trans_dataset = []
    for data in dataset:
        # deal with instances with multiple target texts
        if('\n' in data.tgt_text):
            tgt_text = data.tgt_text.split('\n')[0]
        else:
            tgt_text = data.tgt_text

        # text_a + tgt_txt
        train_input_text = data.text_a + EOS_ID + ' ' +  tgt_text + ' ' + EOS_ID
        train_input_ids = tokenizer(train_input_text, add_special_tokens=True, is_split_into_words=False)['input_ids']
        
        # text_a
        text_a_ids = tokenizer(data.text_a + EOS_ID)['input_ids']
        text_a_ids_len = len(text_a_ids)

        # length limit
        if(text_a_ids_len > max_length):
            continue
        if(len(train_input_ids) > max_length and text_a_ids_len > input_limit):
            train_input_ids = train_input_ids[:input_limit-2] + train_input_ids[text_a_ids_len-2:]
            text_a_ids_len = input_limit
        if(len(train_input_ids) > max_length):
            train_input_ids = train_input_ids[:max_length]

        # loss_ids
        loss_ids = gen_loss_ids(text_a_ids_len, len(train_input_ids) - text_a_ids_len, max_length)

        # input_ids_len
        input_ids_len = len(train_input_ids)

        # pad
        pad_input_ids = pad(train_input_ids, tokenizer.encode(PAD_ID), input_ids_len, max_length)
        pad_first_eos_idx = pad_input_ids.index(tokenizer.encode(EOS_ID)[0]) + 1
        pad_label_ids = copy.deepcopy(pad_input_ids)
        pad_label_ids[0:pad_first_eos_idx] = [tokenizer.pad_token_id] * pad_first_eos_idx

        for i in range(0, len(pad_label_ids)):
           if pad_label_ids[i] == tokenizer.pad_token_id:
               pad_label_ids[i] = -100

        attention_mask = get_attention_mask(pad_input_ids, tokenizer.encode(PAD_ID))

        data_dict={
            'input_ids':jittor.Var(pad_input_ids),
            'tgt_text':tgt_text,
            'loss_ids':loss_ids,
            'attention_mask': attention_mask,
            'input_ids_len': jittor.Var(text_a_ids_len),
        }
        trans_dataset.append(data_dict)
    return DataLoader(
        trans_dataset,
    ).set_attrs(batch_size=batch_size, shuffle=shuffle)

def get_e2e_train_dataloader(path, template, batch_size, max_length, tokenizer, PAD_ID, EOS_ID, shuffle):
    trans_dataset = []
    with open(path, "r") as file:
        for idx, line in enumerate(file):
            x = line.split("||")
            tgt_text = x[1]
            train_input_text = ' ' + x[0] + ' ' + EOS_ID + ' ' + tgt_text + EOS_ID
            train_input_ids = tokenizer.encode(train_input_text)

            eval_input_ids = tokenizer.encode(' ' + x[0] + ' ' + EOS_ID)
            tgt_ids = tokenizer.encode(' ' + tgt_text + EOS_ID)

            if(len(eval_input_ids) >= max_length):
                continue
            
            loss_ids = gen_loss_ids(len(eval_input_ids), len(tgt_ids), max_length)

            if len(train_input_ids) < max_length:
                input_ids_len = len(train_input_ids)
            else:
                input_ids_len = max_length

            pad_input_text = pad(train_input_text, PAD_ID, input_ids_len, max_length)
            pad_input_ids = tokenizer.encode(pad_input_text)[0:max_length]
            pad_first_eos_idx = pad_input_ids.index(tokenizer.encode(EOS_ID)[0]) + 1
            pad_label_ids = copy.deepcopy(pad_input_ids)
            pad_label_ids[0:pad_first_eos_idx] = [tokenizer.pad_token_id] * pad_first_eos_idx

            for i in range(0, len(pad_label_ids)):
                if pad_label_ids[i] == tokenizer.pad_token_id:
                    pad_label_ids[i] = -100

            attention_mask = get_attention_mask(pad_input_ids, tokenizer.encode(PAD_ID))
            
            data_dict={
                'input_ids':jittor.Var(pad_input_ids),
                # 'gen_input_ids': jittor.Var(eval_input_ids),
                'tgt_text':tgt_text,
                'loss_ids':jittor.Var(loss_ids),
                'attention_mask':jittor.Var(attention_mask),
                'input_ids_len':input_ids_len,
                'labels': jittor.Var(pad_label_ids)
            }
            trans_dataset.append(data_dict)
            '''
            for i in data_dict:
                print(i)
                print(data_dict[i])
            exit(0)
            '''
    return DataLoader(trans_dataset).set_attrs(batch_size=batch_size, shuffle=shuffle)



def get_e2e_eval_dataloader(path, template, batch_size, max_length, tokenizer, PAD_ID, EOS_ID, shuffle):
    trans_dataset = []
    with open(path, "r") as file:
        for line in file:
            x = line.split("||")
            tgt_text = x[1]
            train_input_text = ' ' + x[0] + ' ' + EOS_ID + ' ' + tgt_text + EOS_ID
            train_input_ids = tokenizer.encode(train_input_text)

            eval_input_ids = tokenizer.encode(' ' + x[0] + ' ' + EOS_ID)
            tgt_ids = tokenizer.encode(' ' + tgt_text + EOS_ID)

            if(len(eval_input_ids) >= max_length):
                continue
            
            loss_ids = gen_loss_ids(len(eval_input_ids), len(tgt_ids), max_length)

            if len(train_input_ids) < max_length:
                input_ids_len = len(train_input_ids)
            else:
                input_ids_len = max_length

            pad_input_text = pad(train_input_text, PAD_ID, input_ids_len, max_length)
            pad_input_ids = tokenizer.encode(pad_input_text)[0:max_length]
            pad_first_eos_idx = pad_input_ids.index(tokenizer.encode(EOS_ID)[0]) + 1
            pad_label_ids = copy.deepcopy(pad_input_ids)
            pad_label_ids[0:pad_first_eos_idx] = [tokenizer.pad_token_id] * pad_first_eos_idx

            for i in range(0, len(pad_label_ids)):
                if pad_label_ids[i] == tokenizer.pad_token_id:
                    pad_label_ids[i] = -100

            attention_mask = get_attention_mask(pad_input_ids, tokenizer.encode(PAD_ID))

            data_dict={
                'input_ids':jittor.Var(pad_input_ids),
                'gen_input_ids': jittor.Var(eval_input_ids),
                'tgt_text':tgt_text,
                'loss_ids':jittor.Var(loss_ids),
                'attention_mask':jittor.Var(attention_mask),
                'input_ids_len':input_ids_len,
                'labels': jittor.Var(pad_label_ids)
            }
            trans_dataset.append(data_dict)
        
    return EvalDataLoader(trans_dataset).set_attrs(batch_size=1, shuffle=shuffle)

def get_AP_train_dataloader(path, template, batch_size, max_length, tokenizer, PAD_ID, EOS_ID, shuffle):
    trans_dataset = []
    with open(path, "r") as file:
        for idx, line in enumerate(file):
            x = json.loads(line)
            text = ""
            for i in x:
                if i != "TEXT":
                    text += i + " : "
                    if isinstance(x[i], list):
                        text += x[i][0]['mainsnak']
                    else:
                        text += x[i]
                    text += " | "
            tgt_text = ""
            for word in x['TEXT'][0]:
                tgt_text += word + " "


            train_input_text = ' ' + text + ' ' + EOS_ID + ' ' + tgt_text + EOS_ID
            train_input_ids = tokenizer.encode(train_input_text)

            eval_input_ids = tokenizer.encode(' ' + text + ' ' + EOS_ID)
            tgt_ids = tokenizer.encode(' ' + tgt_text + EOS_ID)

            if (len(eval_input_ids) >= max_length):
                continue

            loss_ids = gen_loss_ids(len(eval_input_ids), len(tgt_ids), max_length)

            if len(train_input_ids) < max_length:
                input_ids_len = len(train_input_ids)
            else:
                input_ids_len = max_length

            pad_input_text = pad(train_input_text, PAD_ID, input_ids_len, max_length)
            pad_input_ids = tokenizer.encode(pad_input_text)[0:max_length]
            pad_first_eos_idx = pad_input_ids.index(tokenizer.encode(EOS_ID)[0]) + 1
            pad_label_ids = copy.deepcopy(pad_input_ids)
            pad_label_ids[0:pad_first_eos_idx] = [tokenizer.pad_token_id] * pad_first_eos_idx

            for i in range(0, len(pad_label_ids)):
                if pad_label_ids[i] == tokenizer.pad_token_id:
                    pad_label_ids[i] = -100

            attention_mask = get_attention_mask(pad_input_ids, tokenizer.encode(PAD_ID))

            data_dict = {
                'input_ids': jittor.Var(pad_input_ids),
                # 'gen_input_ids': jittor.Var(eval_input_ids),
                'tgt_text': tgt_text,
                'loss_ids': jittor.Var(loss_ids),
                'attention_mask': jittor.Var(attention_mask),
                'input_ids_len': input_ids_len,
                'labels': jittor.Var(pad_label_ids)
            }
            trans_dataset.append(data_dict)
    return DataLoader(trans_dataset).set_attrs(batch_size=batch_size, shuffle=shuffle)

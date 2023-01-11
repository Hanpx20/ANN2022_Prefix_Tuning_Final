from gpt2.model import TfmrLMHeadModel
import jittor
import jittor.nn as nn
import numpy as np

def masked_select(arr, masks):
    return_arr = []
    for i, mask in enumerate(masks):
        if mask[0].item():
            return_arr.append(i)

    return jittor.Var(return_arr)


def float32_min():
    return np.finfo(np.float32).min

class PrefixEncoder(jittor.nn.Module):
    def __init__(self, seq_len, dim_ebd, num_layer):
        super().__init__()
        self.embedding = jittor.nn.Embedding(seq_len, dim_ebd)
        self.mid_dim = dim_ebd * 4
        self.trans = jittor.nn.Sequential(
            jittor.nn.Linear(dim_ebd, self.mid_dim),
            jittor.nn.Tanh(),
            jittor.nn.Linear(self.mid_dim, num_layer * 2 * dim_ebd)
        )

    def execute(self, prefix):
        prefix_tokens = self.embedding(prefix)
        past_key_values = self.trans(prefix_tokens)
        return past_key_values


class PrefixGPTModel(jittor.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.base_gpt_model = TfmrLMHeadModel(config)

        self.pre_seq_len = 5
        self.num_layer = config.num_hidden_layers
        self.dim_ebd = config.hidden_size
        self.num_head = config.num_attention_heads

        self.dropout = jittor.nn.Dropout(0.3)
        self.prefix_encoder = PrefixEncoder(self.pre_seq_len, self.dim_ebd, self.num_layer)
        self.prefix_tokens = jittor.arange(self.pre_seq_len).int64()


    def set_gradient(self):
        for param in self.base_gpt_model.parameters():
            param.requires_grad = False

        for param in self.prefix_encoder.parameters():
            param.requires_grad = True
    
    def resize_token_embeddings(self, new_num_tokens: int = None) -> None:
        self.base_gpt_model.resize_token_embeddings(new_num_tokens=new_num_tokens)

    def get_prompt(self, bsz=None):
        prefix_tokens = self.prefix_tokens.unsqueeze(
            0).expand(bsz, -1)
        past_key_values = self.prefix_encoder(prefix_tokens)
        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(bsz, seqlen, self.num_layer * 2, self.num_head,
                                               self.dim_ebd//self.num_head)
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    def execute(self, input_ids, labels=None, PAD_ID=None, attention_mask=None, past_key_values=None, use_cache=False, generate=False):
        bsz = input_ids.shape[0] # batch size

        if past_key_values is None:
            past_key_values = self.get_prompt(bsz)
        if not generate:
            prefix_attention_mask = jittor.ones(bsz, self.pre_seq_len).view(bsz, -1)
            attention_mask = jittor.concat((prefix_attention_mask, attention_mask), dim=1).view(bsz, -1)
        
        outputs = self.base_gpt_model(
            input_ids=input_ids, labels=labels, PAD_ID=PAD_ID, past_key_values=past_key_values, attention_mask=attention_mask, use_cache=use_cache)

        return outputs

    def generate(self,input_ids, tokenizer, PAD_ID, maxlen, decode_strategy, temperature, top_p=1.0, top_k=50267):
        EOS_ID = tokenizer.eos_token_id
        self.eval()
        allgen = []
        with jittor.no_grad():
            past_key_values = None
            output_ids = None
            for _ in range(maxlen):
                outputs = self(input_ids, past_key_values=past_key_values, use_cache=True, attention_mask=None, generate=True)
                logits = outputs["logits"]
                past_key_values = outputs["past_key_values"]
                logits = logits[:, -1, :] / temperature
                if decode_strategy == "top-p":
                    p_star = nn.softmax(logits,dim=-1)
                    sort_p, sort_indices = jittor.argsort(p_star, descending=True)
                    batch_size, length = sort_indices.shape[0], sort_indices.shape[1]
                    sum_p = jittor.misc.cumsum(sort_p,dim=-1)
                    
                    _sort_p_save_mask = sum_p < top_p
                    _sort_p_save_mask[:, 1:] = _sort_p_save_mask[:, :-1].clone()
                    _sort_p_save_mask[:, 0] = 1
                    delete_mask = 1 - _sort_p_save_mask.float32()
                    sort_indices = (sort_indices + jittor.arange(sort_indices.shape[0]).int64().unsqueeze(-1) * sort_indices.shape[1])
                    select_indices = masked_select(sort_indices, delete_mask.bool())
                    logits = logits.view(-1) 
                    logits = jittor.misc.index_fill_(logits, 0, select_indices, -float('inf'))
                    logits = logits.view(batch_size, length)
                elif decode_strategy == "top-k":
                    p_star = nn.softmax(logits, dim=-1)
                    sort_p, sort_indices = jittor.argsort(p_star, descending=True)
                    vocab_size = sort_p.shape[-1]
                    batch_size, length = sort_indices.shape[0], sort_indices.shape[1]

                    _top_k_ones = jittor.ones([batch_size,top_k])
                    _top_k_zeros = jittor.zeros([batch_size, vocab_size - top_k])
                    _sort_k_save_mask = jittor.concat([_top_k_ones,_top_k_zeros],dim=1)
                    
                    delete_mask = 1 - _sort_k_save_mask.float32()
                    sort_indices = (sort_indices + jittor.arange(sort_indices.shape[0]).int64().unsqueeze(-1) * sort_indices.shape[1])
                    
                    select_indices = masked_select(sort_indices, delete_mask.bool())

                    logits = logits.view(-1)  # (whole_length)
                    logits = jittor.misc.index_fill_(logits, 0, select_indices, -float('inf'))
                    logits = logits.view(batch_size, length)
                prob = logits.softmax(dim=-1) # shape: (batch_size, num_vocabs)
                now_token = jittor.multinomial(prob, 1)[:, :1] # shape: (batch_size)

                if output_ids is None:
                    output_ids = now_token
                else:
                    output_ids = jittor.concat([output_ids, now_token], 1)
                input_ids = now_token
            allgen += output_ids.cpu().numpy().tolist()
        
        pro_allgen = []
        for gen in allgen:
            pro_allgen.append([])
            for idx in gen:
                if idx == EOS_ID:
                    break
                pro_allgen[-1].append(idx)
        return pro_allgen



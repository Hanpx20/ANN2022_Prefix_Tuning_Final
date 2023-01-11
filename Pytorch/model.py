from gpt2.model import TfmrLMHeadModel
import torch
import torch.nn.functional as F

class PrefixEncoder(torch.nn.Module):
    def __init__(self, seq_len, dim_ebd, num_layer, device):
        super().__init__()
        self.embedding = torch.nn.Embedding(seq_len, dim_ebd)
        self.mid_dim = dim_ebd * 4
        self.trans = torch.nn.Sequential(
            torch.nn.Linear(dim_ebd, self.mid_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(self.mid_dim, num_layer * 2 * dim_ebd)
        ).to(device)

    def forward(self, prefix):
        prefix_tokens = self.embedding(prefix)
        past_key_values = self.trans(prefix_tokens)
        return past_key_values

class PrefixGPTModel(torch.nn.Module):
    def __init__(self, config, device, non_prefix_layers = []):
        super().__init__()
        self.base_gpt_model = TfmrLMHeadModel(config)

        self.pre_seq_len = 5
        self.num_layer = config.num_hidden_layers
        self.dim_ebd = config.hidden_size
        self.num_head = config.num_attention_heads
        self.device = device

        self.non_prefix_layers = non_prefix_layers

        self.dropout = torch.nn.Dropout(0.3)
        self.prefix_encoder = PrefixEncoder(self.pre_seq_len, self.dim_ebd, self.num_layer, device)
        self.prefix_tokens = torch.arange(self.pre_seq_len).long()

    def set_gradient(self):
        for param in self.base_gpt_model.transformer.parameters():
            param.requires_grad = False

    def resize_token_embeddings(self, new_num_tokens: int = None) -> None:
        self.base_gpt_model.resize_token_embeddings(new_num_tokens=new_num_tokens)

    def get_prompt(self, bsz=None):
        prefix_tokens = self.prefix_tokens.unsqueeze(
            0).expand(bsz, -1).to(self.device)
        past_key_values = self.prefix_encoder(prefix_tokens)
        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(bsz, seqlen, self.num_layer * 2, self.num_head,
                                               self.dim_ebd//self.num_head)
        past_key_values = self.dropout(past_key_values)
        past_key_values = list(past_key_values.permute([2, 0, 3, 1, 4]).split(2))
         
        for i in self.non_prefix_layers:
            past_key_values[i] = None
        return past_key_values

    def forward(self, input_ids, labels=None, PAD_ID=None, attention_mask=None, past_key_values=None, use_cache=False, generate=False):
        bsz = input_ids.shape[0] # batch size

        if past_key_values is None:
            past_key_values = self.get_prompt(bsz)
        if not generate:
            prefix_attention_mask = torch.ones(bsz, self.pre_seq_len).view(bsz, -1).to(self.device)
            attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1).view(bsz, -1)
        
        outputs = self.base_gpt_model(
            input_ids=input_ids, labels=labels, PAD_ID=PAD_ID, past_key_values=past_key_values, attention_mask=attention_mask, use_cache=use_cache)

        return outputs

    def generate(self,input_ids, tokenizer, device, PAD_ID, maxlen, decode_strategy, temperature, top_p=1.0, top_k=50267):
        EOS_ID = tokenizer.eos_token_id
        self.eval()
        allgen = []
        with torch.no_grad():
            past_key_values = None
            output_ids = None
            for _ in range(maxlen):
                outputs = self(input_ids, past_key_values=past_key_values, use_cache=True, attention_mask=None,generate=True)
                logits = outputs["logits"]
                past_key_values = outputs["past_key_values"]
                logits = logits[:, -1, :] / temperature
                if decode_strategy == "top-p":
                    p_star = F.softmax(logits,dim=-1)
                    sort_p, sort_indices = torch.sort(p_star, descending=True)
                    batch_size, length = sort_indices.shape[0], sort_indices.shape[1]
                    sum_p = torch.cumsum(sort_p,dim=-1)
                    
                    _sort_p_save_mask = sum_p < top_p
                    _sort_p_save_mask[:, 1:] = _sort_p_save_mask[:, :-1].clone()
                    _sort_p_save_mask[:, 0] = 1
                    delete_mask = 1 - _sort_p_save_mask.to(torch.float)
                    sort_indices = (sort_indices + torch.arange(sort_indices.shape[0],device=device,dtype=torch.long).unsqueeze(-1) * sort_indices.shape[1])
                    select_indices = torch.masked_select(sort_indices, delete_mask.bool())
                    logits = logits.view(-1) 
                    logits = torch.index_fill(logits, 0, select_indices, -float('inf'))
                    logits = logits.view(batch_size, length)
                elif decode_strategy == "top-k":
                    p_star = F.softmax(logits, dim=-1)
                    sort_p, sort_indices = torch.sort(p_star, descending=True)
                    vocab_size = sort_p.shape[-1]
                    batch_size, length = sort_indices.shape[0], sort_indices.shape[1]

                    _top_k_ones = torch.ones([batch_size,top_k])
                    _top_k_zeros = torch.zeros([batch_size, vocab_size - top_k])
                    _sort_k_save_mask = torch.cat([_top_k_ones,_top_k_zeros],dim=1).to(device)
                    
                    delete_mask = 1 - _sort_k_save_mask.to(torch.float)
                    sort_indices = (sort_indices + torch.arange(sort_indices.shape[0], device=device,
                                                                dtype=torch.long).unsqueeze(-1) * sort_indices.shape[1])
                    
                    select_indices = torch.masked_select(sort_indices, delete_mask.bool())

                    logits = logits.view(-1)  # (whole_length)
                    logits = torch.index_fill(logits, 0, select_indices, -float('inf'))
                    logits = logits.view(batch_size, length)
                prob = logits.softmax(dim=-1) # shape: (batch_size, num_vocabs)
                now_token = torch.multinomial(prob, 1)[:, :1] # shape: (batch_size)
                if output_ids == None:
                    output_ids = now_token
                else:
                    output_ids = torch.cat([output_ids, now_token], 1)
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
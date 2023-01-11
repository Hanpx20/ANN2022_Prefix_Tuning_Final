import jittor
import jittor.nn as nn
import numpy as np

ACT2FN = {
    "relu": nn.relu,
    "tanh": jittor.tanh,
    "linear": lambda x: x,
    "sigmoid": jittor.sigmoid,
    "gelu": nn.gelu,
}

def masked_select(arr, masks):
    return_arr = []
    for i, mask in enumerate(masks):
        if mask[0].item():
            return_arr.append(i)

    return jittor.Var(return_arr)


def float32_min():
    return np.finfo(np.float32).min


class TransposeLinear(nn.Module):
    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        w = jittor.normal(mean=0, std=0.02, size=(nx, nf))
        self.weight = w
        self.bias = jittor.zeros(nf)

    def execute(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = (x.view(-1, x.size(-1)) @ self.weight ) + self.bias
        x = x.view(*size_out)
        return x


class TfmrAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.max_positions = config.max_position_embeddings  # maxinum word length

        self.bias = jittor.tril(jittor.ones((self.max_positions, self.max_positions)).uint8()).view(1,
                                                                                                    1,
                                                                                                    self.max_positions,
                                                                                                    self.max_positions)
        self.masked_bias = jittor.Var(-10000)
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights
        self.c_attn = TransposeLinear(3 * self.embed_dim, self.embed_dim)
        self.c_proj = TransposeLinear(self.embed_dim, self.embed_dim)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        self.pruned_heads = set()

    def _attn(self, query, key, value, attention_mask=None):
        attn_weights = query @ key.transpose(2,3)
        if self.scale_attn_weights:
            attn_weights = attn_weights / jittor.float32(
                value.size(-1) ** 0.5
            )

        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length].bool()
        mask_value = float32_min()
        mask_value = jittor.float32(mask_value)
        causal_mask = causal_mask.expand(attn_weights.shape)
        attn_weights = jittor.where(causal_mask, attn_weights, mask_value)        


        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.Softmax(dim=-1)(attn_weights)
        attn_weights = self.attn_dropout(attn_weights)
        attn_output = attn_weights @ value

        return attn_output, attn_weights

    def _split_heads(self, tensor, num_heads, attn_head_size):
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)


    def _merge_heads(self, tensor, num_heads, attn_head_size):
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)
        
    def execute(
            self,
            hidden_states,
            layer_past=None,
            use_cache=False,
            attention_mask=None
    ):
        query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = jittor.concat((past_key, key), dim=-2)
            value = jittor.concat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        attn_output, attn_weights = self._attn(query, key, value, attention_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


class TfmrMLP(nn.Module):
    def __init__(self, intermediate_size, config):
        super().__init__()
        embed_dim = config.hidden_size
        self.c_fc = TransposeLinear(intermediate_size, embed_dim)
        self.c_proj = TransposeLinear(embed_dim, intermediate_size)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def execute(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class TfmrBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = TfmrAttention(config)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = TfmrMLP(inner_dim, config)

    def execute(
            self,
            hidden_states,
            layer_past=None,
            use_cache=False,
            attention_mask=None
    ):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            use_cache=use_cache,
            attention_mask=attention_mask
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]

        # TODO START
        # Bulid connecetions of different modules in the Tranformer block
        hidden_states = attn_output + residual  # add the residual connection
        hidden_states_2 = self.mlp(self.ln_2(hidden_states))
        hidden_states = hidden_states + hidden_states_2
        # TODO END

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs


class TfmrModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.initializer_range = config.initializer_range
        self.embed_dim = config.hidden_size

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)  # (词数，维数)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)
        self.wte.weight.requires_grad = True
        self.wpe.weight.requires_grad = True

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([TfmrBlock(config) for _ in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)


    def get_input_embeddings(self):
        return self.wte

    def resize_token_embeddings(self, new_num_tokens: int = None) -> nn.Embedding:
        model_embeds = self._resize_token_embeddings(new_num_tokens)
        if new_num_tokens is None:
            return model_embeds

        # Update base model and current model config
        self.vocab_size = new_num_tokens

        return model_embeds

    def _resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.get_input_embeddings()
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.wte = new_embeddings

        return self.get_input_embeddings()

    def _get_resized_embeddings(
        self, old_embeddings: nn.Embedding, new_num_tokens: int = None
    ) -> nn.Embedding:

        if new_num_tokens is None:
            return old_embeddings

        old_num_tokens, old_embedding_dim = old_embeddings.weight.size()

        if old_num_tokens == new_num_tokens:
            return old_embeddings

        if not isinstance(old_embeddings, nn.Embedding):
            raise TypeError(
                f"Old embeddings are of type {type(old_embeddings)}, which is not an instance of {nn.Embedding}. You"
                " should either use a different resize function or make sure that `old_embeddings` are an instance of"
                f" {nn.Embedding}."
            )

        # Build new embeddings
        new_embeddings = nn.Embedding(new_num_tokens, old_embedding_dim)

        # initialize all new embeddings (in particular added tokens)
        new_embeddings.weight.trunc_normal_(mean=0.0, std=self.initializer_range, a=-1000000, b=1000000)

        # Copy token embeddings from the previous weights

        # numbers of tokens to copy
        n = min(old_num_tokens, new_num_tokens)

        new_embeddings.weight[:n, :] = old_embeddings.weight[:n, :]

        new_embeddings.weight.requires_grad = old_embeddings.weight.requires_grad

        return new_embeddings

    def execute(
            self,
            input_ids,
            past_key_values=None,
            attention_mask=None,
            use_cache=None, 
    ):

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        batch_size = input_ids.shape[0]

        past_length = 0
        if past_key_values:
            past_length = past_key_values[0][0].size(-2)
        else:
            past_key_values = tuple([None] * len(self.h))
       
        position_id = jittor.arange(past_length, input_shape[1] + past_length)
        position_id = position_id.unsqueeze(0).view(-1, input_shape[-1])
        
        if attention_mask is not None:
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.float32()  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * float32_min()

        inputs_embeds = self.wte(input_ids)

        position_embeds = self.wpe(position_id)
        
        hidden_states = inputs_embeds + position_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        presents = () if use_cache else None
        all_self_attentions = ()
        all_cross_attentions = ()
        all_hidden_states = ()

        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            all_hidden_states = all_hidden_states + (hidden_states,)
            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                use_cache=use_cache
            )

            hidden_states = outputs[0]

            if use_cache is True:
                presents = presents + (outputs[1],)

            all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)

        # Add last hidden state
        all_hidden_states = all_hidden_states + (hidden_states,)

        return {
            "last_hidden_state": hidden_states,
            "past_key_values": presents,
            "hidden_states": all_hidden_states,
            "attentions": all_self_attentions,
            "cross_attentions": all_cross_attentions,
        }


class TfmrLMHeadModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.transformer = TfmrModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
    def resize_token_embeddings(self, new_num_tokens: int = None) -> None:
        self.transformer.resize_token_embeddings(new_num_tokens=new_num_tokens)
        self.lm_head = self.get_resized_lm_head(new_num_tokens)

    def get_resized_lm_head(self, new_num_tokens: int):
        old_lm_head = self.lm_head
        new_lm_head = nn.Linear(self.n_embd, new_num_tokens, bias=False)
        return new_lm_head

    def execute(
            self,
            input_ids,
            past_key_values=None,
            attention_mask=None,
            labels=None,
            use_cache=None,
            PAD_ID=None,
    ):

        transformer_outputs = self.transformer(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            use_cache=use_cache
        )
        hidden_states = transformer_outputs["last_hidden_state"]
        
        lm_logits = self.lm_head(hidden_states)

        return {
            "logits": lm_logits,
            "past_key_values": transformer_outputs["past_key_values"],
            "hidden_states": transformer_outputs["hidden_states"],
            "attentions": transformer_outputs["attentions"],
            "cross_attentions": transformer_outputs["cross_attentions"],
        }

    def generate(self,input_ids, tokenizer, PAD_ID, maxlen, decode_strategy, temperature, top_p=1.0, top_k=50267):
        EOS_ID = tokenizer.eos_token_id
        self.eval()
        allgen = []
        with jittor.no_grad():
            past_key_values = None
            output_ids = None
            for _ in range(maxlen):
                outputs = self(input_ids, past_key_values=past_key_values, use_cache=True)
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


class CrossEntropyLoss(nn.Module):
    def __init__(self, reduction):
        self.reduction = reduction

    def execute(self, output, target):
        return nn.cross_entropy_loss(output, target, reduction=self.reduction)


class GenerationModel(nn.Module):
    def __init__(self, model, freeze_plm, tokenizer, PAD_ID):
        super().__init__()
        self.tokenizer = tokenizer
        self.transformer = model
        self.loss_fct = CrossEntropyLoss(reduction='none')
        self.in_generation_function = False
        self.PAD_ID = PAD_ID

    def shift_logits_and_labels(self, logits, loss_ids, reference_ids):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_loss_ids = loss_ids[..., 1:].contiguous()
        shift_input_ids = reference_ids[..., 1:].contiguous()
        shift_input_ids = jittor.where(shift_loss_ids>0, shift_input_ids, jittor.int32(-100))
        return shift_logits, shift_input_ids

    def execute(self, batch):
        if self.in_generation_function:
            return self.transformer.execute(batch)
        else:
            return self._execute(batch)

    def _execute(self, batch):
        reference_ids = batch['input_ids']
        outputs = self.transformer(
            input_ids = batch['input_ids'],
            past_key_values = None,
            attention_mask = batch['attention_mask'],
            labels = None,
            use_cache = None,
            PAD_ID = self.PAD_ID
            )
        # TODO END
        logits = outputs['logits']
        logits, labels = self.shift_logits_and_labels(logits, batch['loss_ids'], reference_ids)
        batch_size, seq_len, vocab_size = logits.shape
        loss = self.loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss = loss.view(batch_size, -1).sum(dim=-1)
        loss = loss.mean()
        return loss

    def generate(self, batch, **generation_kwargs):
        batch_size, input_length = batch['input_ids'].size(0), batch['input_ids'].size(1)
        
        if 'input_ids_len' in batch:
            input_real_lens = batch['input_ids_len']
        else:
            input_real_lens = jittor.sum((batch['input_ids'] != self.tokenizer.pad_token_id).int32(), dim=-1)
        
        output_sequences = []
        for instance_id in range(batch_size):
            instance = {key: batch[key][instance_id:instance_id+1][:,:int(input_real_lens[instance_id])] for key in batch if isinstance(batch[key], jittor.Var) and batch[key].shape[:2]==[batch_size, input_length]}
            result = self.transformer.generate(instance['input_ids'], **generation_kwargs)[0]
            generate_sent = self.tokenizer.decode(result)
            generate_sent = self.post_processing(generate_sent)
            output_sequences.append(generate_sent)
        return output_sequences # , generated_sentences

    def post_processing(self, output):
        if output[0] == ' ':
            return output[1:]
        else:
            return output

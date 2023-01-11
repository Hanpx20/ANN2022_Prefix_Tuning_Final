from transformers import GPT2Config, GPT2Tokenizer, GPT2LMHeadModel

def load_pretrained(model_name):
    if(model_name=='gpt2-medium'):
        gpt2_model = GPT2LMHeadModel.from_pretrained(model_name)
        gpt2_tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        return gpt2_model, gpt2_tokenizer


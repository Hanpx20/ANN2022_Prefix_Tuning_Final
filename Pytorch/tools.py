
def auto_download_gpt():
    model = 'gpt2'
    model_name_or_path = 'gpt2-medium'
    save_path = 'gpt2.pth'
    from openprompt.plms import load_plm 
    import torch
    plm, tokenizer, model_config, WrapperClass = load_plm(model, model_name_or_path)
    torch.save(plm.state_dict(),save_path)

if __name__ == '__main__':
    auto_download_gpt()
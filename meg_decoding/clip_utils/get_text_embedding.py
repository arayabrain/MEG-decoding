import os
import torch


def get_language_model(prompt_dict:dict, savedir):
    if os.path.exists(os.path.join(savedir, 'text_features')):

        text_features = torch.load(os.path.join(savedir, 'text_features'))
        with open(os.path.join(savedir, 'prompts.txt'), 'r') as f:
            prompts = f.readlines()
    else:
        import clip
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)
        model = model.eval()
        prompts = []
        prefix = prompt_dict['prefix']
        for i, t in prompt_dict.items():
            if i == 'prefix':
                continue
            prompts.append(t+'\n')
        text = clip.tokenize([prefix + s.replace('\n','') for s in prompts]).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text)
        # with open(os.path.join(savedir, 'text_features'), 'wb') as f:
        torch.save(text_features, os.path.join(savedir, 'text_features'))
        with open(os.path.join(savedir, 'prompts.txt'), 'w') as f:
            f.writelines(prompts)
    return text_features, prompts

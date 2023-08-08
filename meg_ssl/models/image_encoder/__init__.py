from transformers import CLIPProcessor, CLIPModel
import torch
import clip

def get_image_encoder(model_name:str, parameters:dict):
    if model_name == 'vit_clip14':
        model = CLIPModel.from_pretrained(parameters['path'])
        preprocess = CLIPProcessor.from_pretrained(parameters['path'])
        return model, preprocess
    elif model_name == 'vit_clip16':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model, preprocess = clip.load("ViT-B/16", device=device)
        return model, preprocess
    else:
        raise NotImplementedError(f'{model_name} is not implemented')